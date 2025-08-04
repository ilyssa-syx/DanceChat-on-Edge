import torch
import torch.nn as nn
import numpy as np
import librosa
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class BeatEncoder(nn.Module):
    """
    独立的节拍编码器，使用Librosa进行节拍检测
    从原始BeatEncoder中提取出来
    """
    def __init__(self, output_dim: int = 256, sr: int = 22050, hop_length: int = 512):
        super().__init__()
        self.output_dim = output_dim
        self.sr = sr
        self.hop_length = hop_length
        self.beat_projection = nn.Linear(2, output_dim)
        
        print(f"✓ Beat encoder initialized (sr={sr}, hop_length={hop_length}, output_dim={output_dim})")
        
    def extract_beats(self, audio_path: str, target_length: int = None) -> Dict[str, any]:
        """
        从音频文件提取节拍特征
        
        Args:
            audio_path: 音频文件路径
            target_length: 目标序列长度，如果为None则自动计算
            
        Returns:
            result: 包含编码和元数据的字典
        """
        try:
            # 1. 加载音频
            y, sr = librosa.load(audio_path, sr=self.sr)
            audio_duration = len(y) / sr
            
            if target_length is None:
                # 自动计算目标长度（每秒约4帧）
                target_length = max(int(audio_duration * 4), 10)
            
            print(f"Processing: {os.path.basename(audio_path)}")
            print(f"  Duration: {audio_duration:.2f}s, Target length: {target_length}")
            
            # 2. 节拍检测
            tempo, beats = librosa.beat.beat_track(
                y=y, 
                sr=sr, 
                hop_length=self.hop_length,
                units='time'
            )
            
            # 3. 强拍检测
            downbeat_times = self._detect_downbeats_advanced(y, sr, beats, tempo)
            
            # 4. 创建时间轴
            frame_times = np.linspace(0, audio_duration, target_length)
            
            # 5. 计算高斯临近度特征
            proximity_features = self._compute_gaussian_proximity(
                frame_times, beats, downbeat_times, sigma=0.1
            )
            
            # 6. 投影到嵌入空间
            proximity_tensor = torch.from_numpy(proximity_features).float()
            beat_embeddings = self.beat_projection(proximity_tensor)
            
            return {
                'embeddings': beat_embeddings.detach().numpy(),  # (target_length, output_dim)
                'metadata': {
                    'filename': os.path.basename(audio_path),
                    'duration': audio_duration,
                    'tempo': float(tempo),
                    'num_beats': len(beats),
                    'num_downbeats': len(downbeat_times),
                    'target_length': target_length,
                    'embedding_dim': self.output_dim
                },
                'raw_features': {
                    'beat_times': beats.tolist(),
                    'downbeat_times': downbeat_times.tolist(),
                    'frame_times': frame_times.tolist(),
                    'proximity_features': proximity_features.tolist()
                }
            }
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            # 返回零特征作为后备
            zero_embeddings = np.zeros((target_length or 100, self.output_dim))
            return {
                'embeddings': zero_embeddings,
                'metadata': {
                    'filename': os.path.basename(audio_path),
                    'duration': 0.0,
                    'tempo': 0.0,
                    'num_beats': 0,
                    'num_downbeats': 0,
                    'target_length': target_length or 100,
                    'embedding_dim': self.output_dim,
                    'error': str(e)
                },
                'raw_features': None
            }
    
    def _detect_downbeats_advanced(self, y, sr, beat_times, tempo):
        """改进的强拍检测方法"""
        try:
            # 方法1: 基于音频特征的强拍检测
            beat_strengths = []
            hop_length_sec = self.hop_length / sr
            
            for beat_time in beat_times:
                start_sample = max(0, int((beat_time - hop_length_sec/2) * sr))
                end_sample = min(len(y), int((beat_time + hop_length_sec/2) * sr))
                beat_segment = y[start_sample:end_sample]
                
                if len(beat_segment) > 0:
                    # RMS能量
                    rms = np.sqrt(np.mean(beat_segment**2))
                    
                    # 谱质心
                    if len(beat_segment) > self.hop_length//4:
                        stft = librosa.stft(beat_segment, hop_length=self.hop_length//4)
                        spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(stft), sr=sr)
                        centroid = np.mean(spectral_centroids) if len(spectral_centroids[0]) > 0 else 0
                    else:
                        centroid = 0
                    
                    # 组合特征
                    strength = rms * 0.7 + (centroid / 1000) * 0.3
                    beat_strengths.append(strength)
                else:
                    beat_strengths.append(0)
            
            # 方法2: 基于节拍间隔的强拍检测
            if len(beat_times) > 4:
                beat_intervals = np.diff(beat_times)
                median_interval = np.median(beat_intervals)
                
                # 常见拍号假设
                possible_bar_lengths = [4, 3, 6, 8]
                best_bar_length = 4  # 默认4/4拍
                
                for bar_len in possible_bar_lengths:
                    expected_bar_duration = bar_len * median_interval
                    if self._validate_bar_structure(beat_times, expected_bar_duration):
                        best_bar_length = bar_len
                        break
                
                # 基于拍子结构选择强拍
                downbeat_candidates = []
                for i in range(0, len(beat_times), best_bar_length):
                    if i < len(beat_strengths):
                        bar_end = min(i + best_bar_length, len(beat_times))
                        bar_strengths = beat_strengths[i:bar_end]
                        if bar_strengths:
                            strongest_idx = i + np.argmax(bar_strengths)
                            downbeat_candidates.append(beat_times[strongest_idx])
                
                if downbeat_candidates:
                    return np.array(downbeat_candidates)
            
            # 后备方案：每4拍选择能量最大的
            if len(beat_strengths) > 4:
                downbeat_candidates = []
                for i in range(0, len(beat_times), 4):
                    bar_end = min(i + 4, len(beat_times))
                    bar_strengths = beat_strengths[i:bar_end]
                    if bar_strengths:
                        strongest_idx = i + np.argmax(bar_strengths)
                        downbeat_candidates.append(beat_times[strongest_idx])
                return np.array(downbeat_candidates)
            else:
                return beat_times[::4] if len(beat_times) > 4 else beat_times[:1]
                
        except Exception as e:
            print(f"    Advanced downbeat detection failed: {e}")
            return beat_times[::4] if len(beat_times) > 4 else beat_times[:1]
    
    def _validate_bar_structure(self, beat_times, expected_bar_duration, tolerance=0.1):
        """验证假设的小节结构"""
        if len(beat_times) < 8:
            return False
        
        num_bars = int(beat_times[-1] / expected_bar_duration)
        matches = 0
        
        for bar_idx in range(1, num_bars):
            expected_time = bar_idx * expected_bar_duration
            closest_beat_time = beat_times[np.argmin(np.abs(beat_times - expected_time))]
            if abs(closest_beat_time - expected_time) < tolerance:
                matches += 1
        
        return matches / max(1, num_bars - 1) > 0.6
    
    def _compute_gaussian_proximity(self, frame_times, beat_times, downbeat_times, sigma=0.1):
        """计算高斯临近度特征"""
        n_frames = len(frame_times)
        beat_proximity = np.zeros(n_frames)
        downbeat_proximity = np.zeros(n_frames)
        
        for i, frame_time in enumerate(frame_times):
            # 节拍临近度
            if len(beat_times) > 0:
                distances = np.abs(beat_times - frame_time)
                min_distance = np.min(distances)
                beat_proximity[i] = np.exp(-(min_distance ** 2) / (2 * sigma ** 2))
            
            # 强拍临近度
            if len(downbeat_times) > 0:
                distances = np.abs(downbeat_times - frame_time)
                min_distance = np.min(distances)
                downbeat_proximity[i] = np.exp(-(min_distance ** 2) / (2 * sigma ** 2))
        
        return np.stack([beat_proximity, downbeat_proximity], axis=1)


def find_audio_files(input_dir: str, extensions: List[str]) -> List[str]:
    """查找音频文件"""
    input_path = Path(input_dir)
    audio_files = []
    
    for ext in extensions:
        audio_files.extend(list(input_path.glob(f"*{ext}")))
        audio_files.extend(list(input_path.rglob(f"*{ext}")))  # 递归搜索
    
    return [str(f) for f in set(audio_files)]


def extract_beat_encodings_batch(beat_encoder: BeatEncoder, 
                                audio_files: List[str],
                                target_length: int = None) -> Dict[str, Dict]:
    """
    批量提取节拍编码
    
    Args:
        beat_encoder: 节拍编码器实例
        audio_files: 音频文件路径列表
        target_length: 目标序列长度
        
    Returns:
        results: {filename: result_dict} 字典
    """
    results = {}
    
    for audio_file in tqdm(audio_files, desc="Extracting beat encodings"):
        filename = os.path.basename(audio_file)
        result = beat_encoder.extract_beats(audio_file, target_length)
        results[filename] = result
    
    return results


def save_beat_encodings(results: Dict[str, Dict], 
                       output_dir: str,
                       save_format: str = 'npy',
                       save_metadata: bool = True):
    """
    保存节拍编码
    
    Args:
        results: 提取结果字典
        output_dir: 输出目录
        save_format: 保存格式
        save_metadata: 是否保存元数据
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_format in ['npy', 'both']:
        # 保存编码为.npy文件
        npy_dir = output_path / 'npy'
        npy_dir.mkdir(exist_ok=True)
        
        for filename, result in tqdm(results.items(), desc="Saving .npy files"):
            base_name = os.path.splitext(filename)[0]
            npy_path = npy_dir / f"{base_name}_beat_encoding.npy"
            np.save(npy_path, result['embeddings'])
    
    if save_format in ['json', 'both'] or save_metadata:
        # 保存元数据
        json_dir = output_path / 'json'
        json_dir.mkdir(exist_ok=True)
        
        # 收集所有元数据
        all_metadata = {}
        for filename, result in results.items():
            all_metadata[filename] = result['metadata']
            
            # 可选：保存原始特征
            if result['raw_features'] is not None:
                base_name = os.path.splitext(filename)[0]
                raw_features_path = json_dir / f"{base_name}_raw_features.json"
                with open(raw_features_path, 'w') as f:
                    json.dump(result['raw_features'], f, indent=2)
        
        # 保存汇总元数据
        summary_metadata = {
            'num_files': len(results),
            'encoding_dim': list(results.values())[0]['metadata']['embedding_dim'] if results else 0,
            'files': all_metadata
        }
        
        with open(json_dir / 'beat_metadata.json', 'w') as f:
            json.dump(summary_metadata, f, indent=2)
    
    print(f"✓ Saved {len(results)} beat encodings to {output_dir}")
    
    # 打印统计信息
    successful = sum(1 for r in results.values() if 'error' not in r['metadata'])
    failed = len(results) - successful
    
    if successful > 0:
        avg_tempo = np.mean([r['metadata']['tempo'] for r in results.values() 
                           if 'error' not in r['metadata'] and r['metadata']['tempo'] > 0])
        print(f"  - Successfully processed: {successful}/{len(results)}")
        print(f"  - Average tempo: {avg_tempo:.1f} BPM")
    
    if failed > 0:
        print(f"  - Failed: {failed}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(description='Extract beat encodings from audio files')
    parser.add_argument('input_dir', help='Directory containing audio files')
    parser.add_argument('output_dir', help='Directory to save encodings')
    parser.add_argument('--audio_extensions', nargs='+', 
                       default=['.wav', '.mp3', '.flac', '.m4a', '.aac'],
                       help='Audio file extensions to process')
    parser.add_argument('--target_length', type=int, default=None,
                       help='Target sequence length (default: auto)')
    parser.add_argument('--sample_rate', type=int, default=22050,
                       help='Audio sample rate (default: 22050)')
    parser.add_argument('--output_dim', type=int, default=256,
                       help='Output embedding dimension (default: 256)')
    parser.add_argument('--save_format', choices=['npy', 'json', 'both'], default='npy',
                       help='Save format (default: npy)')
    parser.add_argument('--save_metadata', action='store_true',
                       help='Save metadata and raw features')
    
    args = parser.parse_args()
    
    # 验证输入目录
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # 查找音频文件
    audio_files = find_audio_files(args.input_dir, args.audio_extensions)
    
    if not audio_files:
        print(f"No audio files found in {args.input_dir} with extensions {args.audio_extensions}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # 初始化节拍编码器
    beat_encoder = BeatEncoder(
        output_dim=args.output_dim,
        sr=args.sample_rate
    )
    
    # 提取编码
    results = extract_beat_encodings_batch(
        beat_encoder=beat_encoder,
        audio_files=audio_files,
        target_length=args.target_length
    )
    
    if not results:
        print("No encodings were extracted")
        return
    
    # 保存编码
    save_beat_encodings(
        results=results,
        output_dir=args.output_dir,
        save_format=args.save_format,
        save_metadata=args.save_metadata
    )


if __name__ == "__main__":
    # 示例用法
    if len(os.sys.argv) == 1:
        print("Beat Encoding Extractor")
        print("=" * 50)
        print("Usage examples:")
        print("  python extract_beat_encodings.py ./audio_files ./beat_encodings")
        print("  python extract_beat_encodings.py ./music ./output --target_length 200")
        print("  python extract_beat_encodings.py ./songs ./out --save_metadata --save_format both")
        print()
        print("Use --help for full options")
    else:
        main()