import torch
import torch.nn as nn
import clip
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Union
import argparse
from tqdm import tqdm


class TextEncoder(nn.Module):
    """
    独立的文本编码器，使用冻结的CLIP模型
    从SimplifiedTextEncoder中提取出来
    """
    def __init__(self, clip_dim: int = 512, hidden_dim: int = 512, output_dim: int = 512):
        super().__init__()
        
        # 加载CLIP文本编码器（冻结参数）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 4层Transformer条件编码器（随机初始化）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.conditioning_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 投影层
        self.clip_projection = nn.Linear(clip_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        print("✓ Text encoder initialized with frozen CLIP")
        
    def forward(self, text_instructions: List[str]) -> torch.Tensor:
        """
        Args:
            text_instructions: 文本指令列表
        Returns:
            text_embeddings: (batch_size, output_dim)
        """
        device = next(self.parameters()).device
        
        # CLIP分词和编码
        text_tokens = clip.tokenize(text_instructions, truncate=True).to(device)
        
        with torch.no_grad():
            clip_features = self.clip_model.encode_text(text_tokens)  # (B, clip_dim)
        
        # 投影CLIP特征
        x = self.clip_projection(clip_features.float())  # (B, hidden_dim)
        x = x.unsqueeze(1)  # (B, 1, hidden_dim)
        
        # 应用条件变换器
        x = self.conditioning_encoder(x)  # (B, 1, hidden_dim)
        
        # 最终投影
        text_embeddings = self.output_projection(x.squeeze(1))  # (B, output_dim)
        
        return text_embeddings


def load_text_from_file(file_path: str) -> str:
    """从文件中加载文本内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return text
    except Exception as e:
        print(f"Error loading text from {file_path}: {e}")
        return ""


def extract_text_encodings_batch(text_encoder: TextEncoder, 
                                text_files: List[str], 
                                batch_size: int = 8) -> Dict[str, np.ndarray]:
    """
    批量提取文本编码
    
    Args:
        text_encoder: 文本编码器实例
        text_files: 文本文件路径列表
        batch_size: 批处理大小
        
    Returns:
        encodings: {filename: encoding_array} 字典
    """
    encodings = {}
    
    # 分批处理
    for i in tqdm(range(0, len(text_files), batch_size), desc="Extracting text encodings"):
        batch_files = text_files[i:i + batch_size]
        batch_texts = []
        batch_filenames = []
        
        # 加载批次中的所有文本
        for file_path in batch_files:
            text = load_text_from_file(file_path)
            if text:  # 只处理非空文本
                batch_texts.append(text)
                batch_filenames.append(os.path.basename(file_path))
        
        if not batch_texts:
            continue
            
        # 批量编码
        try:
            with torch.no_grad():
                embeddings = text_encoder(batch_texts)  # (batch_size, output_dim)
                embeddings_np = embeddings.cpu().numpy()
            
            # 存储结果
            for filename, embedding in zip(batch_filenames, embeddings_np):
                encodings[filename] = embedding
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            continue
    
    return encodings


def save_encodings(encodings: Dict[str, np.ndarray], 
                  output_dir: str, 
                  save_format: str = 'npy'):
    """
    保存编码到文件
    
    Args:
        encodings: 编码字典
        output_dir: 输出目录
        save_format: 保存格式 ('npy', 'json', 'both')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_format in ['npy', 'both']:
        # 保存为.npy文件
        npy_dir = output_path / 'npy'
        npy_dir.mkdir(exist_ok=True)
        
        for filename, encoding in tqdm(encodings.items(), desc="Saving .npy files"):
            # 移除原扩展名，添加.npy
            base_name = os.path.splitext(filename)[0]
            npy_path = npy_dir / f"{base_name}_text_encoding.npy"
            np.save(npy_path, encoding)
    
    if save_format in ['json', 'both']:
        # 保存为.json文件（用于调试和检查）
        json_dir = output_path / 'json'
        json_dir.mkdir(exist_ok=True)
        
        # 创建元数据
        metadata = {
            'num_files': len(encodings),
            'encoding_dim': list(encodings.values())[0].shape[0] if encodings else 0,
            'files': list(encodings.keys())
        }
        
        # 保存元数据
        with open(json_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved {len(encodings)} text encodings to {output_dir}")
        print(f"  - Encoding dimension: {metadata['encoding_dim']}")


def main():
    parser = argparse.ArgumentParser(description='Extract text encodings using frozen CLIP')
    parser.add_argument('input_dir', help='Directory containing text files')
    parser.add_argument('output_dir', help='Directory to save encodings')
    parser.add_argument('--file_extensions', nargs='+', default=['.txt', '.md'], 
                       help='File extensions to process (default: .txt .md)')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size for processing (default: 8)')
    parser.add_argument('--save_format', choices=['npy', 'json', 'both'], default='npy',
                       help='Save format (default: npy)')
    parser.add_argument('--output_dim', type=int, default=512,
                       help='Output embedding dimension (default: 512)')
    
    args = parser.parse_args()
    
    # 验证输入目录
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # 查找文本文件
    text_files = []
    for ext in args.file_extensions:
        text_files.extend(list(input_path.glob(f"*{ext}")))
        text_files.extend(list(input_path.rglob(f"*{ext}")))  # 递归搜索
    
    text_files = [str(f) for f in set(text_files)]  # 去重
    
    if not text_files:
        print(f"No text files found in {args.input_dir} with extensions {args.file_extensions}")
        return
    
    print(f"Found {len(text_files)} text files")
    
    # 初始化文本编码器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = TextEncoder(output_dim=args.output_dim)
    text_encoder.to(device)
    text_encoder.eval()
    
    print(f"Using device: {device}")
    
    # 提取编码
    encodings = extract_text_encodings_batch(
        text_encoder=text_encoder,
        text_files=text_files,
        batch_size=args.batch_size
    )
    
    if not encodings:
        print("No encodings were extracted")
        return
    
    # 保存编码
    save_encodings(encodings, args.output_dir, args.save_format)


if __name__ == "__main__":
    # 示例用法
    if len(os.sys.argv) == 1:
        print("Text Encoding Extractor")
        print("=" * 50)
        print("Usage examples:")
        print("  python extract_text_encodings.py ./text_files ./text_encodings")
        print("  python extract_text_encodings.py ./data/texts ./output/encodings --batch_size 16")
        print("  python extract_text_encodings.py ./texts ./out --save_format both --output_dim 768")
        print()
        print("Use --help for full options")
    else:
        main()