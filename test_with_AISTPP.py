#!/usr/bin/env python3
"""
Complete test script for AISTPP dataset with multi-feature support
"""

import os
import sys
import glob
import random
import pickle
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset.dance_dataset import AISTPPDataset
from EDGE import EDGE
from args import parse_test_opt
# 添加必要的导入 - 根据你的项目结构调整这些导入
# from models.edge import EDGE  # 你的模型文件
# from dataset.aistpp_dataset import AISTPPDataset  # 你的数据集文件
# from utils.audio_processing import slice_audio, juke_extract, baseline_extract  # 音频处理工具
# from utils.misc import stringintkey  # 工具函数



def stringintkey(x):
    """排序用的key函数，处理文件名中的数字"""
    import re
    numbers = re.findall(r'\d+', os.path.basename(x))
    return [int(n) for n in numbers] if numbers else [0]


def slice_audio(wav_file, segment_length, overlap, output_dir):
    """
    切片音频文件 - 这是一个示例实现，你需要根据实际情况调整
    """
    # 这里需要你的实际音频切片实现
    # 或者导入你现有的函数
    pass


def juke_extract(audio_file):
    """
    提取jukebox特征 - 示例实现
    """
    # 这里需要你的实际特征提取实现
    # 返回特征和其他信息的元组
    pass


def beat_extract(audio_file):
    """
    提取beat特征 - 示例实现
    """
    # 这里需要你的实际beat特征提取实现
    pass


def text_extract(audio_file):
    """
    提取text特征 - 示例实现
    """
    # 这里需要你的实际text特征提取实现
    pass


def test_with_dataset(opt):
    """使用AISTPPDataset进行测试"""
    print("Testing with AISTPPDataset...")
    
    # 加载normalizer（如果训练时保存了的话）
    
    # 创建测试数据集
    test_dataset = AISTPPDataset(
        data_path=opt.feature_cache_dir,
        backup_path='',
        train=False,  # 使用测试集
        feature_type=opt.feature_type,
        data_len=-1,
        include_contacts=True,
        force_reload=False
    )
    
    print(f"Test dataset loaded with {len(test_dataset)} samples")
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,  # 测试时通常batch_size=1
        shuffle=True,  # 随机打乱
        num_workers=0
    )
    
    # 加载模型
    print(f"Loading model from {opt.checkpoint}")
    model = EDGE(opt.feature_type, opt.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 创建输出目录
    Path(opt.render_dir).mkdir(parents=True, exist_ok=True)
    if opt.save_motions:
        Path(opt.motion_save_dir).mkdir(parents=True, exist_ok=True)
    
    fk_out = opt.motion_save_dir if opt.save_motions else None
    
    print("Generating dances from test dataset...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            
            print(f"Processing sample {i+1}/{min(opt.max_test_samples, len(test_dataset))}")
            
            # 解包batch数据
            pose, juke_feature, beat_feature, text_feature, filename_juke, wav_path = batch
            
            # 移动到设备
            pose = pose.to(device) if pose is not None else None
            juke_feature = juke_feature.to(device)
            beat_feature = beat_feature.to(device)
            text_feature = text_feature.to(device)

            # 构造data_tuple
            data_tuple = (pose, juke_feature, beat_feature, text_feature, filename_juke, wav_path)
            
            # 生成舞蹈
            try:
                model.render_sample(
                    data_tuple, "test", opt.render_dir, 
                    render_count=i, fk_out=fk_out, render=not opt.no_render
                )
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    print("Done with dataset testing!")


def test_with_multi_features(opt):
    """使用多特征的原始测试方法"""
    print("Testing with multi-feature extraction...")
    
    sample_length = opt.out_length
    sample_size = int(sample_length / 2.5) - 1
    
    temp_dir_list = []
    all_cond_juke = []
    all_cond_beat = []
    all_cond_text = []
    all_filenames = []
    
    if opt.use_cached_features:
        print("Using precomputed features")
        
        # 特征路径设置
        jukebox_dir = os.path.join(opt.feature_cache_dir, "jukebox_feats")
        beat_dir = os.path.join(opt.feature_cache_dir, "beat_feats/npy") 
        text_dir = os.path.join(opt.feature_cache_dir, "text_encodings/npy")
        
        # 获取所有特征文件
        juke_files = sorted(glob.glob(f"{jukebox_dir}/*.npy"), key=stringintkey)
        beat_files = sorted(glob.glob(f"{beat_dir}/*.npy"), key=stringintkey)
        text_files = sorted(glob.glob(f"{text_dir}/*.npy"), key=stringintkey)
        
        print(f"Found {len(juke_files)} jukebox files, {len(beat_files)} beat files, {len(text_files)} text files")
        
        # 确保文件数量匹配
        assert len(juke_files) == len(beat_files) == len(text_files), \
            f"Feature file counts don't match: {len(juke_files)}, {len(beat_files)}, {len(text_files)}"
        
        # 随机选择一个chunk
        if len(juke_files) >= sample_size:
            rand_idx = random.randint(0, len(juke_files) - sample_size)
            juke_files = juke_files[rand_idx : rand_idx + sample_size]
            beat_files = beat_files[rand_idx : rand_idx + sample_size]
            text_files = text_files[rand_idx : rand_idx + sample_size]
        else:
            print(f"Warning: Not enough files ({len(juke_files)}) for requested sample size ({sample_size})")
            sample_size = len(juke_files)
        
        # 加载特征
        print(f"Loading {sample_size} feature files...")
        juke_cond_list = [np.load(x) for x in tqdm(juke_files)]
        beat_cond_list = [np.load(x) for x in tqdm(beat_files)]
        text_cond_list = [np.load(x) for x in tqdm(text_files)]
        
        all_cond_juke.append(torch.from_numpy(np.array(juke_cond_list)))
        all_cond_beat.append(torch.from_numpy(np.array(beat_cond_list)))
        all_cond_text.append(torch.from_numpy(np.array(text_cond_list)))
        all_filenames.append(juke_files)
        
    else:
        print("Computing features for input music")
        # 这部分需要你实现实时特征提取
        # 由于比较复杂，建议使用缓存特征的方式
        raise NotImplementedError("Real-time feature extraction not implemented. Please use --use_cached_features")
    
    # 加载模型
    print(f"Loading model from {opt.checkpoint}")
    model = EDGE(opt.feature_type, opt.checkpoint)
    model.to(opt.device)
    model.eval()
    
    # 创建输出目录
    Path(opt.render_dir).mkdir(parents=True, exist_ok=True)
    if opt.save_motions:
        Path(opt.motion_save_dir).mkdir(parents=True, exist_ok=True)
    
    fk_out = opt.motion_save_dir if opt.save_motions else None
    
    print("Generating dances...")
    for i in range(len(all_cond_juke)):
        print(f"Processing batch {i+1}/{len(all_cond_juke)}")
        
        # 移动到设备
        juke_cond = all_cond_juke[i].to(opt.device)
        beat_cond = all_cond_beat[i].to(opt.device)
        text_cond = all_cond_text[i].to(opt.device)
        
        # 构造data_tuple
        data_tuple = (None, juke_cond, beat_cond, text_cond, all_filenames[i])
        
        try:
            model.render_sample(
                data_tuple, "test", opt.render_dir, 
                render_count=i, fk_out=fk_out, render=not opt.no_render
            )
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue
    
    print("Done with multi-feature testing!")
    
    # 清理临时目录
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


def main():
    """主函数"""
    opt = parse_test_opt()
    
    
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print("Configuration:")
    for arg, value in sorted(vars(opt).items()):
        print(f"  {arg}: {value}")
    print()
    
    try:
        if opt.use_cached_features:
            test_with_dataset(opt)
        else:
            test_with_multi_features(opt)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
        raise
    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()