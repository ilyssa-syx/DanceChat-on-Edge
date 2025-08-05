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
    文本编码器，只保留冻结的CLIP编码特征，不做额外随机投影
    """
    def __init__(self, clip_dim: int = 512):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 加载CLIP文本编码器，并冻结参数
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        print(f"✓ Text encoder initialized with frozen CLIP only (dim={clip_dim})")

    def forward(self, text_instructions: List[str]) -> torch.Tensor:
        """
        Args:
            text_instructions: 文本指令列表
        Returns:
            clip_features: (batch_size, clip_dim)
        """
        device = next(self.clip_model.parameters()).device
        # 分词并编码
        text_tokens = clip.tokenize(text_instructions, truncate=True).to(device)
        with torch.no_grad():
            clip_features = self.clip_model.encode_text(text_tokens)  # (B, clip_dim)
        return clip_features


def load_text_from_file(file_path: str) -> str:
    """从文件中加载文本内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading text from {file_path}: {e}")
        return ""


def extract_text_encodings_batch(
    text_encoder: TextEncoder,
    text_files: List[str],
    batch_size: int = 8
) -> Dict[str, np.ndarray]:
    """
    批量提取文本编码，仅输出CLIP原始特征
    """
    encodings = {}
    for i in tqdm(range(0, len(text_files), batch_size), desc="Extracting text encodings"):
        batch = text_files[i:i + batch_size]
        texts, names = [], []
        for fp in batch:
            txt = load_text_from_file(fp)
            if txt:
                texts.append(txt)
                names.append(os.path.basename(fp))
        if not texts:
            continue
        with torch.no_grad():
            feats = text_encoder(texts).cpu().numpy()
        for name, feat in zip(names, feats):
            encodings[name] = feat
    return encodings


def save_encodings(encodings: Dict[str, np.ndarray], output_dir: str, save_format: str = 'npy'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if save_format in ['npy', 'both']:
        npy_dir = output_path / 'npy'
        npy_dir.mkdir(exist_ok=True)
        for fn, enc in tqdm(encodings.items(), desc="Saving .npy files"):
            base = os.path.splitext(fn)[0]
            np.save(npy_dir / f"{base}_text_encoding.npy", enc)
    if save_format in ['json', 'both']:
        md = {
            'num_files': len(encodings),
            'dim': next(iter(encodings.values())).shape[0] if encodings else 0,
            'files': list(encodings.keys())
        }
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(md, f, indent=2)
    print(f"✓ Saved {len(encodings)} text encodings to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_format', choices=['npy','json','both'], default='npy')
    args = parser.parse_args()
    files = []
    for ext in ['.txt', '.md']:
        files += list(Path(args.input_dir).rglob(f"*{ext}"))
    files = [str(f) for f in files]
    if not files:
        print("No text files found.")
        return
    enc = extract_text_encodings_batch(TextEncoder(), files, args.batch_size)
    save_encodings(enc, args.output_dir, args.save_format)

if __name__ == '__main__':
    main()
