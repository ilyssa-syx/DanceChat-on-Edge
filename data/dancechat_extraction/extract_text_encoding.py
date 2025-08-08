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
    def __init__(self, clip_model_name: str = "ViT-B/32"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 加载CLIP文本编码器，并冻结参数
        self.motiondiffuse_checkpoint_path = "../text2motion/checkpoints/t2m/t2m_motiondiffuse/model/latest.tar"
        self.clip_model, _ = clip.load(clip_model_name, device=self.device)
        
        with torch.no_grad():
            dummy_text = clip.tokenize(["test"]).to(self.device)
            dummy_features = self.clip_model.encode_text(dummy_text)
            self.clip_dim = dummy_features.shape[1]
    
        if self.motiondiffuse_checkpoint_path:
            self._load_motiondiffuse_clip_weights(self.motiondiffuse_checkpoint_path)
            print(f"✓ Text encoder initialized with MotionDiffuse CLIP weights (dim={self.clip_dim})")
        else:
            print(f"✓ Text encoder initialized with standard CLIP weights (dim={self.clip_dim})")
        
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def _load_motiondiffuse_clip_weights(self, checkpoint_path: str):
        """从 MotionDiffuse 的 checkpoint 中加载 CLIP 文本编码器权重"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['encoder']
        
        clip_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('clip.'):
                clip_key = key[5:]  # 去掉 'clip.' 前缀)
                clip_state_dict[clip_key] = value
        
        missing_keys, unexpected_keys = self.clip_model.load_state_dict(clip_state_dict, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys in CLIP model: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            
        print("✓ Successfully loaded MotionDiffuse CLIP weights")
        self.clip_model.to(self.device)

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


def save_encodings(encodings: Dict[str, np.ndarray], output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    npy_dir = output_path / 'npy'
    npy_dir.mkdir(exist_ok=True)
    for fn, enc in tqdm(encodings.items(), desc="Saving .npy files"):
        base = os.path.splitext(fn)[0]
        np.save(npy_dir / f"{base}.npy", enc)
    
    print(f"✓ Saved {len(encodings)} text encodings to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    files = []
    for ext in ['.txt', '.md']:
        files += list(Path(args.input_dir).rglob(f"*{ext}"))
    files = [str(f) for f in files]
    if not files:
        print("No text files found.")
        return
    enc = extract_text_encodings_batch(TextEncoder(), files, args.batch_size)
    save_encodings(enc, args.output_dir)

if __name__ == '__main__':
    main()
