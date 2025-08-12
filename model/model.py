from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F

from model.rotary_embedding_torch import RotaryEmbedding
from model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like

from typing import Dict
import os

class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    # 似乎与tgt_mask / memory_mask无关？并未看到调用时传入
    def forward(
        self,
        tgt,
        memory,
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            # self-attention -> film -> residual
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            # cross-attention -> film -> residual
            x_2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            # feedforward -> film -> residual
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn( # nn.MultiheadAttention层
            qk,
            qk,
            x,
            attn_mask=attn_mask, # 全注意力 or 因果注意力
            key_padding_mask=key_padding_mask, # 是否要attend到padding
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # cross attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t):
        for layer in self.stack:
            x = layer(x, cond, t)
        return x

class MultiModalProjector(nn.Module):
    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 512, output_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.text_dim = 256

        # Jukebox: 2-layer Transformer
        encoder_layer = lambda: nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.jukebox_proj = nn.Sequential(
            nn.Linear(input_dims['jukebox'], hidden_dim),
            nn.ReLU(),
            nn.TransformerEncoder(encoder_layer(), num_layers=2),
        )

        # Beat: Linear projection
        self.beat_proj = nn.Linear(input_dims['beat'], hidden_dim)

        # Text: 4-layer Transformer to get global pooled (512,)
        encoder_layer_text = lambda: nn.TransformerEncoderLayer(d_model=self.text_dim, nhead=8, batch_first=True)
        self.text_proj = nn.Sequential(
            nn.Linear(input_dims['text'], self.text_dim),
            nn.ReLU(),
            nn.TransformerEncoder(encoder_layer_text(), num_layers=4),
        )

        self.text_align_proj = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )

        # Fusion MLP for feature1
        # 对应文中的cross-modal encoder
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),       # 1024 → 标准化
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)            # 再次归一化输出
        )

        self.cross_modal_fusion_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),       # 1024 → 标准化
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)            # 再次归一化输出
        )

    def forward(self, jukebox_feat, beat_feat, text_feat):
        # shapes: (B, 150, ?), (B, 150, ?), (B, ?, ?)

        if text_feat.dim() == 1:
            # 如果是 (256,)，扩展为 (1, 1, 256)
            text_feat = text_feat.unsqueeze(0).unsqueeze(0)
        elif text_feat.dim() == 2:
            # 如果是 (B, 256)，扩展为 (B, 1, 256)
            text_feat = text_feat.unsqueeze(1)
        # Step 1: projection
        jukebox_out = self.jukebox_proj(jukebox_feat)  # (B, 150, 512)
        beat_out = self.beat_proj(beat_feat)           # (B, 150, 512)

        # Text projection: assume (B, T_text, D_text) → pool to (B, 512)
        text_encoded = self.text_proj(text_feat.float())       # (B, T, 256)
        text_encoded = self.text_align_proj(text_encoded) # (B,T,256)->(B,T,512)
        text_pooled = text_encoded.mean(dim=1)         # (B, 512)

        # Step 2: feature1 for FiLM
        fused_music = torch.cat([jukebox_out, beat_out], dim=-1)  # (B, 150, 1024)
        fused_music_mlp = self.fusion_mlp(fused_music) # (B, 150, 512)
        text_added = torch.cat([fused_music_mlp, text_pooled.unsqueeze(1).expand(-1, 150, -1)], dim=-1)  # (B, 150, 512)
        feature1 = self.cross_modal_fusion_mlp(text_added)  # (B, 150, 512)

        # Step 3: feature2 for CA
        feature2 = torch.cat([fused_music_mlp, text_pooled.unsqueeze(1)], dim=1)  # (B, 151, 512)

        return feature1, feature2, fused_music_mlp.mean(dim=1), text_pooled

def load_text_encoder(multi_modal_projector, motiondiffuse_state_dict: Dict[str, torch.Tensor]):
    new_state_dict = {}
    # 映射text_pre_proj权重
    new_state_dict['text_proj.0.weight'] = motiondiffuse_state_dict['text_pre_proj.weight']
    new_state_dict['text_proj.0.bias'] = motiondiffuse_state_dict['text_pre_proj.bias']
    
    # 映射transformer layers权重
    for i in range(4):
        source_prefix = f'textTransEncoder.layers.{i}'
        target_prefix = f'text_proj.2.layers.{i}'
        
        key_map = {
            'self_attn.in_proj_weight': 'self_attn.in_proj_weight',
            'self_attn.in_proj_bias': 'self_attn.in_proj_bias',
            'self_attn.out_proj.weight': 'self_attn.out_proj.weight',
            'self_attn.out_proj.bias': 'self_attn.out_proj.bias',
            'linear1.weight': 'linear1.weight',
            'linear1.bias': 'linear1.bias',
            'linear2.weight': 'linear2.weight',
            'linear2.bias': 'linear2.bias',
            'norm1.weight': 'norm1.weight',
            'norm1.bias': 'norm1.bias',
            'norm2.weight': 'norm2.weight',
            'norm2.bias': 'norm2.bias',
        }
        
        for source_suffix, target_suffix in key_map.items():
            source_key = f'{source_prefix}.{source_suffix}'
            target_key = f'{target_prefix}.{target_suffix}'
            
            if source_key in motiondiffuse_state_dict:
                new_state_dict[target_key] = motiondiffuse_state_dict[source_key]
    
    # 加载权重到multi_modal_projector
    multi_modal_projector.load_state_dict(new_state_dict, strict=False)
    print("✅ Successfully loaded MotionDiffuse weights into text encoder")



class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return self.out_net(outputs)

class MotionProjectionEncoder(nn.Module):
    """
    完整的Motion处理模型：151维 -> 259维投影 -> 预训练Motion Encoder -> 512维输出
    """
    def __init__(self, projection_hidden_dim=256):
        super(MotionProjectionEncoder, self).__init__()
        
        # 投影层：151维 -> 259维 (不frozen，可训练)
        self.projection = nn.Sequential(
            nn.Linear(151, projection_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_hidden_dim, 259)
        )
        
        # 预训练的Motion Encoder (frozen)
        # 从checkpoint信息可知：input_size=259, hidden_size=512, output_size=512
        self.motion_encoder = MovementConvEncoder(
            input_size=259,
            hidden_size=512, 
            output_size=512
        )
        
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 形状为 (batch_size, seq_len, 151) 的输入张量
        Returns:
            output: 形状为 (batch_size, seq_len//4, 512) 的输出张量
                   (注意：由于Conv1d中stride=2被用了两次，序列长度会减少到原来的1/4)
        """
        # Step 1: 投影 151维 -> 259维
        # x shape: (batch_size, seq_len, 151) -> (batch_size, seq_len, 259)
        projected = self.projection(x)
        
        # Step 2: 通过预训练的Motion Encoder
        # projected shape: (batch_size, seq_len, 259) -> (batch_size, seq_len//4, 512)
        encoded = self.motion_encoder(projected)
        
        return encoded

def load_motion_encoder_weights(motion_encoder: MotionProjectionEncoder, 
                              checkpoint_path: str = "../HumanML3D/t2m/Decomp_SP001_SM001_H512/model/latest.tar"):
    """
    加载预训练的 Motion Encoder 权重到完整模型中
    
    Args:
        model (MotionProjectionEncoder): 已经实例化的完整模型
        checkpoint_path (str): 保存权重的 .tar 或 .pth 文件路径
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"[*] Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 检查 'movement_enc' 键是否存在
    if 'movement_enc' not in checkpoint:
        raise KeyError("Checkpoint does not contain the key 'movement_enc'. "
                      f"Available keys are: {list(checkpoint.keys())}")
    
    # 提取 movement_enc 的 state_dict
    movement_enc_state_dict = checkpoint['movement_enc']
    
    # 只加载motion_encoder部分的权重
    motion_encoder.load_state_dict(movement_enc_state_dict)
    
     
    for param in motion_encoder.parameters():
        param.requires_grad = False  # 冻结所有参数


"""
主Decoder类
"""
class DanceDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        seq_len: int = 150,  # 5 seconds, 30 fps
        latent_dim: int = 512, # 注：EDGE.py中初始化的时候设成了512
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 512,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        **kwargs
    ) -> None:

        super().__init__()

        output_feats = nfeats

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),  # learned?
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)

        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len + 1, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        
        self.multi_modal_projector = MultiModalProjector(
            input_dims={'jukebox': 4800, 'beat': 2, 'text': 512},  # 假设 text 编码前是 512
            hidden_dim=512
        )
        
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        
        self.final_layer = nn.Linear(latent_dim, output_feats)


        self.motion_projection_encoder = MotionProjectionEncoder()

    """
    classifier-free guidance
    """
    def guided_forward(self, x, cond1, cond2, cond3, times, guidance_weight):
        unc = self.forward(x, cond1, cond2, cond3, times, cond_drop_prob=1)
        conditioned = self.forward(x, cond1, cond2, cond3, times, cond_drop_prob=0)

        return unc + (conditioned - unc) * guidance_weight

    def forward(
        self, x: Tensor, cond1: Tensor, cond2: Tensor, cond3: Tensor, times: Tensor, cond_drop_prob: float = 0.0
    ):
        """
        feature1 for FiLM, feature2 for cross-attention
        """
        feature1, feature2, fuse_music_mlp_pooled, text_pooled = self.multi_modal_projector(cond1, cond2, cond3)
        batch_size, device = x.shape[0], x.device

        """
        Motion input projection
        """
        # project to latent space
        x = self.input_projection(x) # (B, 150, 512)
        # add the positional embeddings of the input sequence to provide temporal information
        x = self.abs_pos_encoding(x)

        """
        Conditional Dropout Mask
        """
        # create music conditional embedding with conditional dropout
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        # 为一整个 batch 随机决定哪些样本“保留”条件信息，哪些样本“丢弃”条件（即做条件性 dropout）。
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")
        # 拓展其维度，便于后面操作

        """
        cross-attention input
        """
        cond_tokens = feature2
        # encode tokens
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        
        # 可学习的空条件嵌入，希望mask掉的时候可以用它代替真实值
        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)
        
        """
        FiLM input
        """
        mean_pooled_feature1 = feature1.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_feature1) # (B, 512) -> (B, 512)

        # create the diffusion timestep embedding, add the extra music projection
        t_hidden = self.time_mlp(times) # (B, 512*4)

        # project to attention and FiLM conditioning
        t = self.to_time_cond(t_hidden) # (B, 512)
        t_tokens = self.to_time_tokens(t_hidden) # (B, 2, 512)

        # FiLM conditioning
        null_cond_hidden = self.null_cond_hidden.to(t.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        t += cond_hidden # (B, 512)

        # cross-attention conditioning
        c = torch.cat((cond_tokens, t_tokens), dim=-2) # (B, 153, 512)
        cond_tokens = self.norm_cond(c)

        # Pass through the transformer decoder
        # attending to the conditional embedding
        # 相当于diffusion model里面那个xN
        output = self.seqTransDecoder(x, cond_tokens, t)

        output = self.final_layer(output)
        return output

    def get_embeddings(self, x: Tensor, cond1: Tensor, cond2: Tensor, cond3: Tensor):
        feature1, feature2, fuse_music_mlp_pooled, text_pooled = self.multi_modal_projector(cond1, cond2, cond3)
        encoded_x = self.motion_projection_encoder(x)
        encoded_x = encoded_x.mean(dim=1)
        # print('shapes: ',fuse_music_mlp_pooled.shape, text_pooled.shape, encoded_x.shape)
        return fuse_music_mlp_pooled, text_pooled, encoded_x
