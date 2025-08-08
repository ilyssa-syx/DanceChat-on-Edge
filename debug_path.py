import torch

motiondiffuse_weights = torch.load("../text2motion/checkpoints/t2m/t2m_motiondiffuse/model/latest.tar")
motiondiffuse = motiondiffuse_weights['encoder']

for key in motiondiffuse.keys():
    print(key)  # 查看 MotionDiffuse 模型中的键