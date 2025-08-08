import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from smplx import SMPL
import pyrender
import trimesh

with open("motions/test_gBR_sBM_cAll_d04_mBR0_ch02.pkl", "rb") as f:
    data = pickle.load(f)

# 先看看里面是什么
print(type(data))
for key in data:
    print(f"{key}: {type(data[key])}, shape: {getattr(data[key], 'shape', None)}")
