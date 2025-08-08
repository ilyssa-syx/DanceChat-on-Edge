import numpy as np
import os

# 设置你的 .npy 文件路径（可以是单个文件，也可以是一个文件夹）
npy_path = "data/test/text_encodings/npy"

def print_npy_shapes(path):
    if os.path.isfile(path) and path.endswith(".npy"):
        data = np.load(path)
        print(f"{os.path.basename(path)}: shape = {data.shape}")
    elif os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith(".npy")]
        for f in sorted(files):
            file_path = os.path.join(path, f)
            data = np.load(file_path)
            print(f"{f}: shape = {data.shape}")
    else:
        print("路径无效，既不是.npy文件，也不是包含.npy文件的文件夹。")

# 调用函数
print_npy_shapes(npy_path)
