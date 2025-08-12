import numpy as np
import pickle
import os

# 设置路径
npy_path = "data/edge_aistpp/motions"

def print_file_shapes(path):
    if os.path.isfile(path) and path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            print(f"{os.path.basename(path)}:")
            if isinstance(data, dict):
                for k, v in data.items():
                    if hasattr(v, 'shape'):
                        print(f"  {k}: shape = {v.shape}")
                    else:
                        print(f"  {k}: type = {type(v)}")
            elif hasattr(data, 'shape'):
                print(f"  shape = {data.shape}")
            else:
                print(f"  type = {type(data)}")
    
    elif os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith((".pkl", ".npy"))]
        for f in sorted(files):
            file_path = os.path.join(path, f)
            if f.endswith(".pkl"):
                with open(file_path, "rb") as f_in:
                    data = pickle.load(f_in)
                    print(f"{f}:")
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if hasattr(v, 'shape'):
                                print(f"  {k}: shape = {v.shape}")
                            else:
                                print(f"  {k}: type = {type(v)}")
                    elif hasattr(data, 'shape'):
                        print(f"  shape = {data.shape}")
                    else:
                        print(f"  type = {type(data)}")
            elif f.endswith(".npy"):
                data = np.load(file_path)
                print(f"{f}: shape = {data.shape}")
    else:
        print("路径无效，既不是文件也不是文件夹。")

# 调用函数
print_file_shapes(npy_path)
