"""
建议同一条数据的命名方式保持统一。。。。。。。。。。。。。。。。。。。。。。
真的会被自己气笑。
"""
# 一次性重命名
import os
import glob

def batch_rename(folder, suffix):
    files = glob.glob(os.path.join(folder, f"*{suffix}.txt"))
    for f in files:
        new_name = f.replace(suffix, "")
        os.rename(f, new_name)
        print(f"Renamed: {f} -> {new_name}")

# 示例
batch_rename("data/dancechat_extraction/responses/train", "_response")
batch_rename("data/dancechat_extraction/responses/test", "_response")
