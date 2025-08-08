import os
import pickle
from dataset.dance_dataset import AISTPPDataset  # ⚠️ 路径按你项目结构修改

def test_aistpp_name_matching():
    # 设置路径（根据你的项目修改）
    data_path = "data"
    backup_path = "data"

    # 实例化 dataset（不需要训练，只测试加载）
    dataset = AISTPPDataset(
        data_path="data",
        backup_path=backup_path,
        train=True,
        force_reload=True  # 强制重新 load_aistpp()
    )

    print("\n🔍 Testing name matching for loaded samples...")

    mismatches = []
    for i in range(20):
        print()

if __name__ == "__main__":
    test_aistpp_name_matching()
