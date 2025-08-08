import os
import sys

def print_prompt_content(fpath):
    base_dir = "data/dancechat_extraction/prompts/test"
    file_path = os.path.join(base_dir, f"{fpath}.txt")

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        print(f"--- 文件内容: {file_path} ---\n")
        print(content)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python script.py <fpath>")
        print("示例: python script.py example_filename")
        sys.exit(1)

    fpath = sys.argv[1]
    print_prompt_content(fpath)
