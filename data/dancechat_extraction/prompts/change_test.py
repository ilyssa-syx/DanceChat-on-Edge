import os
import re
from pathlib import Path
from tqdm import tqdm
import argparse

def replace_genre_in_file(file_path: Path, backup: bool = True) -> bool:
    """
    替换单个文件中的genre单词
    
    Args:
        file_path (Path): 文件路径
        backup (bool): 是否创建备份文件
        
    Returns:
        bool: 是否成功替换
    """
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 创建备份（如果需要）
        if backup:
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # 使用正则表达式匹配并替换
        # 匹配模式: "Can you provide a [任何单词或多个单词] dance instruction"
        # 修改正则表达式以支持多个单词，并修复冠词问题
        pattern = r'(Can you provide a )([\w\s]+?)( dance instruction)'
        
        # 替换函数
        def replace_match(match):
            return match.group(1).replace('a ', 'an ') + 'any genre' + match.group(3)
        
        # 执行替换
        new_content = re.sub(pattern, replace_match, content)
        
        # 检查是否有变化
        if content != new_content:
            # 写入新内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        else:
            print(f"文件 {file_path.name} 中未找到匹配的模式")
            return False
            
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False

def replace_genre_in_folder(folder_path: str, backup: bool = True, preview: bool = False):
    """
    批量替换文件夹中所有txt文件的genre单词
    
    Args:
        folder_path (str): 文件夹路径
        backup (bool): 是否创建备份文件
        preview (bool): 是否只预览不实际修改
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"错误: 文件夹 {folder_path} 不存在")
        return
    
    # 获取所有txt文件
    txt_files = list(folder.glob('*.txt'))
    
    if not txt_files:
        print(f"在 {folder_path} 中未找到任何.txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个.txt文件")
    
    if preview:
        print("\n=== 预览模式 ===")
        for file_path in txt_files[:5]:  # 只显示前5个文件的预览
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 查找匹配的模式
                pattern = r'(Can you provide a )([^?]+?)( dance instruction)'
                matches = re.findall(pattern, content)
                
                if matches:
                    for match in matches:
                        print(f"文件: {file_path.name}")
                        print(f"  原文: Can you provide a {match[1]} dance instruction")
                        print(f"  替换: Can you provide an any genre dance instruction")
                        print()
                        
            except Exception as e:
                print(f"预览文件 {file_path} 时出错: {e}")
        
        if len(txt_files) > 5:
            print(f"... 还有 {len(txt_files) - 5} 个文件")
        
        return
    
    # 实际处理文件
    success_count = 0
    error_count = 0
    
    print(f"\n开始处理文件{'（创建备份）' if backup else '（不创建备份）'}...")
    
    for file_path in tqdm(txt_files, desc="处理文件"):
        if replace_genre_in_file(file_path, backup=backup):
            success_count += 1
        else:
            error_count += 1
    
    print(f"\n=== 处理完成 ===")
    print(f"成功处理: {success_count} 个文件")
    print(f"处理失败: {error_count} 个文件")
    
    if backup and success_count > 0:
        print(f"备份文件已保存（.backup后缀）")

def restore_from_backup(folder_path: str):
    """
    从备份文件恢复原始内容
    
    Args:
        folder_path (str): 文件夹路径
    """
    folder = Path(folder_path)
    backup_files = list(folder.glob('*.backup'))
    
    if not backup_files:
        print("未找到任何备份文件")
        return
    
    print(f"找到 {len(backup_files)} 个备份文件")
    
    for backup_file in tqdm(backup_files, desc="恢复文件"):
        original_file = backup_file.with_suffix('')
        
        try:
            # 读取备份内容
            with open(backup_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 写入原文件
            with open(original_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"已恢复: {original_file.name}")
            
        except Exception as e:
            print(f"恢复文件 {backup_file} 时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='批量替换txt文件中的dance genre单词')
    parser.add_argument('--folder', required=True, help='包含txt文件的文件夹路径')
    parser.add_argument('--no-backup', action='store_true', help='不创建备份文件')
    parser.add_argument('--preview', action='store_true', help='预览模式，只显示将要进行的替换')
    parser.add_argument('--restore', action='store_true', help='从备份文件恢复原始内容')
    
    args = parser.parse_args()
    
    if args.restore:
        restore_from_backup(args.folder)
    elif args.preview:
        replace_genre_in_folder(args.folder, backup=True, preview=True)
    else:
        backup = not args.no_backup
        replace_genre_in_folder(args.folder, backup=backup, preview=False)

if __name__ == '__main__':
    main()