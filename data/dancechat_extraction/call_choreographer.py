import os
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import openai
from tqdm import tqdm
import json
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAIBatchProcessor:
    """OpenAI API批量处理器"""
    
    def __init__(
        self, 
        api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        rate_limit_delay: float = 1.0,
        base_url: str = None
    ):
        """
        初始化OpenAI批量处理器
        
        Args:
            api_key (str): OpenAI API密钥
            model (str): 使用的模型，默认gpt-4o
            max_tokens (int): 最大token数
            temperature (float): 温度参数
            rate_limit_delay (float): 请求间隔时间（秒），用于避免频率限制
            base_url (str): API基础URL，用于中转服务（如jeniya.top等）
        """
        # 根据是否提供base_url来初始化client
        try:
            if base_url:
                # 尝试禁用代理来避免socks5h问题
                import httpx
                self.client = openai.OpenAI(
                    api_key=api_key, 
                    base_url=base_url,
                    http_client=httpx.Client(proxies={})  # 明确禁用代理
                )
                logger.info(f"使用自定义API端点: {base_url}")
            else:
                self.client = openai.OpenAI(
                    api_key=api_key,
                    http_client=httpx.Client(proxies={})  # 明确禁用代理
                )
                logger.info("使用官方OpenAI API端点")
        except Exception as e:
            logger.warning(f"创建带禁用代理的客户端失败，尝试默认设置: {e}")
            # 回退到原始方式
            if base_url:
                self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rate_limit_delay = rate_limit_delay
        
        # 验证API连接
        self._test_connection()
    
    def _test_connection(self):
        """测试API连接"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            logger.info("OpenAI API连接成功")
        except Exception as e:
            logger.error(f"OpenAI API连接失败: {e}")
            raise
    
    def call_openai_api(self, prompt: str, retry_count: int = 3) -> Optional[str]:
        """
        调用OpenAI API
        
        Args:
            prompt (str): 输入的prompt
            retry_count (int): 重试次数
            
        Returns:
            Optional[str]: API响应内容，失败时返回None
        """
        for attempt in range(retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                return response.choices[0].message.content.strip()
                
            except openai.RateLimitError as e:
                logger.warning(f"遇到频率限制，等待更长时间后重试... (尝试 {attempt + 1}/{retry_count})")
                time.sleep(min(60, (attempt + 1) * 10))  # 指数退避
                
            except openai.APIError as e:
                logger.error(f"OpenAI API错误: {e} (尝试 {attempt + 1}/{retry_count})")
                if attempt == retry_count - 1:
                    return None
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"未知错误: {e} (尝试 {attempt + 1}/{retry_count})")
                if attempt == retry_count - 1:
                    return None
                time.sleep(5)
        
        return None
    
    def read_prompt_file(self, file_path: Path, encoding: str = 'utf-8') -> Optional[str]:
        """
        读取prompt文件
        
        Args:
            file_path (Path): 文件路径
            encoding (str): 文件编码
            
        Returns:
            Optional[str]: 文件内容，失败时返回None
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"读取文件 {file_path} 失败: {e}")
            return None
    
    def save_response(self, response: str, output_path: Path, encoding: str = 'utf-8') -> bool:
        """
        保存API响应到文件
        
        Args:
            response (str): API响应内容
            output_path (Path): 输出文件路径
            encoding (str): 文件编码
            
        Returns:
            bool: 保存是否成功
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding=encoding) as f:
                f.write(response)
            return True
        except Exception as e:
            logger.error(f"保存文件 {output_path} 失败: {e}")
            return False
    
    def save_metadata(self, metadata: Dict[str, Any], output_path: Path) -> bool:
        """
        保存处理元数据
        
        Args:
            metadata (Dict): 元数据
            output_path (Path): 输出路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
            return False
    
    def process_folder(
        self, 
        prompt_folder: str, 
        output_folder: str,
        file_extensions: list = None,
        save_metadata: bool = True,
        limit: int = None,
        force_overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        批量处理文件夹中的prompt文件
        
        Args:
            prompt_folder (str): prompt文件夹路径
            output_folder (str): 输出文件夹路径
            file_extensions (list): 要处理的文件扩展名
            save_metadata (bool): 是否保存元数据
            limit (int): 限制处理的文件数量，None表示处理所有文件
            force_overwrite (bool): 是否强制覆盖已存在的响应文件
            
        Returns:
            Dict: 处理结果统计
        """
        if file_extensions is None:
            file_extensions = ['.txt', '.prompt', '.md']
        
        prompt_path = Path(prompt_folder)
        output_path = Path(output_folder)
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt文件夹不存在: {prompt_folder}")
        
        # 获取所有prompt文件
        prompt_files = []
        for ext in file_extensions:
            prompt_files.extend(prompt_path.glob(f'*{ext}'))
        
        prompt_files = sorted(prompt_files)
        
        # 限制处理的文件数量
        if limit is not None and limit > 0:
            prompt_files = prompt_files[:limit]
            logger.info(f"限制处理前 {limit} 个文件")
        
        if not prompt_files:
            logger.warning(f"在 {prompt_folder} 中未找到任何prompt文件")
            return {"total": 0, "success": 0, "failed": 0, "skipped": 0}
        
        logger.info(f"找到 {len(prompt_files)} 个prompt文件")
        
        # 处理统计
        stats = {
            "total": len(prompt_files),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "failed_files": [],
            "processing_time": 0
        }
        
        start_time = time.time()
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 批量处理
        for prompt_file in tqdm(prompt_files, desc="处理prompt文件"):
            # 生成输出文件路径
            output_file = output_path / f"{prompt_file.stem}_response.txt"
            
            # 检查是否已存在（跳过已处理的文件）
            if output_file.exists() and not force_overwrite:
                logger.info(f"跳过已存在的响应文件: {output_file.name}")
                stats["skipped"] += 1
                continue
            
            # 读取prompt
            prompt_content = self.read_prompt_file(prompt_file)
            if prompt_content is None:
                stats["failed"] += 1
                stats["failed_files"].append(str(prompt_file))
                continue
            
            # 调用API
            logger.info(f"处理文件: {prompt_file.name}")
            response = self.call_openai_api(prompt_content)
            
            if response is None:
                logger.error(f"API调用失败: {prompt_file.name}")
                stats["failed"] += 1
                stats["failed_files"].append(str(prompt_file))
                continue
            
            # 保存响应
            if self.save_response(response, output_file):
                stats["success"] += 1
                logger.info(f"成功保存响应: {output_file}")
            else:
                stats["failed"] += 1
                stats["failed_files"].append(str(prompt_file))
            
            # 频率限制延迟
            time.sleep(self.rate_limit_delay)
        
        stats["processing_time"] = time.time() - start_time
        
        # 保存处理元数据
        if save_metadata:
            metadata_file = output_path / "processing_metadata.json"
            stats_copy = stats.copy()
            stats_copy["model"] = self.model
            stats_copy["temperature"] = self.temperature
            stats_copy["max_tokens"] = self.max_tokens
            self.save_metadata(stats_copy, metadata_file)
        
        # 打印统计信息
        logger.info(f"处理完成！")
        logger.info(f"总文件数: {stats['total']}")
        logger.info(f"成功: {stats['success']}")
        logger.info(f"失败: {stats['failed']}")
        logger.info(f"跳过: {stats['skipped']}")
        logger.info(f"处理时间: {stats['processing_time']:.2f}秒")
        
        if stats["failed_files"]:
            logger.warning(f"失败的文件: {stats['failed_files']}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description='批量处理prompt文件并调用OpenAI API'
    )
    parser.add_argument('--prompt_folder', required=True, help='存放prompt文件的文件夹路径')
    parser.add_argument('--output_folder', required=True, help='保存API响应的文件夹路径')
    parser.add_argument('--api_key', help='OpenAI API密钥（也可通过环境变量OPENAI_API_KEY设置）')
    parser.add_argument('--model', default='gpt-4o', help='使用的模型，默认gpt-4o')
    parser.add_argument('--max_tokens', type=int, default=1000, help='最大token数，默认1000')
    parser.add_argument('--temperature', type=float, default=0.7, help='温度参数，默认0.7')
    parser.add_argument('--rate_limit_delay', type=float, default=1.0, help='请求间隔时间（秒），默认1.0')
    parser.add_argument('--force_overwrite', action='store_true', help='强制覆盖已存在的响应文件')
    parser.add_argument('--base_url', help='API基础URL（用于中转服务，如jeniya.top等）')
    parser.add_argument('--limit', type=int, help='限制处理的文件数量（用于测试），例如 --limit 3')
    parser.add_argument('--file_extensions', nargs='+', default=['.txt'], help='要处理的文件扩展名')
    
    args = parser.parse_args()
    
    # 获取API密钥
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("请提供OpenAI API密钥，通过--api_key参数或OPENAI_API_KEY环境变量")
    
    # 创建处理器
    processor = OpenAIBatchProcessor(
        api_key=api_key,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        rate_limit_delay=args.rate_limit_delay,
        base_url=args.base_url
    )
    
    # 处理文件夹
    try:
        stats = processor.process_folder(
            prompt_folder=args.prompt_folder,
            output_folder=args.output_folder,
            file_extensions=args.file_extensions,
            limit=args.limit,
            force_overwrite=args.force_overwrite
        )
        
        print(f"\n处理完成！成功: {stats['success']}, 失败: {stats['failed']}, 跳过: {stats['skipped']}")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    main()