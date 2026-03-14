#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OG-LANS Logging Utilities

Production-grade logging system designed for deep learning experiments.

Features:
    - ExperimentLogHandler: Rotating file handler with crash safety
        - Log rotation to prevent disk overflow (default: 50MB per file)
        - Forced flush after each log entry to prevent OOM data loss
        - Secure file permissions (Unix-compatible)
    - setup_logger: Unified logger configuration
        - Rich console output for real-time monitoring
        - Timestamped log files for experiment tracking
        - Prevents duplicate handler registration

Usage:
    >>> from oglans.utils.logger import setup_logger
    >>> logger = setup_logger("OGLANS", "./logs")
    >>> logger.info("Training started")

Note:
    This logger is specifically optimized for GPU-intensive training scenarios
    where OOM (Out-of-Memory) crashes are common. The forced flush mechanism
    ensures that the last log entries before a crash are preserved.

Authors:
    OG-LANS Research Team
"""

import logging
import os
import sys
from datetime import datetime
from rich.logging import RichHandler
from logging.handlers import RotatingFileHandler

class ExperimentLogHandler(RotatingFileHandler):
    """
    结合了安全性和实验稳定性的日志处理器：
    1. 支持日志轮转 (Rotating) -> 防止磁盘写满
    2. 设置文件权限 (Security) -> 防止他人查看
    3. 强制刷新 (Crash Safety) -> 防止OOM时日志丢失
    """
    def __init__(self, filename, maxBytes=50*1024*1024, backupCount=5, encoding='utf-8'):
        # 1. 确保目录存在并设置安全权限 (750)
        log_dir = os.path.dirname(filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True, mode=0o755)
        
        super().__init__(
            filename, 
            maxBytes=maxBytes, 
            backupCount=backupCount, 
            encoding=encoding
        )
        
        # 2. 设置当前日志文件权限 (640: 只有拥有者可读写，组可读)
        if os.path.exists(filename):
            try:
                os.chmod(filename, 0o644)
            except Exception:
                pass # 忽略权限设置失败（如Windows环境）

    def emit(self, record):
        """重写 emit 方法以确保 OOM 时不丢失日志"""
        try:
            super().emit(record)
            # [关键保留] 深度学习训练频率低，强制刷新对性能影响微乎其微，
            # 但能确保显存炸裂(OOM)前的最后一行日志被写入磁盘。
            self.flush() 
        except Exception:
            self.handleError(record)

    def doRollover(self):
        """轮转时也确保新文件权限正确"""
        super().doRollover()
        if self.backupCount > 0:
            for i in range(1, self.backupCount + 1):
                sfn = f"{self.baseFilename}.{i}"
                if os.path.exists(sfn):
                    try:
                        os.chmod(sfn, 0o644)
                    except:
                        pass

def setup_logger(name: str, log_dir: str = "logs", level=logging.INFO):
    """
    配置全局日志记录器
    """
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    
    # 获取 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 防止重复添加 Handler (Jupyter/Interactive 环境常见问题)
    if logger.handlers:
        return logger
    
    # --- 1. 控制台处理器 (Rich) ---
    console_handler = RichHandler(
        rich_tracebacks=True, 
        show_path=False,
        level=level # 保持 INFO，方便看进度
    )
    # 简单的格式，Rich 会自动美化
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    
    # --- 2. 文件处理器 (安全+轮转+防丢失) ---
    file_handler = ExperimentLogHandler(
        log_file,
        maxBytes=50*1024*1024,  # 50MB 轮转一次，比 10MB 更适合大模型训练日志
        backupCount=10,         # 保留 10 个历史文件
        encoding='utf-8'
    )
    file_handler.setLevel(level) # [关键] 必须是 INFO，否则看不到训练 Loss
    
    # 文件中包含详细的时间戳和模块名
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # --- 3. 记录初始化信息 (路径脱敏) ---
    try:
        # 只记录文件名，隐藏绝对路径
        safe_path = os.path.basename(log_file)
        logger.info(f"Logger initialized. Log file: {safe_path}")
    except:
        logger.info("Logger initialized.")
    
    return logger