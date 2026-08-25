from __future__ import annotations

import logging
import sys


def _get_log_level(level: str) -> int:
    """将日志级别字符串转换为 logging 整数常量，无效时回退为 INFO。"""
    return getattr(logging, level.upper(), logging.INFO)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """配置全局 logger 'omni_ad' 并返回。"""
    logger = logging.getLogger("omni_ad")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(_get_log_level(level))
    logger.propagate = False
    return logger
