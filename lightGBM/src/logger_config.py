"""
工业级日志配置（标准库实现，无第三方依赖）。

设计目标：
- 统一 train / inference 的日志格式与输出位置
- 控制台 + 文件同时输出
- 每天一个日志文件（按天滚动），超过 14 天的日志自动删除
- INFO/DEBUG 与 ERROR 分文件，方便排障
- 避免重复添加 handler（反复 import / 反复调用也不会重复打印）
- 通过环境变量快速切换日志级别与目录

推荐用法（train / inference 顶部加入）：

    from lightGBM.src.logger_config import get_logger
    logger = get_logger(__name__)
    logger.info("start training...")

你也可以在 main 入口显式调用一次：

    from lightGBM.src.logger_config import setup_logging
    setup_logging(app_name="lightGBM_train")
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


_INIT_LOCK = threading.RLock()
_IS_CONFIGURED = False


def _safe_mkdir(dir_path: Path) -> None:
    """创建目录（并发/重复调用安全）。"""
    dir_path.mkdir(parents=True, exist_ok=True)


def _parse_level(level: Optional[str], default: int) -> int:
    """解析日志级别：支持 'INFO' / '10' / None。"""
    if level is None:
        return default
    s = str(level).strip()
    if not s:
        return default
    if s.isdigit():
        try:
            return int(s)
        except ValueError:
            return default
    return logging._nameToLevel.get(s.upper(), default)


class JsonFormatter(logging.Formatter):
    """JSON 格式日志，适合后续接入 ELK / Loki 等系统。"""

    def format(self, record: logging.LogRecord) -> str:
        # 统一用 UTC 时间戳，避免跨机器/跨时区排查困难
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "pid": record.process,
            "thread": record.threadName,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


@dataclass(frozen=True)
class LoggingOptions:
    """日志参数（一般不需要手动构造，用 setup_logging 参数即可）。"""

    app_name: str = "ranking_practice"
    log_dir: Optional[str] = None
    level: Optional[str] = None
    console_level: Optional[str] = None
    file_level: Optional[str] = None
    when: str = "D"  # 按天滚动，每天换一个文件
    interval: int = 1  # 每 1 天滚动一次
    backup_count: int = 14  # 只保留最近 14 天，超过 14 天的日志文件自动删除
    use_json: bool = False
    utc: bool = (
        False  # TimedRotatingFileHandler 是否使用 UTC（默认 False 更贴近本地时间）
    )


def setup_logging(
    *,
    app_name: str = "ranking_practice",
    log_dir: Optional[str] = None,
    level: Optional[str] = None,
    console_level: Optional[str] = None,
    file_level: Optional[str] = None,
    when: str = "D",
    interval: int = 1,
    backup_count: int = 14,
    use_json: bool = False,
    utc: bool = False,
) -> None:
    """
    初始化全局日志配置（建议在进程启动时调用一次）。

    日志文件策略：
    - 按天滚动：每天换一个文件（当前写入主文件，午夜滚动时重命名为带日期的备份）
    - 超过 14 天的日志自动删除（仅保留最近 14 天的备份）

    环境变量（不传参数时生效）：
    - RANKING_LOG_DIR：日志目录（默认：<repo>/logs）
    - RANKING_LOG_LEVEL：全局级别（默认：INFO）
    - RANKING_CONSOLE_LEVEL：控制台级别（默认：INFO）
    - RANKING_FILE_LEVEL：文件级别（默认：DEBUG）
    - RANKING_LOG_JSON：'1'/'true' 开启 JSON 日志
    """
    global _IS_CONFIGURED

    with _INIT_LOCK:
        if _IS_CONFIGURED:
            return

        env_log_dir = os.getenv("RANKING_LOG_DIR")
        env_level = os.getenv("RANKING_LOG_LEVEL")
        env_console_level = os.getenv("RANKING_CONSOLE_LEVEL")
        env_file_level = os.getenv("RANKING_FILE_LEVEL")
        env_json = os.getenv("RANKING_LOG_JSON")

        options = LoggingOptions(
            app_name=app_name,
            log_dir=log_dir or env_log_dir,
            level=level or env_level,
            console_level=console_level or env_console_level,
            file_level=file_level or env_file_level,
            when=when,
            interval=interval,
            backup_count=backup_count,
            use_json=use_json
            or str(env_json).strip().lower() in {"1", "true", "yes", "y"},
            utc=utc,
        )

        repo_root = Path(__file__).resolve().parents[2]  # .../ranking_practice/
        final_log_dir = (
            Path(options.log_dir) if options.log_dir else (repo_root / "logs")
        )
        _safe_mkdir(final_log_dir)

        hostname = socket.gethostname()
        pid = os.getpid()

        # 根 logger：所有 logger 默认都会向上冒泡到 root
        root = logging.getLogger()
        root.setLevel(_parse_level(options.level, logging.INFO))

        # 避免重复 handler（比如 notebook 反复执行、或多次 import）
        root.handlers.clear()

        # 统一格式
        if options.use_json:
            formatter: logging.Formatter = JsonFormatter()
        else:
            # 格式：2026-03-06 03:52:45,064 [INFO] [inference.py:45] - message
            formatter = logging.Formatter(
                fmt="%(asctime)s,%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        # 控制台输出
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(_parse_level(options.console_level, logging.INFO))
        ch.setFormatter(formatter)
        root.addHandler(ch)

        # 文件输出：info/debug（每天一个文件，超过 14 天的自动删除）
        info_path = final_log_dir / f"{options.app_name}.log"
        fh = TimedRotatingFileHandler(
            filename=str(info_path),
            when=options.when,
            interval=options.interval,
            backupCount=options.backup_count,
            encoding="utf-8",
            utc=options.utc,
        )
        fh.setLevel(_parse_level(options.file_level, logging.DEBUG))
        fh.setFormatter(formatter)
        root.addHandler(fh)

        # 文件输出：error 单独一份（每天一个文件，超过 14 天的自动删除）
        err_path = final_log_dir / f"{options.app_name}.error.log"
        eh = TimedRotatingFileHandler(
            filename=str(err_path),
            when=options.when,
            interval=options.interval,
            backupCount=options.backup_count,
            encoding="utf-8",
            utc=options.utc,
        )
        eh.setLevel(logging.ERROR)
        eh.setFormatter(formatter)
        root.addHandler(eh)

        # 启动时打一个“自描述”日志，方便确认配置生效
        root.info(
            "logging configured | app=%s | host=%s | pid=%s | dir=%s | json=%s",
            options.app_name,
            hostname,
            pid,
            str(final_log_dir),
            options.use_json,
        )

        _IS_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取 logger。若未初始化则自动调用 setup_logging()。

    - name 建议传 __name__，便于定位日志来源
    - 返回的 logger 默认会向 root 冒泡，不需要每个模块单独加 handler
    """
    if not _IS_CONFIGURED:
        setup_logging()
    return logging.getLogger(name if name else "root")
