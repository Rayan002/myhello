"""
Logging configuration using loguru.
Logs to terminal, logger.txt (all logs), and error_log.txt (errors only).
"""
from pathlib import Path
from loguru import logger

# Remove default handler
logger.remove()

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure logger
logger.add(
    LOG_DIR / "logger.txt",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    encoding="utf-8"
)

# Configure error logger (errors and above only)
logger.add(
    LOG_DIR / "error_log.txt",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    rotation="10 MB",
    retention="30 days",
    level="ERROR",
    encoding="utf-8"
)

# Add console handler (terminal)
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    level="INFO",
    colorize=True
)

__all__ = ["logger"]

