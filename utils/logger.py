import logging
import sys

def setup_logger(name: str = "SatAgent", level: int = logging.INFO) -> logging.Logger:
    """Configures a clean, conference-grade console logger."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        
        # Clean formatting: [INFO] ModuleName: Message
        formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
