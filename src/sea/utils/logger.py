import os
from sea.utils.config import Config


def write_message_to_log_file(message: str) -> None:
    """Logs a message to a specified file."""
    cfg = Config(load=True)

    with open(cfg.LOG_PATH, 'a') as log_file:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_file.write(f"[{timestamp}]  {message}\n")