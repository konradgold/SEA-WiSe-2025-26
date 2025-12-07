from pathlib import Path
from sea.utils.config import Config
from datetime import datetime

def write_message_to_log_file(message: str) -> None:
    """Logs a message to a specified file."""
    cfg = Config(load=True)

    with open(cfg.LOG_PATH, 'a') as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_file.write(f"[{timestamp}]  {message}\n")


def dir_size(path: str | Path, follow_symlinks: bool = False) -> int:
    """Return total size in bytes of all files under path."""
    p = Path(path)
    total = 0
    for entry in p.rglob("*"):
        try:
            if entry.is_file() and (follow_symlinks or not entry.is_symlink()):
                total += entry.stat(follow_symlinks=follow_symlinks).st_size
        except (OSError, PermissionError):
            pass  # skip unreadable entries
    return total
