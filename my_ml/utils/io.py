from pathlib import Path

import yaml
import datetime


def create_dir(save_dir: Path) -> Path:
    """Create and get a unique dir path to save to using a timestamp."""
    time = str(datetime.datetime.now())
    for char in ":- .":
        time = time.replace(char, "_")
    path: path = Path(save_dir / f"results_{time}")
    path.mkdir(parents=True, exist_ok=True)
    return path
