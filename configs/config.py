from box import Box
import yaml
from pathlib import Path

cfg = Box(yaml.safe_load(Path("configs/default.yaml").read_text()))