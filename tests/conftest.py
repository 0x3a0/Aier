"""pytest 全局配置。

在 `src` 布局下，无需安装即可导入 `aier` 包。
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
