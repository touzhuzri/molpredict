"""项目路径配置。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # 原始输入（SMILES、名称等）
PROCESSED_DIR = DATA_DIR / "processed"  # 处理后的分子记录（2D/3D/特征）
MODELS_DIR = PROJECT_ROOT / "models"    # 训练好的模型文件


def ensure_dirs() -> None:
    """创建项目运行所需的目录（幂等）。"""
    for path in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
