from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

# 数据文件
MERGED_DATA_PATH = PROCESSED_DIR / "merged_sample_300k.csv"

# 随机种子
RANDOM_SEED = 42

# 划分比例
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

# 训练超参
BATCH_SIZE = 1024
EMBEDDING_DIM = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 10
