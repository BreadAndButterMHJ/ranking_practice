"""
LightGBM 主程序：读取原始数据，经 data_process 处理后存储到 data 文件夹。
"""

import sys
from pathlib import Path

# 将项目根目录加入 path，确保能正确导入 lightGBM.src
ROOT = Path(__file__).resolve().parent.parent  # ranking_practice
sys.path.insert(0, str(ROOT))

import pandas as pd

from src import setup_logging, get_logger
from src import RobustFrequencyLabelBinarizer
from src import train_and_save_lgb

# 1. 最先配置日志（在 import 子模块之前）
setup_logging(
    app_name="lightGBM",
    log_dir=str(ROOT / "logs"),
)

logger = get_logger(__name__)

# 路径配置
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "lightGBM" / "config"
INPUT_CSV = DATA_DIR / "final_ranking_dataset_300k.csv"
OUTPUT_CSV = DATA_DIR / "processed_ranking_dataset_300k.csv"


def main():
    """读取 CSV -> 格式处理 -> 保存到 data 文件夹"""
    logger.info("=" * 50)
    logger.info("开始数据处理流程")
    logger.info("=" * 50)

    # 1. 读取原始数据
    logger.info("读取数据文件: %s", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    logger.info("原始数据加载完成，行数: %d，列: %s", len(df), list(df.columns))

    # 2. 确保 config 目录存在
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # 3. 实例化处理器，对多标签列进行 fit_transform
    genre_processor = RobustFrequencyLabelBinarizer(
        threshold=2,
        unknown_label="UNK_GENRE",
        config_path=str(CONFIG_DIR / "genres_config.json"),
    )
    genome_processor = RobustFrequencyLabelBinarizer(
        threshold=2,
        unknown_label="UNK_GENOME",
        config_path=str(CONFIG_DIR / "genome_config.json"),
    )
    user_tags_processor = RobustFrequencyLabelBinarizer(
        threshold=2,
        unknown_label="UNK_USER_TAG",
        config_path=str(CONFIG_DIR / "user_tags_config.json"),
    )

    # 4. 对 genres_list、item_top_genome_tags、user_top_tags 进行二值化
    logger.info("开始多标签二值化处理...")
    df_genres = genre_processor.fit_transform(df["genres_list"])
    df_genome = genome_processor.fit_transform(df["item_top_genome_tags"])
    df_user_tags = user_tags_processor.fit_transform(df["user_top_tags"])

    # 5. 拼接特征，移除原始多标签列
    df_processed = pd.concat(
        [df, df_genres, df_genome, df_user_tags],
        axis=1,
    )
    df_processed = df_processed.drop(
        columns=["genres_list", "item_top_genome_tags", "user_top_tags"],
        errors="ignore",
    )

    # 6. 保存到 data 文件夹
    df_processed.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info(
        "处理完成！新数据已保存至: %s，行数: %d，列数: %d",
        OUTPUT_CSV,
        len(df_processed),
        len(df_processed.columns),
    )
    logger.info("=" * 50)
    df = pd.read_csv(OUTPUT_CSV)
    logger.info("开始训练模型...")
    train_and_save_lgb(df, "label", "./models")
    logger.info("模型训练完成！")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
