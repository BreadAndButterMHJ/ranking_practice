import pandas as pd
import lightgbm as lgb
import os
import sys
from pathlib import Path

# 兼容两种运行方式：
# 1) 作为包导入：from lightGBM.src.inference_production import inference_production
# 2) 直接运行脚本：python lightGBM/src/inference_production.py（此时相对导入会失败）
try:
    from .data_process import RobustFrequencyLabelBinarizer
except ImportError:
    # 直接运行脚本时，补齐项目根目录到 sys.path，保证可用绝对导入
    repo_root = Path(__file__).resolve().parents[2]  # .../ranking_practice/
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from lightGBM.src.data_process import RobustFrequencyLabelBinarizer


def inference_production(
    model_path: str,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    top_n: int = 20,
):
    CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
    input_data = pd.concat([user_features, item_features], axis=1)
    input_data = input_data[
        [
            "userId",
            "movieId",
            "user_rating_count",
            "user_rating_mean",
            "user_top_tags",
            "genres_list",
            "item_rating_count",
            "item_rating_mean",
            "item_top_genome_tags",
            "hour",
            "dayofweek",
        ]
    ]
    genre_processor = RobustFrequencyLabelBinarizer(
        config_path=os.path.join(CONFIG_DIR, "genres_config.json")
    )
    genome_processor = RobustFrequencyLabelBinarizer(
        config_path=os.path.join(CONFIG_DIR, "genome_config.json")
    )
    user_tags_processor = RobustFrequencyLabelBinarizer(
        config_path=os.path.join(CONFIG_DIR, "user_tags_config.json")
    )
    df_genres = genre_processor.transform(input_data["genres_list"])
    df_genome = genome_processor.transform(input_data["item_top_genome_tags"])
    df_user_tags = user_tags_processor.transform(input_data["user_top_tags"])
    input_data = pd.concat([input_data, df_genres, df_genome, df_user_tags], axis=1)
    movie_id = input_data["movieId"]
    input_data = input_data.drop(
        columns=[
            "userId",
            "movieId",
            "genres_list",
            "item_top_genome_tags",
            "user_top_tags",
            "user_rating_mean",
        ],
        errors="ignore",
    )
    model = lgb.Booster(model_file=model_path)
    predictions = model.predict(input_data)
    movies_info = pd.read_csv(
        r"D:\menghengjun.1\Desktop\ranking_practice\data\movie.csv"
    )
    movies_info = movies_info[["movieId", "title"]]
    predictions = pd.DataFrame(predictions, columns=["prediction"])
    predictions["movieId"] = movie_id
    predictions = predictions.merge(movies_info, on="movieId", how="left")
    predictions = predictions.sort_values(by="prediction", ascending=False)
    output = predictions[["movieId", "title", "prediction"]].head(top_n)
    return output
