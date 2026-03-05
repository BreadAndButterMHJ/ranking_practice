import pandas as pd
import numpy as np

ratings = pd.read_csv("rating.csv")
movies = pd.read_csv("movie.csv")
tags = pd.read_csv("tag.csv")
genome_scores = pd.read_csv("genome_scores.csv")
genome_tags = pd.read_csv("genome_tags.csv")

# ==========================================
# 1. 确定 Label (Target)
# ==========================================
# 我们以 rating.csv 为主表，打分 >= 4.0 视为喜欢 (Label=1)，否则为不喜欢 (Label=0)
dataset = ratings[["userId", "movieId", "rating", "timestamp"]].copy()
dataset["label"] = (dataset["rating"] >= 4.0).astype(int)

# ==========================================
# 2. 构建 Item Features (电影特征)
# ==========================================
# 2.1 电影流派 (Genres) 转化为列表或直接使用
movies["genres_list"] = movies["genres"].apply(lambda x: x.split("|"))

# 2.2 电影热度和平均得分
item_stats = (
    ratings.groupby("movieId")
    .agg(item_rating_count=("rating", "count"), item_rating_mean=("rating", "mean"))
    .reset_index()
)

# 2.3 电影的高频 Genome 语义标签
# 对每部电影，提取 relevance 分数最高的前 5 个 tagId
top_genomes = genome_scores.sort_values(
    ["movieId", "relevance"], ascending=[True, False]
)
top_genomes = top_genomes.groupby("movieId").head(5)
item_genome_tags = (
    top_genomes.groupby("movieId")["tagId"]
    .apply(list)
    .reset_index(name="item_top_genome_tags")
)

# 合并 Item 特征
item_features = movies[["movieId", "genres_list"]].merge(
    item_stats, on="movieId", how="left"
)
item_features = item_features.merge(item_genome_tags, on="movieId", how="left")

# ==========================================
# 3. 构建 User Features (用户特征)
# ==========================================
# 3.1 用户的打分习惯
user_stats = (
    ratings.groupby("userId")
    .agg(user_rating_count=("rating", "count"), user_rating_mean=("rating", "mean"))
    .reset_index()
)

# 3.2 用户的标签偏好 (从 tag.csv)
# 统计用户打过最多次数的前 3 个 Tag
user_tags = (
    tags.groupby("userId")["tag"]
    .apply(lambda x: list(x.value_counts().index[:3]))
    .reset_index(name="user_top_tags")
)

# 合并 User 特征
user_features = user_stats.merge(user_tags, on="userId", how="left")

# ==========================================
# 4. 组装最终的大宽表 (Join Master Table)
# ==========================================
# 将主表与特征表拼接
dataset = dataset.merge(user_features, on="userId", how="left")
dataset = dataset.merge(item_features, on="movieId", how="left")

# 提取 Context 时间特征
dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
dataset["hour"] = dataset["timestamp"].dt.hour
dataset["dayofweek"] = dataset["timestamp"].dt.dayofweek

# 剔除不参与训练的列
final_ranking_dataset = dataset.drop(columns=["rating", "timestamp"])

# 缺失值填充 (对于没有打过tag的用户，用空列表填充)
final_ranking_dataset["user_top_tags"] = final_ranking_dataset["user_top_tags"].apply(
    lambda x: x if isinstance(x, list) else []
)
final_ranking_dataset["item_top_genome_tags"] = final_ranking_dataset[
    "item_top_genome_tags"
].apply(lambda x: x if isinstance(x, list) else [])
