import pandas as pd

"""
"movieId","title","genres"
1,"Toy Story (1995)","Adventure|Animation|Children|Comedy|Fantasy"
"movieId","tagId","relevance"
1,1,0.025
"userId","movieId","rating","timestamp"
1,2,3.5,2005-04-02 23:53:47
"""

MOVIE_CSV = r"D:\menghengjun.1\Desktop\ranking_practice\data\movie.csv"
GENOME_SCORES_CSV = r"D:\menghengjun.1\Desktop\ranking_practice\data\genome_scores.csv"
RATING_CSV = r"D:\menghengjun.1\Desktop\ranking_practice\data\rating.csv"
ratings = pd.read_csv(RATING_CSV)
movies = pd.read_csv(MOVIE_CSV)
genome_scores = pd.read_csv(GENOME_SCORES_CSV)
movies["genres_list"] = movies["genres"].apply(lambda x: x.split("|"))
print("=======打印movies表头=======")
print(movies.head())
# 修正：只保留每部电影 relevance 最高的5个tag，并且把tagId组成list，列名为item_top_genome_tags
genome_scores_sorted = genome_scores.sort_values(
    ["movieId", "relevance"], ascending=[True, False]
)
# 正确写法
genome_scores_top_5 = (
    genome_scores_sorted.groupby("movieId")
    .head(5)
    .groupby("movieId")["tagId"]
    .apply(list)
    .reset_index(name="item_top_genome_tags")
)
print("=======打印genome_scores_top_5表头=======")
print(genome_scores_top_5.head())
item_rating_count = ratings.groupby("movieId").agg(
    item_rating_count=("rating", "count")
)
print("=======打印item_rating_count表头=======")
print(item_rating_count.head())
item_rating_mean = ratings.groupby("movieId").agg(item_rating_mean=("rating", "mean"))
print("=======打印item_rating_mean表头=======")
print(item_rating_mean.head())
output_df = (
    movies[["movieId", "genres_list"]]
    .merge(item_rating_count, on="movieId", how="left")
    .merge(item_rating_mean, on="movieId", how="left")
    .merge(genome_scores_top_5, on="movieId", how="left")
    .reset_index(drop=True)
)
print("=======打印output_df表头=======")
print(output_df.head())
output_df.to_csv(
    r"D:\menghengjun.1\Desktop\ranking_practice\data\movie_features.csv", index=False
)
