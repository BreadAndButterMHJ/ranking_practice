from src import inference_production
import pandas as pd

user_item_features = pd.DataFrame(
    {
        "userId": 1,
        "user_rating_count": 27,
        "user_rating_mean": 3.2,
        "user_top_tags": ["Action"],
        "hour": 22,
        "dayofweek": 4,
    }
)
item_features = pd.read_csv(
    r"D:\menghengjun.1\Desktop\ranking_practice\data\movie_features.csv"
)
predictions = inference_production(
    model_path=r"D:\menghengjun.1\Desktop\ranking_practice\models\lgb_model.txt",
    user_features=user_item_features,
    item_features=item_features,
    top_n=20,
)
print(predictions)
