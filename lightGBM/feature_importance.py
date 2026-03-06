import lightgbm as lgb
import pandas as pd


def feature_importance(model_path: str, top_k: int = 10):
    model = lgb.Booster(model_file=model_path)
    feature_importance = model.feature_importance()
    feature_names = model.feature_name()
    feature_importance_dict = dict(zip(feature_names, feature_importance))

    feature_importance_dict = sorted(
        feature_importance_dict.items(), key=lambda x: x[1], reverse=True
    )
    feature_importance_df = pd.DataFrame(
        feature_importance_dict, columns=["feature", "importance"]
    )
    percentage = (
        feature_importance_df["importance"] / feature_importance_df["importance"].sum()
    )
    feature_importance_df["percentage"] = percentage
    feature_importance_df = feature_importance_df.sort_values(
        by="percentage", ascending=False
    )
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    feature_importance_df = feature_importance_df.head(top_k)
    return feature_importance_df


if __name__ == "__main__":
    model_path = r"D:\menghengjun.1\Desktop\ranking_practice\models\lgb_model.txt"
    feature_importance = feature_importance(model_path)
    print(feature_importance)
