import os
import re
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# 兼容：作为子模块导入 或 直接运行 train.py
try:
    from .logger_config import get_logger
except ImportError:
    from logger_config import get_logger

logger = get_logger(__name__)


def _sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """将特征名中的特殊字符替换为下划线，LightGBM 不支持 < > [ ] 等 JSON 特殊字符"""
    new_columns = {col: re.sub(r"[<>\[\]{}:,\"\\]", "_", col) for col in df.columns}
    return df.rename(columns=new_columns)


def train_and_save_lgb(
    df: pd.DataFrame,
    ingore_cols: list[str],
    target_col: str,
    model_save_dir: str,
    model_filename: str = "lgb_model.txt",
) -> None:
    """
    训练 LightGBM 模型并将其保存到指定目录。

    参数:
        df: 包含特征和标签的 Pandas DataFrame。
        target_col: 目标变量（标签）的列名，例如 'label'。
        model_save_dir: 模型文件保存的目录路径。
        model_filename: 保存的模型文件名。
    返回:
        无返回值 (None)
    """
    logger.info("=== 开始 LightGBM 模型训练流程 ===")

    try:
        # 1. 基础校验
        if target_col not in df.columns:
            logger.error(
                "数据集中未找到目标列: %s，可用列: %s", target_col, list(df.columns)
            )
            raise ValueError(f"数据集中未找到目标列: {target_col}")

        # 2. 准备特征 (X) 和标签 (y)
        # 注意：通常我们会去掉 userId 和 movieId 这种纯 ID 类特征，避免模型死记硬背（过拟合）。
        # 如果你想保留，请确保它们被视作类别特征 (categorical_feature)。这里演示简单起见直接去掉标签列。
        drop_cols = [target_col, *ingore_cols]
        # 如果有单纯的 ID 列不参与训练，也可以在这里一并 drop，例如：drop_cols = [target_col, 'userId', 'movieId']

        X = df.drop(columns=drop_cols)
        y = df[target_col]

        # 清洗特征名：LightGBM 不支持 < > [ ] 等特殊字符
        X = _sanitize_feature_names(X)

        # 3. 划分训练集和验证集 (80% 训练, 20% 验证)
        logger.info("正在划分训练集和验证集 (80/20)...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 4. 构建 LightGBM 的 Dataset 格式 (这种格式可以大大提高内存效率和训练速度)
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # 5. 配置模型参数 (针对二分类任务)
        params = {
            "objective": "binary",  # 任务类型：二分类
            "metric": [
                "binary_logloss",
                "auc",
            ],  # 评估指标：ROC AUC (推荐系统常用)
            "boosting_type": "gbdt",  # 提升树类型：传统梯度提升决策树
            "learning_rate": 0.05,  # 学习率
            "num_leaves": 31,  # 每棵树的最大叶子节点数 (控制复杂度)
            "max_depth": -1,  # 树的最大深度，-1 表示不限制 (由 num_leaves 控制)
            "feature_fraction": 0.8,  # 建树时随机抽样的特征比例 (防止过拟合)
            "verbose": -1,  # 关闭 LightGBM 底层的 C++ 冗余日志
            "n_jobs": -1,  # 使用所有 CPU 核心
        }

        # 6. 设置回调函数：早停机制与训练日志输出
        # 现代 LightGBM API 推荐使用 callbacks，而不是在 train 里传 early_stopping_rounds
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=50, first_metric_only=False, verbose=True
            ),
            lgb.log_evaluation(period=20),  # 每 20 轮打印一次训练日志
        ]

        # 7. 开始训练
        logger.info("模型参数配置完成，开始训练...")
        booster = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=1000,  # 最大迭代次数，因为有早停机制，可以设大一点
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )
        logger.info("验证集 AUC: %f", booster.best_score["valid"]["auc"])
        # 8. 保存模型到指定文件夹
        logger.info("训练结束，准备保存模型...")
        os.makedirs(model_save_dir, exist_ok=True)  # 如果文件夹不存在则自动创建
        model_path = os.path.join(model_save_dir, model_filename)

        # num_iteration=booster.best_iteration 确保保存的是验证集上表现最好的那一轮
        booster.save_model(model_path, num_iteration=booster.best_iteration)
        logger.info("=== 模型已成功保存至: %s ===", model_path)

        return None  # 严格遵守无返回值要求

    except Exception as e:
        # 记录完整异常堆栈到 ERROR 日志（会写入 .error.log）
        logger.exception("LightGBM 训练过程发生异常: %s", e)
        raise
