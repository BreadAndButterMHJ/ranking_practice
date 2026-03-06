"""
数据处理模块：多标签二值化、长尾截断、状态持久化。

使用统一的 logger_config 记录日志，日志会输出到 main 中配置的目录。
"""

import os
import json
import ast
from collections import Counter
from typing import List, Set
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# 兼容：作为子模块导入 或 直接运行 data_process.py
try:
    from .logger_config import get_logger
except ImportError:
    from logger_config import get_logger

# 使用统一日志配置，日志来源会显示为 data_process
logger = get_logger(__name__)


class RobustFrequencyLabelBinarizer:
    """
    健壮的多标签二值化处理器。
    支持长尾标签截断（转换为未知标签）、脏数据清洗以及状态持久化（JSON）。
    """

    def __init__(
        self,
        threshold: int = 2,
        unknown_label: str = "<UNK>",
        config_path: str = "label_config.json",
    ):
        self.threshold = threshold
        self.unknown_label = unknown_label
        self.config_path = config_path
        self.frequent_labels_: Set[str] = set()
        self.is_fitted = False
        self.mlb = MultiLabelBinarizer()

    @staticmethod
    def safe_parse(val) -> List[str]:
        """健壮的解析器：处理缺失值、字符串形式的列表、数字等"""
        if pd.isna(val):
            return []

        # 如果已经是列表，直接转字符串
        if isinstance(val, list):
            return [str(item) for item in val]

        # 如果是字符串，尝试解析为列表
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return []
            try:
                # 安全评估类似 "['Action', 'Comedy']" 的字符串
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
                return [str(parsed)]
            except (ValueError, SyntaxError):
                # 解析失败，把整个字符串当做一个标签
                return [val]

        # 其他类型（如 int/float）直接转字符串包裹在列表中
        return [str(val)]

    def _preprocess(self, data: List[List[str]]) -> List[List[str]]:
        """将非高频词替换为 unknown_label，并去重"""
        processed = []
        for row in data:
            new_row = [
                tag if tag in self.frequent_labels_ else self.unknown_label
                for tag in row
            ]
            processed.append(list(set(new_row)))
        return processed

    def fit(self, series: pd.Series):
        """训练转换器并保存状态"""
        logger.info("开始拟合数据，处理列: %s，样本数: %d", series.name, len(series))

        # 1. 解析数据
        parsed_data = series.apply(self.safe_parse).tolist()
        total_tags = sum(len(row) for row in parsed_data)
        logger.debug("解析完成，原始标签总数: %d", total_tags)

        # 2. 统计词频并筛选高频词
        all_tags = [tag for row in parsed_data for tag in row]
        counts = Counter(all_tags)
        self.frequent_labels_ = {
            tag for tag, count in counts.items() if count >= self.threshold
        }
        logger.debug(
            "词频筛选完成，阈值=%d，保留高频标签数: %d，截断为 <UNK> 的标签数: %d",
            self.threshold,
            len(self.frequent_labels_),
            len(counts) - len(self.frequent_labels_),
        )

        # 3. 构建严格的类别列表（确保 unknown_label 一定存在）
        final_classes = sorted(list(self.frequent_labels_))
        if self.unknown_label not in final_classes:
            final_classes.append(self.unknown_label)

        # 4. 初始化带有严格维度的 MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer(classes=final_classes)
        processed_data = self._preprocess(parsed_data)
        self.mlb.fit(processed_data)

        self.is_fitted = True
        self.save_state()
        logger.info(
            "拟合完成。共保留 %d 个维度（包含 %s）。",
            len(final_classes),
            self.unknown_label,
        )
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        """执行转换，返回带有列名的 DataFrame"""
        if not self.is_fitted:
            logger.warning("转换器未拟合，尝试从配置文件加载状态...")
            self.load_state()

        logger.debug("开始转换，样本数: %d", len(series))
        parsed_data = series.apply(self.safe_parse).tolist()
        processed_data = self._preprocess(parsed_data)

        # 转换并生成特征矩阵
        matrix = self.mlb.transform(processed_data)

        # 生成规范化的列名，如 genre_Action
        col_prefix = series.name if series.name else "feature"
        col_names = [f"{col_prefix}_{c}" for c in self.mlb.classes_]

        logger.debug(
            "转换完成，输出特征维度: %d x %d", matrix.shape[0], matrix.shape[1]
        )
        return pd.DataFrame(matrix, columns=col_names, index=series.index)

    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        """拟合并转换（训练阶段常用）"""

        return self.fit(series).transform(series)

    def save_state(self):
        """将训练好的高频词和类别字典持久化到 JSON"""
        state = {
            "threshold": self.threshold,
            "unknown_label": self.unknown_label,
            "frequent_labels": list(self.frequent_labels_),
            "classes": list(self.mlb.classes_) if hasattr(self.mlb, "classes_") else [],
        }

        dir_name = os.path.dirname(os.path.abspath(self.config_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)

        logger.info("状态已成功保存至 %s", self.config_path)

    def load_state(self):
        """从 JSON 加载历史状态，用于线上推理环境"""
        if not os.path.exists(self.config_path):
            logger.error("找不到配置文件: %s，请先执行 fit()。", self.config_path)
            raise FileNotFoundError(
                f"找不到配置文件: {self.config_path}，请先执行 fit()。"
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.threshold = state.get("threshold", self.threshold)
        self.unknown_label = state.get("unknown_label", self.unknown_label)
        self.frequent_labels_ = set(state.get("frequent_labels", []))

        classes = state.get("classes", [])
        self.mlb = MultiLabelBinarizer(classes=classes)
        # 用全量类别做一个 dummy fit 激活内部状态
        self.mlb.fit([classes])

        self.is_fitted = True
        logger.info("状态已成功从 %s 加载，类别数: %d", self.config_path, len(classes))


if __name__ == "__main__":
    import pandas as pd
    import os

    # 确保 config 目录在 src 的上一级（即项目根目录下）
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    CONFIG_DIR = os.path.join(ROOT_DIR, "config")
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # 假设这是你的原始 DataFrame (此处截取了你给的一小段数据)
    data = [
        [1, 2, "['Adventure', 'Children', 'Fantasy']", "[29, 584, 204]"],
        [
            1,
            29,
            "['Adventure', 'Drama', 'Fantasy', 'Mystery', 'Sci-Fi']",
            "[287, 1092, 1090]",
        ],
        [1, 32, "['Mystery', 'Sci-Fi', 'Thriller']", "[419, 1027, 1028]"],
    ]
    df = pd.DataFrame(
        data, columns=["userId", "movieId", "genres_list", "item_top_genome_tags"]
    )

    # ==========================================
    # 阶段 1：训练阶段 (Training Pipeline)
    # ==========================================

    # 1. 实例化处理器，为不同的特征指定不同的 JSON 保存路径
    # 设置 threshold=2 表示出现至少2次才保留，否则变 <UNK_GENRE>
    genre_processor = RobustFrequencyLabelBinarizer(
        threshold=2,
        unknown_label="<UNK_GENRE>",
        config_path=os.path.join(CONFIG_DIR, "genres_config.json"),
    )

    genome_processor = RobustFrequencyLabelBinarizer(
        threshold=2,
        unknown_label="<UNK_GENOME>",
        config_path=os.path.join(CONFIG_DIR, "genome_config.json"),
    )

    # 2. 直接传入 Pandas Series 进行 fit_transform
    df_genres_features = genre_processor.fit_transform(df["genres_list"])
    df_genome_features = genome_processor.fit_transform(df["item_top_genome_tags"])

    # 3. 拼接到原始表
    final_train_df = pd.concat([df, df_genres_features, df_genome_features], axis=1)
    final_train_df = final_train_df.drop(
        columns=["genres_list", "item_top_genome_tags"]
    )

    print("--- 训练阶段生成的特征 ---")
    print(final_train_df.head(2))

    # ==========================================
    # 阶段 2：推理阶段 (Inference/Serving Pipeline)
    # ==========================================
    # 假设线上新来了一条数据，包含了没见过的题材 'Anime' 和没见过的基因 '9999'
    test_data = pd.DataFrame(
        [[99, 100, "['Adventure', 'Anime']", "[29, 9999]"]],
        columns=["userId", "movieId", "genres_list", "item_top_genome_tags"],
    )

    # 在线上环境中，你可以新建一个干净的实例，不调用 fit，直接调用 transform
    # 它会自动去 config/ 下读取刚刚存好的 json
    serve_genre_processor = RobustFrequencyLabelBinarizer(
        config_path=os.path.join(CONFIG_DIR, "genres_config.json")
    )
    serve_genome_processor = RobustFrequencyLabelBinarizer(
        config_path=os.path.join(CONFIG_DIR, "genome_config.json")
    )

    # 转换
    test_genres = serve_genre_processor.transform(test_data["genres_list"])
    test_genomes = serve_genome_processor.transform(test_data["item_top_genome_tags"])

    final_test_df = pd.concat([test_data, test_genres, test_genomes], axis=1)

    print("\n--- 推理阶段生成的特征 ---")
    print(final_test_df)  # 你会发现 'Anime' 没有生成新列，而是激活了 <UNK_GENRE> 列
