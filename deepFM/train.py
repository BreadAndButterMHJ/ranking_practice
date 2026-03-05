import pandas as pd
import numpy as np
import ast
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from deepctr_torch.inputs import (
    SparseFeat,
    DenseFeat,
    VarLenSparseFeat,
    get_feature_names,
)
from deepctr_torch.models import DeepFM


def _pad_sequences(seqs, maxlen, padding="post", truncating="post", value=0):
    """用 numpy 实现序列 padding，不依赖 TensorFlow。"""
    seqs = [np.asarray(s, dtype=np.int64) for s in seqs]
    if truncating == "post":
        seqs = [s[:maxlen] for s in seqs]
    else:
        seqs = [s[-maxlen:] for s in seqs]
    result = np.full((len(seqs), maxlen), value, dtype=np.int64)
    for i, s in enumerate(seqs):
        if len(s) > 0:
            result[i, : len(s)] = s
    return result


class _SimpleTokenizer:
    """简单词表与序列数值化，兼容原 Tokenizer 的 fit_on_texts / texts_to_sequences。"""

    def __init__(self):
        self.word_index = {}  # token -> id，从 1 开始，0 保留给 padding

    def fit_on_texts(self, list_of_tokens_list):
        for tokens in list_of_tokens_list:
            for t in tokens:
                if t not in self.word_index:
                    self.word_index[t] = len(self.word_index) + 1
        return self

    def texts_to_sequences(self, list_of_tokens_list):
        return [
            [self.word_index.get(t, 1) for t in tokens]  # 未登录词用 1
            for tokens in list_of_tokens_list
        ]


# 1. 模拟你的数据加载 (正常使用 df = pd.read_csv('your_data.csv'))
# 至少包含 0/1 两类标签，否则 AUC 等指标会报错
data = {
    "userId": [1, 1, 2],
    "movieId": [2, 29, 3],
    "label": [0, 0, 1],
    "user_rating_count": [175, 175, 50],
    "user_rating_mean": [3.74, 3.74, 4.2],
    "user_top_tags": ["[]", "[]", "[]"],
    "genres_list": [
        "['Adventure', 'Children', 'Fantasy']",
        "['Adventure', 'Drama', 'Fantasy', 'Mystery', 'Sci-Fi']",
        "['Action', 'Comedy']",
    ],
    "item_rating_count": [22243.0, 8520.0, 1000.0],
    "item_rating_mean": [3.21, 3.95, 4.0],
    "item_top_genome_tags": [
        "[29, 584, 204, 588, 951]",
        "[287, 1092, 1090, 995, 535]",
        "[1, 2, 3]",
    ],
    "hour": [23, 23, 12],
    "dayofweek": [5, 5, 0],
}
df = pd.DataFrame(data)

# 2. 定义特征列的名称
sparse_features = ["userId", "movieId", "hour", "dayofweek"]
dense_features = [
    "user_rating_count",
    "user_rating_mean",
    "item_rating_count",
    "item_rating_mean",
]
varlen_features = ["user_top_tags", "genres_list", "item_top_genome_tags"]
target = ["label"]

# ==================== 3. 特征预处理 ====================

# 3.1 缺失值填充
df[sparse_features] = df[sparse_features].fillna("-1")
df[dense_features] = df[dense_features].fillna(0)

# 3.2 Sparse 特征编码 (Label Encoding)
for feat in sparse_features:
    lbe = LabelEncoder()
    df[feat] = lbe.fit_transform(df[feat])

# 3.3 Dense 特征归一化 (MinMaxScaler)
mms = MinMaxScaler(feature_range=(0, 1))
df[dense_features] = mms.fit_transform(df[dense_features])


# 3.4 VarLen Sparse 多值特征处理 (以字符串列表转义、Tokenize和Padding为例)
# 定义一个辅助函数：将 "[1, 2]" 这种字符串转换为真实的 Python 列表
def parse_list_string(s):
    try:
        parsed = ast.literal_eval(s)
        # 将内容统一转为字符串方便 Tokenizer 处理
        return [str(i) for i in parsed]
    except:
        return []


# 存储变长特征的 pad 结果和最大长度字典
pad_sequences_dict = {}
max_len_dict = {
    "user_top_tags": 5,
    "genres_list": 5,
    "item_top_genome_tags": 5,
}  # 可根据实际数据分布调整最大长度
vocab_size_dict = {}

for feat in varlen_features:
    # 1. 字符串转列表
    list_col = df[feat].apply(parse_list_string)

    # 2. 建立词典并进行数值化 (Tokenization)，使用不依赖 TensorFlow 的简单 Tokenizer
    tokenizer = _SimpleTokenizer()
    tokenizer.fit_on_texts(list_col)
    seqs = tokenizer.texts_to_sequences(list_col)

    # 3. Padding 对齐长度 (补 0)
    pad_seqs = _pad_sequences(
        seqs, maxlen=max_len_dict[feat], padding="post", truncating="post"
    )

    pad_sequences_dict[feat] = pad_seqs
    vocab_size_dict[feat] = len(tokenizer.word_index) + 1  # +1 是因为 padding 的 0

# ==================== 4. 构建 DeepCTR 的特征列 (Feature Columns) ====================

# 4.1 Sparse 特征列（deepctr_torch: SparseFeat(name, vocabulary_size, embedding_dim=4)）
fixlen_feature_columns = [
    SparseFeat(feat, df[feat].nunique(), embedding_dim=4) for feat in sparse_features
]

# 4.2 Dense 特征列（deepctr_torch: DenseFeat(name, dimension, dtype)）
fixlen_feature_columns += [DenseFeat(feat, 1) for feat in dense_features]

# 4.3 VarLen 特征列
varlen_feature_columns = [
    VarLenSparseFeat(
        SparseFeat(feat, vocab_size_dict[feat], embedding_dim=4),
        maxlen=max_len_dict[feat],
        combiner="mean",
    )
    for feat in varlen_features
]

# 合并所有特征列（DeepFM 包括线性的 Linear 部分和深度的 DNN 部分，通常传入所有特征）
linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# ==================== 5. 组装输入数据 (Input Dictionary) ====================
model_input = {name: df[name].values for name in sparse_features + dense_features}

# 将 VarLen 特征加入到输入字典中
for feat in varlen_features:
    model_input[feat] = pad_sequences_dict[feat]

# (可选) 划分训练集和测试集（这里仅作演示，直接用全量数据）
# X_train, X_test, y_train, y_test = train_test_split(...)

# ==================== 6. 构建并训练 DeepFM 模型 (deepctr_torch) ====================
# 设备：有 GPU 则用 cuda，否则 cpu
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 实例化 DeepFM 模型（deepctr_torch 需传入 device）
model = DeepFM(
    linear_feature_columns=linear_feature_columns,
    dnn_feature_columns=dnn_feature_columns,
    task="binary",
    device=device,
)

# 编译模型（二分类任务使用 binary_crossentropy，metrics 小写 auc）
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["binary_crossentropy", "auc"]
)

# 训练模型（样本很少时用 validation_split=0 避免验证集形状问题；数据多时可改为 0.2）
history = model.fit(
    model_input,
    df[target].values,
    batch_size=256,
    epochs=10,
    verbose=2,
    validation_split=0.0,
)

# ==================== 7. 预测 ====================
pred_ans = model.predict(model_input, batch_size=256)
print("模型预测输出概率：\n", pred_ans)
