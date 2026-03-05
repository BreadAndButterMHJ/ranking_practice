import pandas as pd
import lightgbm as lgb
from .logger_config import get_logger

logger = get_logger(__name__)


def inference(model_path: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    推理阶段：加载模型，对新数据进行预测。

    参数:
        model_path: 模型文件路径。
        data: 包含特征的新数据。
    返回:
        预测结果。
    """
    logger.info("开始加载模型...")
    try:
        model = lgb.Booster(model_file=model_path)
    except Exception as e:
        logger.error("模型加载失败！错误信息: %s", e)
        raise e
    logger.info("模型加载完成！")
    return model.predict(data)
