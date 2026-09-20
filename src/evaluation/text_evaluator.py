import numpy as np
import src.util.general as general_utils
from src.util.logger import Logger
def recall_at_k(scores, true_idx, k):
    topk = np.argsort(-scores, axis=1)[:, :2]
    hits = topk == true_idx[:, None]
    topk_mean = hits.any(axis=1).mean()
    Logger.debug(f"The recall of top_k results: {len(topk_mean)}")