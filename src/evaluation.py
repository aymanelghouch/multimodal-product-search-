import numpy as np

from src.similarity import top_k



def  precision_at_k(Xn: np.ndarray, labels: np.ndarray, k: int = 5) -> float:

    precision = []

    for i in range(Xn.shape[0]):
        idx, _ = top_k(Xn, i, k)                
        p = (labels[idx] == labels[i]).mean()   
        precision.append(p)

    return float(np.mean(precision))

