import numpy as np


#Calcular nla norma d ecada fila de X y dividir cada fila por su norma
def l2_normalize(X: np.ndarray) -> np.ndarray: 
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms




def top_k(X: np.ndarray, query_idx: int, k: int) -> np.ndarray:
    """X ya normalizada, forma (n, d). Devuelve (indices, scores)."""

    scores = X @ X[query_idx] 
    scores[query_idx] = -np.inf        # paso 2
    idx = np.argsort(scores)[::-1][:k] # paso 3
    return idx, scores[idx]



    
