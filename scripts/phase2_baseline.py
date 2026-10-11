import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.features import color_histogram
from src.features import color_histogram, color_histogram_no_bg
from src.similarity import l2_normalize, top_k
from src.evaluation import precision_at_k


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "processed" / "products.csv"

df = pd.read_csv(CSV)

# --- ejercicio 1 ---
img = Image.open(ROOT / df.loc[0, "image_path"])
arr = np.array(img)
print("PIL size:", img.size)
print("NumPy shape:", arr.shape)

# --- ejercicio 2 ---
vec = color_histogram(img)
print(vec.shape, vec.sum())
bin_idx = np.argmax(vec)
print(np.unravel_index(bin_idx, (8, 8, 8)))
print("peso del blanco:", vec[np.ravel_multi_index((7, 7, 7), (8, 8, 8))])

# --- ejercicio 3 ---
t0 = time.perf_counter()
vectors = []
for rel_path in df["image_path"]:
    img = Image.open(ROOT / rel_path)
    vectors.append(color_histogram(img))
X = np.stack(vectors)
print("tiempo:", round(time.perf_counter() - t0, 2), "s")

print(X.shape)
print(X.dtype)
print(X.sum(axis=1)[:5])

# --- ejercicio 4: normalizar ---
print("norma antes:", np.linalg.norm(X, axis=1)[:5])
Xn = l2_normalize(X)
print("norma después:", np.linalg.norm(Xn, axis=1)[:5])   # deben ser 1

# --- ejercicio 5: buscar ---
idx, sc = top_k(Xn, query_idx=0, k=5)

print("Consulta:", df.loc[0, "articleType"], "|", df.loc[0, "productDisplayName"])
for i, s in zip(idx, sc):
    print(f"  {s:.3f}  {df.loc[i, 'articleType']:15s} {df.loc[i, 'productDisplayName']}")

print(0 in idx)    # False
print(len(idx))    # 5


labels = df["articleType"].to_numpy()
print("Precision@5 (histograma):", round(precision_at_k(Xn, labels, k=5), 3))


X2 = np.stack([color_histogram_no_bg(Image.open(ROOT / p)) for p in df["image_path"]])
Xn2 = l2_normalize(X2)
print("Precision@5 (sin fondo):", round(precision_at_k(Xn2, labels, k=5), 3))