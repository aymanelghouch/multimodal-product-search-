from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.features import color_histogram
from src.similarity import l2_normalize, top_k

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "processed" / "products.csv")

X = np.stack([color_histogram(Image.open(ROOT / p)) for p in df["image_path"]])
Xn = l2_normalize(X)


def show_query(query_idx: int, k: int = 5):
    idx, sc = top_k(Xn, query_idx, k)
    fig, axes = plt.subplots(1, k + 1, figsize=(2.2 * (k + 1), 3.2))

    ids = [query_idx, *idx]
    for ax, i, title in zip(axes, ids, ["CONSULTA"] + [f"{s:.3f}" for s in sc]):
        ax.imshow(Image.open(ROOT / df.loc[i, "image_path"]))
        ax.set_title(f"{title}\n{df.loc[i, 'articleType']}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


for q in [0, 100, 250, 500, 900]:
    show_query(q)