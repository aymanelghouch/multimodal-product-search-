import numpy as np
from PIL import Image


def color_histogram(img: Image.Image, bins: int = 8) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    pixels = arr.reshape(-1, 3)
    hist, _ = np.histogramdd(pixels, bins=bins, range=[(0, 256), (0, 256), (0, 256)])
    vec = hist.flatten() / hist.sum()
    return vec


def color_histogram_no_bg(img: Image.Image, bins: int = 8) -> np.ndarray:
    vec = color_histogram(img, bins)
    vec[-1] = 0                    # el cajón (7,7,7) es el último del vector aplanado
    return vec / vec.sum()