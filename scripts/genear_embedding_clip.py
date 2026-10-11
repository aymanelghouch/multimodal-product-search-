import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from pathlib import Path
import pandas as pd
import time
from src.evaluation import precision_at_k
from src.similarity import l2_normalize




ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "processed" / "products.csv")
labels = df["articleType"].to_numpy()
paths = [ROOT / p for p in df["image_path"]]


model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
model.eval()
processor = CLIPProcessor.from_pretrained(model_name)

# 3. Procesar la imagen para el modelo
#inputs = processor(images=image, return_tensors="pt")

#with torch.no_grad():
 #   image_features = model.get_image_features(**inputs)


#print(image_features)


def embed_text(texts):

    inputs = processor(text=texts, return_tensors="pt", padding=True)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    return text_features["pooler_output"].numpy()


def embed_images(image_path,  batch_size=32):
    all_features = []
    
    for i in range(0, len(image_path), batch_size):
        batch_path = image_path[i: i + batch_size]
        images = [Image.open(p).convert('RGB') for p in batch_path]

        inputs = processor(images=images, return_tensors='pt')


        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        image_features = image_features['pooler_output'].numpy()
        all_features.append(image_features)

    return np.concatenate(all_features)


t0 = time.perf_counter()
X_clip = embed_images(paths)
elapsed = time.perf_counter() - t0

print("forma:", X_clip.shape)
print("dtype:", X_clip.dtype)
print("tiempo:", round(elapsed, 1), "s")

OUT = ROOT / "data" / "processed" / "clip_image_embeddings.npy"
np.save(OUT, X_clip)

X_check = np.load(OUT)
print(X_check.shape, X_check.dtype)
print("idéntico:", np.array_equal(X_clip, X_check))

Xn_clip = l2_normalize(X_clip)

def search_text(query, k=5):
    q = l2_normalize(embed_text([query]))[0]   # (1, 512) -> normalizado -> (512,)
    scores = Xn_clip @ q                       # (1000, 512) @ (512,) -> (1000,)
    idx = np.argsort(scores)[::-1][:k]         # los k scores más altos, de mayor a menor
    return idx, scores[idx]


for query in ["black running shoes", "silver wrist watch", "blue denim shirt"]:
    idx, sc = search_text(query)
    print(f"\n'{query}'")
    for i, s in zip(idx, sc):
        print(f"  {s:.3f}  {df.loc[i, 'articleType']:15s} {df.loc[i, 'productDisplayName']}")

norms = np.linalg.norm(Xn_clip, axis=1)
print("normas antes:", np.linalg.norm(X_clip, axis=1)[:5])
print("normas después:", norms[:5])
print("todas ≈ 1:", np.allclose(norms, 1.0))

p_clip = precision_at_k(Xn_clip, labels, k=5)

print("Precision@5 azar:       0.039")
print("Precision@5 histograma: 0.226")
print("Precision@5 CLIP:      ", round(p_clip, 3))



