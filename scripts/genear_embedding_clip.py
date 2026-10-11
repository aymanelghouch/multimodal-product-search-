import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "processed" / "products.csv")

paths = [ROOT / p for p in df["image_path"]]

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
model.eval()
processor = CLIPProcessor.from_pretrained(model_name)

image = Image.open("data/images/1561.jpg")
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

print(embed_text(["black running shoes"]).shape())
print(embed_text(["a", "black running shoes"]).shape())



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



X = embed_images(paths[:70])
print(X.shape)
print(np.allclose(X[0], embed_images(paths[:1])[0], atol=1e-5))

