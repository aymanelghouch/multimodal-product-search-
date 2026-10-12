import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel



model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
model.eval()
processor = CLIPProcessor.from_pretrained(model_name)


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