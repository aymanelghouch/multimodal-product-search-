from PIL import Image
import requests


from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open(requests.get("https://img.magnific.com/foto-gratis/cerrar-lindo-gato-interior_23-2148882585.jpg?semt=ais_hybrid&w=740&q=80", stream=True).raw)

options = ["a photo of a cat", 
           "a photo of a dog", 
           "a photo of a microscope"]

inputs = processor(text=options, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)


logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)



label_probabilities = {k: v for k, v in zip(options, probs[0].tolist())}

print(label_probabilities)