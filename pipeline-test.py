from transformers import pipeline
from PIL import Image
import requests
import torch

classifier = pipeline("image-classification")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
predictions = classifier(url)
predictions

# from transformers import ViTFeatureExtractor, ViTForImageClassification
# from PIL import Image
# import requests
# import torch
#
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
# feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
# inputs = feature_extractor(image, return_tensors="pt")
# inputs
# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
# outputs = model(**inputs)
# outputs
