# %% [markdown]
# # Finetuning on MIT Indoor Scenes
# 
# https://www.kaggle.com/itsahmad/indoor-scenes-cvpr-2019
# 
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_the_%F0%9F%A4%97_Trainer.ipynb
# 

# %%
from transformers import AdamW, ViTFeatureExtractor, ViTForImageClassification
import PIL
import torch
import pprint
from pathlib import Path
import os
import datasets


# %% [markdown]
# # Read data

# %%
# data_base_path = "/Users/vincent/datasets/mit_indoor_scenes/dataset/indoorCVPR_09/Images/"
# MODEL = "google/vit-base-patch16-224"
MODEL = "google/vit-base-patch16-224-in21k"
data_base_path = "/Users/vincent/datasets/mit_indoor_scenes/sample/"
data_directory = Path(data_base_path)
classes = [ dir_.name for dir_ in data_directory.iterdir()]
class_no = len(classes)
pprint.pformat(classes)

# %%
# path = "/Users/vincent/datasets/mit_indoor_scenes/dataset/indoorCVPR_09/Images/airport_inside/airport_inside_0201.jpg"
# image = PIL.Image.open(path)
# feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL)
# inputs = feature_extractor(images=image, return_tensors="pt")
# inputs

# %% [markdown]
# ## Filter out JPEG Images

# %%
# python dicts are ordered since py3.6
# test_files_pixels_map = {str(dir_):  }
test_files_pixels_map_possible = {}
test_files_pixels_map_not_possible = {}
# we only want jpeg type of images to avoid downstream errors
for path in Path(data_base_path).rglob("*"):
    path_as_str = str(path)
    print(path_as_str)
    if os.path.isfile(path):
        pixels = PIL.Image.open(path_as_str)
        if isinstance(pixels, PIL.JpegImagePlugin.JpegImageFile):
            test_files_pixels_map_possible[path_as_str] = pixels
        else:
            test_files_pixels_map_not_possible[path_as_str] = pixels



# for path, pixels in test_files_pixels_map.items():
#     if os.path.isfile(path):
#         if isinstance(pixels, PIL.JpegImagePlugin.JpegImageFile):
#                 test_files_pixels_map_possible[path] = pixels
#         else:
#                 test_files_pixels_map_not_possible[path] = pixels
pixels = list(test_files_pixels_map_possible.values())
paths = list(test_files_pixels_map_possible.keys())

# %% [markdown]
# ## Extract Features

# %%

feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL)
batch = feature_extractor(images=pixels, return_tensors="pt")
batch

# %%
labels = [path.split("/")[-2] for path in paths]
pprint.pformat(set(labels))

# %%
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
targets = le.fit_transform(labels)
targets = torch.as_tensor(targets)
targets

# add labels to batch
batch['labels'] = targets


# %%
# inputs = feature_extractor(image, return_tensors="pt")
# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
# outputs = model(**batch)
# probabilities = outputs.logits.softmax(-1)[0]
# scores, ids = probabilities.topk(10)
# predictions = [{"score": score.item(), "label": model.config.id2label[_id.item()]} for score, _id in zip(scores, ids)]
# pprint.pprint(predictions)


# %%
pixel_values = [ image for image in batch["pixel_values"]]
batch["pixel_values"] = pixel_values
batch

# %%
from torch.utils.data import DataLoader
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# %%
from transformers import TrainingArguments, Trainer

# Training Args
args = TrainingArguments(
    f"mit-indoor-scenes",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='logs',
    remove_unused_columns=False,
)


# evaluation
from datasets import load_metric
import numpy as np

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


# model

id2label = {id:label for id, label in zip(targets.tolist(), labels)}
label2id = {label:id for id,label in zip(targets.tolist(), labels )}
num_labels = len(set(labels))


pprint.pprint(f"id2label: {id2label}")
pprint.pprint(f"label2id: {label2id}")
pprint.pprint(f"num labels: {num_labels}")


# %%


model = ViTForImageClassification.from_pretrained(
            MODEL,
            num_labels=2,
            id2label=id2label,
            label2id=label2id
    )


# %%
batch["pixel_values"]

# %%
from datasets import Dataset
# my_dict = {"a": [1, 2, 3]}
features = {"pixel_values": torch.tensor, "labels": int}
dataset = Dataset.from_dict(mapping=batch)
dataset

# %%
# [ for row in dataset]

# %%

# trainer
trainer = Trainer(
    model,
    args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
)

trainer.train()

# %%
from transformers import AdamW

model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
    )

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()

# %%
# path = "/Users/vincent/datasets/mit_indoor_scenes/sample/library/450px_Bibliothek_im_Reformierten_Kollegium_Debrecen.jpg"
# image = PIL.Image.open(path)
# feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
# inputs = feature_extractor(images= image, return_tensors="pt")
# outputs = model(**inputs)
# probabilities = outputs.logits.softmax(-1)[0]
# scores, ids = probabilities.topk(10)
# predictions = [{"score": score.item(), "label": model.config.id2label[_id.item()]} for score, _id in zip(scores, ids)]
# pprint.pprint(predictions)

# %%
image.show()

# %%



