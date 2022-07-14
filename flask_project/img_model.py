

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import requests
import numpy as np
from io import BytesIO
import torch
import tensorflow.python.keras.applications
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from tensorflow.python.keras.preprocessing import image

from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms

from PIL import ImageOps
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import layers, losses

from tensorflow.python.keras.models import Model

import cv2


latent_dim = 13872

class Autoencoder(Model):
  def __init__(self, encoding_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(1, activation='sigmoid'),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

#autoencoder = Autoencoder(latent_dim)



root = os.path.join(os.getcwd(), "crawl_img")

class FeatureExtractor:
  def __init__(self):
    # Use VGG-16 as the architecture and ImageNet for the weight
        base_model = VGG16(weights='imagenet')
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
  def extract(self, img):
    # Resize the image
      img = img.resize((224, 224))
      # Convert the image color space
      img = img.convert('RGB')
      # Reformat the image
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      # Extract Features
      feature = self.model.predict(x)[0]
      return feature / np.linalg.norm(feature)


q_root = os.path.join(os.getcwd(),"fake_images")
query_root = os.path.join(q_root, "fake_img.png")

#print(query_root)
# 페이크 이미지 띄워 보기
import cv2
from PIL import ImageOps
#print(1)
#img2 = Image.open("C:\\Users\\qorgh2akfl\\Desktop\\flask_server\\fake_images\\fake_img.png")
img2 = Image.open(query_root)
#print(query_root)
size = 68,68
img2 = ImageOps.fit(img2, size, Image.ANTIALIAS)
#print(img2.size)
#img2.show()

# 피처 이미지 생성

feature_root = os.path.join(os.getcwd(), "feature_image")
#
#
from pathlib import Path

for img_path in sorted(os.listdir(root)):
    # print(img_path)
    new_root = os.path.join(root, img_path)
    fe = FeatureExtractor()
    # Extract Features
    feature = fe.extract(img=Image.open(new_root))

    # # Save the Numpy array (.npy) on designated path
    feature_path = Path(feature_root) / (img_path + ".npy")
    print(feature_path)
    print(img_path + ".npy")
    np.save(feature_path, feature)
print(feature)

# 유사도 계산
import matplotlib.pyplot as plt
import numpy as np

list = []


def read_image(img_path):
    try:
        # print(img_path)
        response = requests.get(img_path)
        img_path = BytesIO(response.content)
    except:
        pass
    img = Image.open(img_path)
    return img


# Insert the image query
img = Image.open(query_root)
size = 68, 68
img = ImageOps.fit(img, size, Image.ANTIALIAS)
#print(img.size)
fe = FeatureExtractor()
#feature_path2 = feature_path
# Extract its features
# query = vgg16_extractor(query_root)
query = fe.extract(img)
img_obj = read_image(query_root)
kk = ImageOps.fit(img_obj, size, Image.ANTIALIAS)

img_numpy = np.array(kk)
plt.imshow(img_numpy)
plt.title('Query image')
#plt.show()

# Calculate the similarity (distance) between images
# print(feature_root)
list = []
for img_item in sorted(os.listdir(feature_root)):
    # print(img_item)
    feature_item = os.path.join(feature_root, img_item)
    new_item = os.path.join(root, img_item)
    kk = os.path.splitext(img_item)
    print(kk[0])
    new_item = os.path.join(root, kk[0])
    # Extract Features
    features = np.load(feature_item)
    # print(features)
    # feature_item = os.path.join(root, featuree)
    # features = fe.extract(img=Image.open(feature_item))

    # features = fe.extract(img=Image.open(new_root))

    dists = np.linalg.norm(features - query, axis=0)
    # print(dists[0])
    print(dists)
    #   #print(dists)
    # # # # Extract 30 images that have lowest distance
    ids = np.argsort(dists)[:10]
    # print(ids)
    scores = [(dists, new_item) for id in ids]
    for i, (score, path) in enumerate(scores):
        feature_obj = read_image(new_item)
        feature_numpy = np.array(feature_obj)
        plt.imshow(feature_numpy)
        list.append(scores)
        text = f'Image: {os.path.basename(query_root)} / Similarity: {score}'
        plt.title(text)
        #plt.show()
print(min(list))
min_list = min(list)
min_list = str(min_list)
split_list=min_list.split(', ')
sim = split_list[0]
simi = sim[2:]
pa = split_list[1]
pat = pa[:-3]
pat = pat[1:]
print(pat)

new_pat = pat.replace('\\\\','/')
print(new_pat)
similar_site = Image.open(new_pat)
similar_site.save("./static/similar_site/site_img.png",'png')

def re(new_pat):
    nn = new_pat[51:]
    newnew_pat = nn[:-4]
    print(newnew_pat)
    return newnew_pat
var=re(new_pat)
#simi_img = read_image(pat)
# plt.imshow(plt.imshow())
# plt.show()
# print(simi)
# print(pat)
# print()


label = []
data = []
csv_test = pd.read_csv('C:\\Users\\qorgh2akfl\\Desktop\\flask_server\\df.csv')
f_row = csv_test.loc[csv_test["name"] == var]
name = f_row["name"].astype("string")
tel = f_row["tel"].astype("string")
address = f_row["address"].astype("string")
latitude = f_row["latitude"].astype("string")
longitude = f_row["longitude"].astype("string")

print(name)
print(name.values)
print(name.to_string(index=False))