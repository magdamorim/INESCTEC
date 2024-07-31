from train1 import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel, AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import ast
import json


# this code is used to input a test index on test.csv file and then use it to retrieve the most similar cases

# Define the feature extraction class
class extract_features_single:
    def __init__(self, input_image_path, input_caption, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])):
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        self.text_encoder = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.input_image_path = input_image_path
        self.input_caption = input_caption
        self.transform = transform

    def extract_text_features(self):
        inputs = self.tokenizer(self.input_caption, return_tensors="pt")
        outputs = self.text_encoder(**inputs)
        hidden_states = outputs.last_hidden_state[:, 0, :]  # Consider the features corresponding to the SOS token
        self.text_features = hidden_states
        self.text_features = self.text_features.detach().cpu().numpy()
        print("Text features shape:", self.text_features.shape)
        return self.text_features

    def extract_image_features(self):
        img_path = os.path.join("/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT1M_data/all_images", self.input_image_path)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            image_features = self.image_encoder(image).last_hidden_state[:, 0, :]
        self.image_features = image_features.detach().cpu().numpy()
        print("Image features shape:", self.image_features.shape)
        return self.image_features


class multimodal_retrieval:
    def __init__(self):
        self.train_csv = pd.read_csv("/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT1M_data/train_data.csv")
        self.test_csv = pd.read_csv("/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT1M_data/test_data.csv")
        model_path = "/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/model1.pth"
        mlb_path = '/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/mlb1.pkl'
        self.model = Model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        with open(mlb_path, 'rb') as f:
            self.mlb = pickle.load(f)
        with open("/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/dataset.pkl", 'rb') as f:
            self.dataset = pickle.load(f)

    def retrieve_most_similar(self, index):
        input_label = self.test_csv['pathology'][index]
        input_label = ast.literal_eval(input_label)
        input_label = self.mlb.transform([input_label])[0]
        input_label = torch.tensor(input_label, dtype=torch.float32).unsqueeze(0)

        input_caption = self.test_csv['caption'][index]
        input_image_path = self.test_csv['image_path'][index]
        img_path = os.path.join("/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT1M_data/all_images", input_image_path)
        image = Image.open(img_path).convert('RGB')
        plt.imshow(image)
        plt.title(f"Caption: {input_caption}")
        plt.show()
        print("ground truth input label")
        print(input_label)

        extractor = extract_features_single(input_image_path, input_caption)
        extractor.extract_text_features()
        extractor.extract_image_features()

        self.scaler_image = StandardScaler()
        self.scaler_text = StandardScaler()
        
        extractor.image_features = self.scaler_image.fit_transform(extractor.image_features)
        extractor.text_features = self.scaler_text.fit_transform(extractor.text_features)
        
        extractor.image_features = torch.tensor(extractor.image_features, dtype=torch.float32)
        extractor.text_features = torch.tensor(extractor.text_features, dtype=torch.float32)

        with torch.no_grad():
            probabilities = self.model(extractor.text_features, extractor.image_features)
            predictions = (probabilities > 0.5).float()
        print("predicted input label")
        print(predictions)
        for i in range(len(self.train_csv)):
            label = self.train_csv['pathology'][i]
            label = ast.literal_eval(label)
            label = self.mlb.transform([label])[0]
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
            
          
            if torch.equal(predictions, label):
                image_path = self.train_csv['image_path'][i]
                img_path = os.path.join("/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT1M_data/all_images", image_path)
                # Check if the image file exists
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert('RGB')
                    plt.imshow(image)
                    plt.title(f"Caption: {self.train_csv['caption'][i]}")
                    plt.show()
                    print("Found a match with caption:", self.train_csv['caption'][i])
                    print(label)

if __name__ == "__main__":
    retrieval = multimodal_retrieval()
    retrieval.retrieve_most_similar(7304)
