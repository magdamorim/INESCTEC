
# this is a model to perform retrieval of the k=3 most significant cases- CASE BASED EXPLANATION! The idea is to use a classififier to 
# #classify each image as multilabel classification and then retrieve the cases with the same labels. 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, text_dim: int = 768, image_dim: int = 768, fusion_dim: int = 256, hidden_dim: int = 512, num_labels: int = 15, num_heads: int = 8):
        super(Model, self).__init__()
        
        # Text and Image Feature Extractors
        self.text_fc1 = nn.Linear(text_dim, hidden_dim)
        self.image_fc1 = nn.Linear(image_dim, hidden_dim)

        # Batch Normalization
        self.batch_norm_text1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm_image1 = nn.BatchNorm1d(hidden_dim)
        
        # Multihead Attention Layer
        self.embed_dim = hidden_dim  # Must match the hidden_dim
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)

        # Fusion and Classification Layers
        self.fusion_fc1 = nn.Linear(hidden_dim, fusion_dim)  # Input to this layer is hidden_dim
        self.layer_norm_fusion = nn.LayerNorm(fusion_dim)
        
        self.classification_fc = nn.Linear(fusion_dim, num_labels)

    def forward(self, text_input: torch.Tensor, image_input: torch.Tensor) -> torch.Tensor:
        # Text Features Extraction
        text_features = F.relu(self.text_fc1(text_input))
        text_features = self.batch_norm_text1(text_features)
        
        # Image Features Extraction
        image_features = F.relu(self.image_fc1(image_input))
        image_features = self.batch_norm_image1(image_features)
        
        # Combine Features
        combined_features = torch.cat((text_features.unsqueeze(1), image_features.unsqueeze(1)), dim=1)  # Shape: (batch_size, 2, hidden_dim)
        
        # Multihead Attention
        attn_output, _ = self.multihead_attn(combined_features, combined_features, combined_features)
        attn_output = attn_output[:, 0, :]  # Select the output of the first feature (text)
        
        # Fusion Layer
        fusion_features = F.relu(self.fusion_fc1(attn_output))
        fusion_features = self.layer_norm_fusion(fusion_features)

        # Classification
        logits = self.classification_fc(fusion_features)
        
        # Apply sigmoid to logits for multi-label classification
        output = torch.sigmoid(logits)  # Sigmoid ensures output is in the range [0, 1] for each label
        
        return output



class CustomDataset(Dataset):
    def __init__(self, text_features, image_features, labels):
        self.text_features = text_features
        self.image_features = image_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text_data = self.text_features[index]
        image_data = self.image_features[index]
        label = self.labels[index]
        return text_data, image_data, label


import os

class Train:
    def __init__(self):
        # Load data and initialize everything as before
        with open("/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/labels_train.pkl", 'rb') as file:
            self.labels_train = pickle.load(file)
        
        self.image_features_train = np.load("/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/image_features_train.npy")
        self.text_features_train = np.load("/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/text_features_train.npy")
        self.scaler_image = StandardScaler()
        self.scaler_text = StandardScaler()

        self.image_features_train= self.scaler_image.fit_transform(self.image_features_train)
        self.text_features_train = self.scaler_text.fit_transform(self.text_features_train)
        
        
        self.image_features_train = torch.tensor(self.image_features_train, dtype=torch.float32)
        self.text_features_train = torch.tensor(self.text_features_train, dtype=torch.float32)
        
        self.mlb = MultiLabelBinarizer()
        self.labels_train = self.mlb.fit_transform(self.labels_train)
        self.labels_train= torch.tensor(self.labels_train, dtype=torch.float32)
        
        self.dataset_train = CustomDataset(self.text_features_train, self.image_features_train, self.labels_train)
        self.dataloader_train = DataLoader(self.dataset_train, batch_size=32, shuffle=True)
        
        
        with open("/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/labels_test.pkl", 'rb') as file:
            self.labels_test = pickle.load(file)
    
        
        self.image_features_test = np.load("/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/image_features_test.npy")
        self.text_features_test = np.load("/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/text_features_test.npy")
        self.scaler_image = StandardScaler()
        self.scaler_text = StandardScaler()

        self.image_features_test= self.scaler_image.fit_transform(self.image_features_test)
        self.text_features_test = self.scaler_text.fit_transform(self.text_features_test)
        
        
        self.image_features_test = torch.tensor(self.image_features_test, dtype=torch.float32)
        self.text_features_test = torch.tensor(self.text_features_test, dtype=torch.float32)
        
    
        self.labels_test = self.mlb.transform(self.labels_test)
        self.labels_test = torch.tensor(self.labels_test, dtype=torch.float32)

        
        self.dataset_test = CustomDataset(self.text_features_test, self.image_features_test, self.labels_test)
        self.dataloader_test = DataLoader(self.dataset_test, batch_size=2, shuffle=True)
        
        self.model = Model()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, epochs=10):
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for text_data, image_data, labels in tqdm(self.dataloader_train):
                text_data = text_data.to(self.device)
                image_data = image_data.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(text_data, image_data)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(self.dataloader_train)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

        print("Training complete.")

    def evaluate(self): 
        self.model.eval()
        
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for text_data, image_data, labels in tqdm(self.dataloader_test):
                text_data = text_data.to(self.device)
                image_data = image_data.to(self.device)
                labels = labels.to(self.device)

                probabilities = self.model(text_data, image_data)
                predictions = (probabilities > 0.5).float()

                all_labels.append(labels.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
        print(all_labels)
        print(all_predictions)

        all_labels = np.vstack(all_labels)
        all_predictions = np.vstack(all_predictions)
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Test set accuracy: {accuracy:.4f}")
       
    def print_mapping_labels(self):
        label_names = self.mlb.classes_
        for idx, label in enumerate(label_names):
            print(f"Index {idx}: {label}")
        print("Label mapping:")
        print(label_names)

    def save_model(self, model_path: str = '/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/model1.pth', encoder_path: str = '/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/mlb1.pkl'):
        # Save model state dictionary
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Save MultiLabelBinarizer
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.mlb, f)
        print(f"Label encoder saved to {encoder_path}")
        
        
    def load_model(self):
        model_path="/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/model1.pth"
        mlb_path='/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/mlb1.pkl'
        self.model = Model()
        self.model.load_state_dict(torch.load(model_path))



        
        



if __name__ == "__main__":
    model = Train()
    model.train(epochs=100)
    model.save_model()
    print(model.labels_test)
    model.load_model
    model.evaluate()
    
