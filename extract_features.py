import torch
from transformers import AutoModel, AutoTokenizer
from transformers import ViTModel
import numpy as np
from tqdm import tqdm
import pickle
import json
import ast


# This code as inputs the images of quilt dataset in folder all_images and dataset.pkl file with paths and captions and pathologies for training and test
# It produces files with the image and text features and labels for training and test dataset; those features were extracted using BioBert and ViT.
#Then they are saved 




class extract_features():
    def __init__(self):
        super(extract_features, self).__init__()  # Call the constructor of the base class
        path = '/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/dataset.pkl'
        with open(path, 'rb') as file:
          self.dataset = pickle.load(file)
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        self.text_encoder = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.train_dataset = self.dataset['train']
        self.test_dataset = self.dataset['test']

    def extract_text_features(self): # as shape n_captions x 768
        self.text_features = []
        for text in tqdm(self.train_dataset['captions']):
            inputs = self.tokenizer(text, return_tensors="pt")
            # Forward pass, get hidden states
            outputs = self.text_encoder(**inputs)
            hidden_states = outputs.last_hidden_state[:, 0, :] # consider the features corresponding to SOS
            #print(hidden_states.shape)
            self.text_features.append(hidden_states)
            print(hidden_states.shape)

        # Convert list of tensors to a single tensor and then to a NumPy array
        self.text_features = torch.cat(self.text_features, dim=0).detach().cpu().numpy()
        print("text features shape")
        print(self.text_features.shape)
        np.save('/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/text_features_train.npy', self.text_features)
        
        self.text_features = []
        for text in tqdm(self.test_dataset['captions']):
            inputs = self.tokenizer(text, return_tensors="pt")
            # Forward pass, get hidden states
            outputs = self.text_encoder(**inputs)
            hidden_states = outputs.last_hidden_state[:, 0, :] # consider the features corresponding to SOS
            #print(hidden_states.shape)
            self.text_features.append(hidden_states)
            print(hidden_states.shape)

        # Convert list of tensors to a single tensor and then to a NumPy array
        self.text_features = torch.cat(self.text_features, dim=0).detach().cpu().numpy()
        print("text features shape")
        print(self.text_features.shape)
        np.save('/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/text_features_test.npy', self.text_features)


        return self.text_features
    def extract_image_features(self): # as shape n_captions x 768
      self.image_features = []
      for image in tqdm(self.train_dataset['images']):
        image = self.image_encoder(image.unsqueeze(0)).last_hidden_state[:, 0, :]
        self.image_features.append(image)
        print(image.shape)
      self.image_features = torch.cat(self.image_features, dim=0).detach().cpu().numpy()
      print("image features shape")
      print(self.image_features.shape)
      

      np.save('/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/image_features_train.npy', self.image_features)
      
      self.image_features = []
      for image in tqdm(self.test_dataset['images']):
        image = self.image_encoder(image.unsqueeze(0)).last_hidden_state[:, 0, :]
        self.image_features.append(image)
        print(image.shape)
      self.image_features = torch.cat(self.image_features, dim=0).detach().cpu().numpy()
      print("image features shape")
      print(self.image_features.shape)
      

      np.save('/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/image_features_test.npy', self.image_features)
      
    def extract_labels(self):
      self.labels = []
      for label in tqdm(self.train_dataset['labels']):
    
        self.labels.append(label)
      print("labels shape")
      print("labels shape")
      print(len(self.labels))
      output_file = '/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/labels_train.pkl'
      

      with open(output_file, 'wb') as f:
          pickle.dump(self.labels, f)
    

      
      
      self.labels = []
      for label in tqdm(self.test_dataset['labels']):
  
        self.labels.append(label)
      print("labels shape")
      print(len(self.labels))
      output_file = '/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/labels_test.pkl'
      with open(output_file, 'wb') as f:
          pickle.dump(self.labels, f)

  

  


if __name__ == "__main__":       
  extractor=extract_features()
  extractor.extract_labels()
  #extractor.extract_text_features()
  #extractor.extract_image_features()