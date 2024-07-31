import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from transformers import ViTFeatureExtractor, ViTModel
from tqdm import tqdm
import pickle
import shutil
import ast

#**Code to create a clean dataset and save it**
# This code as inputs the images of quilt dataset in folder data and test_data.csv and train_data.csv files with paths and captions and pathologies.

# dataset.pkl is a dic with train and test keys.  Each is dic with 'images', 'captions'
#and 'labels', where captions is text, images is a tensor 3x256x256 and labels is a list of pathologies.


class QUILT1M(Dataset):
    def __init__(self, root_dir, csv_file_test, csv_file_train, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images of train/test.
            csv_file (string): Path to the CSV file with annotations from train/test.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Read the CSV file
        self.data_train= pd.read_csv(csv_file_train)
        self.data_test = pd.read_csv(csv_file_test)

    def __len__(self):
        return len(self.data)


    def create_dataset(self):
        
        Images=[]
        Captions=[]
        Labels=[]

        for idx, row in tqdm(self.data_test.iterrows(), total=len(self.data_test)):
            img_path = os.path.join(self.root_dir, row['image_path'])
            #print(img_path)
            if os.path.exists(img_path):
                print('found')
                print(idx)
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                    Images.append(image)
                    Captions.append(row['caption'])
                    Labels.append(ast.literal_eval(row['pathology']))
                
               
        test={'images':Images, 'captions':Captions, 'labels':Labels}
        print(len(test['labels']))


        # Read the CSV file
        Images=[]
        Captions=[]
        Labels=[]

        for idx, row in tqdm(self.data_train.iterrows(), total=len(self.data_train)):
            img_path = os.path.join(self.root_dir, row['image_path'])
            #print(img_path)
            if os.path.exists(img_path):
                print('found')
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                    Images.append(image)
                    Captions.append(row['caption'])
                    Labels.append(ast.literal_eval(row['pathology']))
                
        train={'images':Images, 'captions':Captions, 'labels':Labels}
        print(len(train['labels']))

        
        # Convert valid data to DataFrame
        self.dataset = {'train':train, 'test':test}
    def save_dataset(self, file_path):
        print("number of training data points")
        print(len(self.dataset['train']['images']))
        print("number of test data points")
        print(len(self.dataset['test']['images']))
        # Save the dataset to a file using pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self.dataset, f)
            
            
            
            
if __name__ == "__main__":   
    print("start")     

    #(update these paths accordingly)
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])
    root_dir= '/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT1M_data/all_images'
    csv_file_test= '/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT1M_data/test_data.csv'
    csv_file_train = '/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT1M_data/train_data.csv'

    quilt = QUILT1M(root_dir, csv_file_test, csv_file_train, transform)
    quilt.create_dataset()
    file_path = '/Users/magdaamorim/Desktop/Master Thesis/Git Repos/QUILT_Classification/files/dataset.pkl'
    quilt.save_dataset(file_path)
    
    
    
    
