# When running in SLURM:
- activate quilt environment and install requirements.txt

-change number labels in model to 18

-QUILT1M_data have the files that need to be in machine before running script.

- Then run the create_dataset.py changing the paths accordingly and with dataset.pkl saved in folder files. It produces dataset.pkl which is a dic with train and test keys.  Each is dic with 'images', 'captions'
#and 'labels', where captions is text, images is a tensor 3x256x256 and labels is a list of pathologies.

- Then run extract_features_py changing the paths accordingly. It creates images and text features and labels for training and test and saves them in folder files.

- Then to run train a specific model use train.py:


--> train1.py produces model1 and mlb1 (labels one hot representation/index): This uses a multilabel classification model and then does the retrievel of the image-text pairs with the same classification! This model is used to perform classification of the input image/text pair into several of the 17 labels in multilabel classification. Then the retrieval is done by extracting classifications of input image/text pair and serach in the database for images/text pairs with the same labels. The idea now is to change the model arquitecture so that we can get a better performance in classification!


- Then run retrieve a specific model using retrieve.py:


---> retrieve1.py  is used to input a test index on test.csv file and then use it to retrieve the most similar cases. It passes the test data point image and text for model trained in train1.py and then classifies it. Then it retrieves data points in train dataset with the same classification (ground truth labels).



