# age-prediction-project


Age recognition project 
The aim of the project is to develop an age recognition model that would take in a photo of a person and output an estimate age. The data are taken from https://www.kaggle.com/mariafrenti/age-prediction.  The initial data is around 2GB of memory consisting of 100 folders with 232647 photos of different people in total. Each folder corresponds to an age from 1 to 100 and each photo has 128x128 resolution. The data is a little bit corrupted. For example, some photos are in wrong folders like one may find an old person in a folder with teenagers or some photos might be cartoon characters. I cleaned some by hands. But there is still a small percentage of that kind of errors. Also, age labels are not totally accurate, many older people might be in 20 age younger sections and vice versa. Here is a plot of photo numbers against ages.
The data is split into training data with 232647 photos and testing data with 47486 photos which is around 80% and 20% of the entire data.

Face encodings. 

Using a python package face_recognition we go over all photos to detect human faces and generate face encodings for each detected face representing it as a set of measurements. Each face encoding is a numeric array in R^128 which is stored as an npy extension file. Each npy file takes 2KB of memory which almost 3 times less than picture memories. Around 10% of the pictures is elected at this stage by several causes: a) picture has several faces b) picture is a cartoon c) extra attributes like sunglasses or masks d) code can’t detect a face due to image imperfections or other reasons. 
After data is cleaned and converted to face encodings, plan is to implement PCA to reduce number of attributes from 128 to lower it, probably to 5, 10 , 15 depending on estimate of hard threshold.  Then I am going to try different models (linear, logistic regressions, K-NN, etc) to train and test the data which will take in principal components and output estimate of a person.
