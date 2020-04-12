# rock-paper-scissor
In this project we will train a deep convolutional network to identify 3 hand symbols i.e. rock, paper and scissor. 
There are 3 directories namely train which holds the database for training, test which holds the database for testing and validation which has random images of all the 3 gestures for the model to make predictions.
The "train.ipynb" file has all the code required to preprocess and train our data. Just be sure to change the directories where your datasets are downloaded. 
The "predict_on_validation" file will make the model predict on the images in our validation folder. 
The "predict_on_video_feed" file will make the model predict what gesture is present in each frame of the webcam video in the Region of Interest.
