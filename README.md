# AI Generating -- Bollywood Song Lyrics
In this project, I have tried to make a Bollywood Song Generator using a simple One Hot Encoder Word Embedding Method. In my further journey, I will try to improve the Word Embedding i.e, Data Pre Processing part to Word2Vec and in further to Glove implementation. This is my Study Project for understanding the concepts of NLP and the use of text in deep learning models. The main aim of the project was to understand different Word Embedding Methods and how it affects the accuracy of the model.

![](https://github.com/ADItyaP999/Bollywood-Song-Generator/blob/master/images/waysofdoing.png)

## Dependencies
All of these can be downloaded in a single command, see below.

-   Pandas
-   Tensorflow (for Keras)
-   Numpy

 `pip install -r requirements.txt`

## About the dataset

The Dataset contains 1291 Lyrics of Bollywood Songs. For generating Corpus the dataset is too. Why not just give a try ??

## Pre Trained Model
Download the repository and run below commands to generate lyrics.

    pip install -r requirements.txt
    python pretrained-onehot.py
For changing the beginning phrases you need to edit the pretrained file..

## TODO

 - [x] One Hot Encoding for Word Embedding
 - [ ] Improve the LSTM Model
 - [ ] Try using Word2Vec 
 - [ ] Final goal implement using GloVe

## Thanking for Dataset
Thanks to [AdityaKetkar](https://www.kaggle.com/adityktkr/bollywood-lyrics-labelled) for helping in providing the dataset.
