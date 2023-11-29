"""
Idea here is to use deep learning to predict the next word in a sentence.
"""
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
import time
import json
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath('bow_classifier'))
from bow_classifier import GloveVectorizer

from markov import WikiReader

# class bigram_model_dl:
#     def __init__
INPUT_DATA = 'large_files/output/bigramprobab2.json'

def get_data(max_len):
    # Get the data
    with open(INPUT_DATA, 'r') as f:
        raw_data = json.load(f)
    
    # Convert the data into a valid numpy array
    dataX, dataY = [], []
    

    vectorizer = GloveVectorizer()

    for a in raw_data.keys():
        for b in raw_data[a].keys():
            if a not in vectorizer.word2vec.keys() or b not in vectorizer.word2vec.keys():
                continue

            dataX.append(np.concatenate((vectorizer.word2vec[a], vectorizer.word2vec[b])).flatten())
        
            ### Lets add a log for better tunability
            dataY.append(np.log(raw_data[a][b]))

            if max_len != -1 and len(dataX) >= max_len:
                break
    
    return dataX, dataY

def get_training_data(max_len=-1):
    time_start = time.time()
    dataX, dataY = get_data(max_len)
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.2)

    print('TrainX shape: ', len(trainX))
    print('TrainY shape: ', len(trainY))
    print('TestX shape: ', len(testX))
    print('TestY shape: ', len(testY))

    time_end = time.time()
    print('Time taken to load data: ', time_end - time_start)
    print('First element data')
    print(len(trainX[0]), len(testX[0]))
    return np.array(trainX), np.array(testX), np.array(trainY), np.array(testY)


def get_model(input_shape=100):
    model = Sequential()
    model.add(tf.keras.layers.Dense(50, input_dim=input_shape, activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


if __name__ == '__main__':
    trainX, testX, trainY, testY = get_training_data()

    model = get_model(input_shape=len(trainX[0]))
    model.fit(trainX, trainY, epochs=10, verbose=2
    )
    model.save('large_model/output/bigram_model_2.h5')
    model.evaluate(testX, testY, verbose=2)
    print('Done')
    