"""
Implementation of Logistic Regression model to predict
the probability p(a|b) where a and b are words in a sentence
and p(a|b) is the probability of word "a" occuring after "b".

We have p(a|b) data.
We will use the logistic regression model to predict the probability and this
data will be used to train the model.
"""

from asyncio import sleep
from markov import WikiReader
import pandas as pd
import numpy as np
import os

import sys
sys.path.append(os.path.abspath('bow_classifier'))
from bow_classifier import GloveVectorizer


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tqdm import tqdm

mode = 'batch'
processed_data_txt = 'large_files/wiki/bigramprobab.txt'
model_weights_file = 'large_files/wiki/logistic_model_weights.h5' 
def convert_to_vector(word, vectorizer):
    if word in vectorizer.word2vec:
        return vectorizer.word2vec[word]
    
    return None
    
def get_bigram_probab_df():
    """Get the data"""
    reader = WikiReader()
    
    """
    Get the bigram probability
    It will be in the format of a matrix
    {{
        a,b -> p(a|b)
    }}
    It is required to process it into a vector
    [
        [][] ==> p(a|b)
    ]
    """
    bigram_prob = reader.bigram_prob
    bigram_prob_dict = {tuple(key): value for key, value in np.ndenumerate(bigram_prob)}

    flat_data_bigram_prob = [(reader.unique_words[key1], reader.unique_words[key2], value) for (key1, key2), value in bigram_prob_dict.items()]

    bigram_prob_df = pd.DataFrame(flat_data_bigram_prob, columns=['a', 'b', 'prob'])
    return bigram_prob_df

def create_data(epoches = 1):
    for i in range(0, epoches):
        
        # Load existing data from the text file (if any)
        try:
            existing_data = pd.read_csv(processed_data_txt, sep='\t')
        except FileNotFoundError:
            existing_data = pd.DataFrame(columns=['a', 'b', 'prob'])  # Create an empty DataFrame if the file doesn't exist

        # Concatenate the existing data with the new DataFrame and drop duplicates in columns 'a' and 'b'
        combined_data = pd.concat([existing_data, get_bigram_probab_df()], ignore_index=True).drop_duplicates(subset=['a', 'b'])

        # Append the unique values to the text file
        combined_data.to_csv(processed_data_txt, sep='\t', index=False, mode='w')
    
    return combined_data

def get_data():
    if not os.path.exists(processed_data_txt):
        print("File does not exist, NEED TO CREATE THE DATA")
        df = create_data()
    else :
        print("File exists")
        df = pd.read_csv(processed_data_txt, sep='\t')

    print(df.head())
    print(df.describe())


    # Create the GloveVectorizer object
    vectorizer = GloveVectorizer()

    df['a'] = df['a'].apply(lambda x: convert_to_vector(x, vectorizer))
    df['b'] = df['b'].apply(lambda x: convert_to_vector(x, vectorizer))

    
    print("Length of the dataframe is: ", len(df))
    
    df.dropna(inplace=True)

    print("Length of the dataframe is: ", len(df))
    return df

def flatten_vector(v):
    if v is not None:
        return np.array(v).flatten()  # Convert vector to a flat array
    return None

def train_random_forest(train_df, test_df):
    # Random Forest model
    rf = RandomForestClassifier()
    
    # Flatten vectors in columns 'a' and 'b'
    train_df['a_flat'] = train_df['a'].apply(flatten_vector)
    train_df['b_flat'] = train_df['b'].apply(flatten_vector)
    
    test_df['a_flat'] = test_df['a'].apply(flatten_vector)
    test_df['b_flat'] = test_df['b'].apply(flatten_vector)
    
    rf.fit(np.stack(train_df['a_flat']), train_df['prob'])  # Pass flattened vectors as input
    rf_accuracy = rf.score(np.stack(test_df['a_flat']), test_df['prob'])
    print(f"Random Forest Accuracy: {rf_accuracy}")

import tensorflow as tf

import tensorflow as tf
import numpy as np

def get_input_shape(train_df):
    # Assuming 'x' column contains the input data
    sample_input = np.array([np.array(x) for x in train_df['x']])
    input_shape = sample_input.shape[1:]  # Extract the shape of a single sample
    
    return input_shape

def train_tensorflow_model(train_df, test_df, batch_size=1000):
    trainY = train_df['prob'].to_numpy()
    testY = test_df['prob'].to_numpy()
    train_df['x'] = train_df.apply(lambda row: np.concatenate((row['a'], row['b'])), axis=1)
    test_df['x'] = test_df.apply(lambda row: np.concatenate((row['a'], row['b'])), axis=1)
    input_shape = get_input_shape(train_df)
    
    num_batches_train = -(-len(train_df) // batch_size)
    num_batches_test = -(-len(test_df) // batch_size)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    if os.path.exists(model_weights_file):
        # Load weights if the model weights file exists
        model.load_weights(model_weights_file)
        print("Model weights loaded successfully!")
    
    print("Number of batches in training data:", num_batches_train)

    for i in tqdm(range(num_batches_train), desc='Training Progress', unit='batch'):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(train_df))
        
        trainX_batch = np.array([np.array(x) for x in train_df['x'][start_idx:end_idx]])
        
        trainY_batch = trainY[start_idx:end_idx]  # Corresponding labels for this batch
        
        assert len(trainX_batch) == len(trainY_batch), "Input and output batch sizes are different!"
        
        model.fit(trainX_batch, trainY_batch, epochs=10, batch_size=batch_size, validation_split=0.1, verbose=0)
    
    # Save the model weights after training
    model.save_weights(model_weights_file)
    print(f"Model weights saved to {model_weights_file}")

    overall_accuracy = 0.0
    for i in range(num_batches_test):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(test_df))
        
        testX_batch = np.array([np.array(x) for x in test_df['x'][start_idx:end_idx]])
        
        testY_batch = testY[start_idx:end_idx]  # Corresponding labels for this batch
        
        assert len(testX_batch) == len(testY_batch), "Input and output batch sizes are different!"
        
        _, accuracy = model.evaluate(testX_batch, testY_batch)
        overall_accuracy += accuracy
    
    overall_accuracy /= num_batches_test
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

if __name__ == '__main__':
    for i in range(1000):
            
        if mode == 'batch' and os.path.exists(processed_data_txt)   :
            os.remove(processed_data_txt)    # Remove the existing data file, create a new data file and train the model
        
        df = get_data()
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # train_random_forest(train_df, test_df)
        train_tensorflow_model(train_df, test_df)

