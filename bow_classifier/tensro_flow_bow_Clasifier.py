import pandas as pd
import cupy as cp
import numpy as np
from scipy import spatial
from sklearn.metrics import accuracy_score

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

train_df = pd.read_csv('large_files/deep_nlp1/r8-train-all-terms.txt', header=None, sep='\t')
test_df = pd.read_csv('large_files/deep_nlp1/r8-test-all-terms.txt', header=None, sep='\t')
train_df.columns = ['label', 'content']
test_df.columns = ['label', 'content']

class GloveVectorizer:
    def __init__(self):
        word2vec = {}   # Dict for converting words to vectors
        embedding = []  # List of vectors (values of the word2vec dict)
        idx2word = []   # List of words (keys of the word2vec dict)
        
        with open('large_files/glove.6B/glove.6B.300d.txt','r', encoding='utf-8') as glove_file:
            for line in glove_file:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec
                embedding.append(vec)
                idx2word.append(word)
        
        self.word2vec = word2vec
        self.embedding = np.asarray(embedding)
        self.word2idx = {v:k for k,v in enumerate(idx2word)}
        self.V, self.D = self.embedding.shape
        

        ###
        self.idx2word = idx2word
        self.spicy_tree = spatial.KDTree(self.embedding)

    def transform(self, data):
        ### Data is basically the dataframe....
        X = np.zeros((len(data), self.D))
        n = 0
        empty_cnt = 0
        for sentence in data:
            valid_word_cnt = 0
            vec = []
            #For every sentense in the data we will have to convert it into a vector
            for word in sentence:
                if word in self.word2vec:
                    vec.append(self.word2vec[word])
                    valid_word_cnt += 1

            
            if valid_word_cnt > 0:
                vec = np.asarray(vec)
                X[n] = vec.mean(axis=0)
            else:
                empty_cnt += 1
            
            n += 1
        print("Empty count is %d" % empty_cnt)
        ##### Now we have the vector representation of the data
        return X

    def find_closest(self, vector):
        return self.idx2word[self.spicy_tree.query(vector)]
        
            
    def inverse_transform(self, X):
        ### This is to get the original word back from the word vector "X"
        wordVector = []
        for vector in X:
            wordVector.append(self.find_closest(np.list(vector)))
        return wordVector

    
vectorizer = GloveVectorizer()
Xtest = vectorizer.transform(test_df.content)
Ytest = test_df.label


Xtrain = vectorizer.transform(train_df.content)
Ytrain = train_df.label

#### Try the same with tensorflow
import tensorflow as tf
from sklearn.metrics import accuracy_score

y_vectorizer = GloveVectorizer()
X_train = Xtrain
y_train = y_vectorizer.transform(Ytrain)
X_test = Xtest
y_test = y_vectorizer.transform(Ytest)

# Print the list of available GPUs
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", physical_devices)

# Build a simple neural network model using Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model on the test set
y_pred = model.predict(X_test)


def custom_accuracy_score(y_true, y_pred, threshold=0.5):
    """
    Compute custom accuracy score based on the absolute distance between vectors.

    Parameters:
    - y_true: Array of true labels (vectors)
    - y_pred: Array of predicted labels (vectors)
    - threshold: Threshold for considering a prediction as correct (default: 0.5)

    Returns:
    - accuracy: Custom accuracy score
    """
    distances = np.linalg.norm(y_true - y_pred, axis=1)  # Calculate L2 norm (Euclidean distance)
    correct_predictions = np.sum(distances)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy


accuracy = custom_accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy)

def real_accuracy(y_true, y_pred):
    """
    Compute real accuracy score based on the absolute distance between vectors.

    Parameters:
    - y_true: Array of true labels (vectors)
    - y_pred: Array of predicted labels (vectors)

    Returns:
    - accuracy: Real accuracy score
    """
    y_true = vectorizer.inverse_transform(y_true)
    y_pred = vectorizer.inverse_transform(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

real_accuracy = real_accuracy(y_test, y_pred)
print("Real accuracy on the test set:", real_accuracy)