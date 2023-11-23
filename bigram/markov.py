"""
Language model: A model of the probability of a sequence of words.
Mode--> Assumption...
    "Map is not the territory"

    Bigram "Combination of two words"
    Trigram "Combination of three words"
    .
    .
    .
    N-gram "Combination of n words"

    p(wt | wt-1) --> Probability of word "wt" given the previous word "wt-1"
    p(wt | wt-1) = count(wt-1 --> wt) / count(wt-1)


    Set of documents
        -> Set of sentences

    p(A->B->C) = p(C|A,B) * p(B|A) * p(A)
    
    p(A) = count(A) / count(all words)
    p(C|A,B) = count(A,B,C) / count(A,B)



    Add-one smoothing:: (Never 0 probability)
    psmooth (B|A) = count(A,B) + 1 / count(A) + V(volcabulary size)

    Markov Assumption:
        p(wt| wt-1, wt-2, ... w0) = p(wt | wt-1)
        i.e.
        p(E|A,B,C,D) = p(E|D)
            (P(E|D)p(D|C)p(C|B)p(B|A)p(A))

    1. Need to use the log probabilities instead of the actual probabilities
        a. since with the size, computer will not be able to handle the actual probabilities due to the values approacting to 0.
    2. Normalize the log probabilites to keep a fair comparision between shorter and longer sentences.

"""


"""
@todo:
    1. Load the data of texts
    2. Create a bigram probability p(B|A)
    3. Test this probability
        a. Test the probability of a sentence
        b. Test the probability of a random sentence
"""
import random
import numpy as np
import re
import os
import random
import json
import argparse
import sys

sys.path.append(os.path.abspath('utils'))
from utils.bigram_utils import get_sentences_for_testing, plot_graph

MAX_VOCAB = 20000000
BIGRAM_OUTPUT = 'output/bigramprobab.json'

class WikiReader:
    def process_sentence(self, sentence):
        # Remove special characters, make lowercase
        sentence = sentence.expandtabs()
        words = [re.sub(r'[^a-zA-Z]', '', word).lower() for word in sentence.split()]
        

        # Remove numbers, blank words, and words with a single character
        filtered_words = [word for word in words if word and not word.isdigit() and len(word) > 1]
            
        return filtered_words

    def process_file(self, file_path, max_vocab=MAX_VOCAB):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Split the text into sentences based on "."
        sentences = text.split('.')
        random.shuffle(sentences)

        vocab = set()
        processed_list = []
        # Process each sentence and create a list of strings
        for sentence in sentences:
            word_list = self.process_sentence(sentence)
            if len(word_list) < 3:
                continue # Ignore short sentences
        
            processed_list.append(word_list)
            for words in word_list:
                vocab.add(words)
            if len(vocab) > max_vocab:
                break   # We have enough words

        return vocab, processed_list
        
    def process_directory(self, directory_path, max_vocab=MAX_VOCAB):
        result_list = []
        unique_words = set()
        dir_list = os.listdir(directory_path)
        random.shuffle(dir_list)
        # Iterate through all files in the directory
        for filename in dir_list:
            file_path = os.path.join(directory_path, filename)

            # Check if it's a file (not a subdirectory)
            if os.path.isfile(file_path):
                words, new_result_list = self.process_file(file_path, max_vocab)
                unique_words = unique_words.union(words)
                result_list.extend(new_result_list)
                if len(unique_words) > max_vocab:
                    break # We have enough words
                print("Size of the vocab is", len(unique_words))
        return list(unique_words), result_list

    def word_to_idx(self, unique_words):
        """
        Returns a dictionary of word to index
        """
        word2idx = {}
        for idx, word in enumerate(unique_words):
            word2idx[word] = idx
        return word2idx

    def bigram_prob(self, sentences, unique_words, word2idx):
        """
        Returns the bigram probability of the sentences
        """
        count_of_words = {}
        bigram_prob = {}
        V = len(unique_words) # Vocabulary size
        for sentence in sentences:
            for i in range(len(sentence)):
                if i == len(sentence) - 1:
                    continue # Do nothing p(a|b,a) 
                else:
                    if sentence[i] in word2idx and sentence[i+1] in word2idx:
                        if not sentence[i] in bigram_prob.keys():
                            bigram_prob[sentence[i]] = {}
                        if not sentence[i+1] in bigram_prob[sentence[i]]:
                            bigram_prob[sentence[i]][sentence[i+1]] = 0
                        if not sentence[i] in count_of_words.keys():
                            count_of_words[sentence[i]] = 0

                        count_of_words[sentence[i]] += 1
                        bigram_prob[sentence[i]][sentence[i+1]] += 1
        
        for a in bigram_prob:
            for b in bigram_prob[a]:
                bigram_prob[a][b] = (bigram_prob[a][b] + 1)/ (V + count_of_words[a])
        return bigram_prob, V
    
    def __init__(self, data_path=None, max_vocab=MAX_VOCAB):
        if data_path is None:
                
            """
            For all the files in the wiki folder, create a list of sentences
            Append all the sentences in the single wiki_list
            """
            self.unique_words, self.sentence_list  = self.process_directory('large_files/wiki')[:max_vocab] # List of sentences              
            self.word2idx = self.word_to_idx(self.unique_words)
            self.bigram_prob, self.V = self.bigram_prob(self.sentence_list, self.unique_words, self.word2idx)

            ### Save the data
            with open(BIGRAM_OUTPUT, "w") as outfile: 
                json.dump(reader.bigram_prob, outfile)
        else:
            with open(data_path, 'r') as file:
                self.bigram_prob = json.load(file)
            self.unique_words = list(self.bigram_prob.keys())
            self.V = len(self.unique_words)

    def sentence_prob(self, sentence):
        """
        Returns the probability of the sentence
            Use the log probabilities normalized by the lenght of the sentence
        """
        if len(sentence) == 0:
            return 0
        log_prob = 0
        for i in range(len(sentence)):
            if i == len(sentence) - 1:
                continue # Do nothing p(a|b,a) 
            else:
                if sentence[i] not in self.bigram_prob.keys() or (sentence[i+1] not in self.bigram_prob[sentence[i]].keys()) :
                    log_prob += np.log(1 / self.V)   # Add-one smoothing
                else:
                    log_prob += np.log(self.bigram_prob[sentence[i]][sentence[i+1]])
        return log_prob / (len(sentence) - 1)


def validate_unique_words(unique_words):
    """
    Validate the unique words
    """
    empty_words = 0
    digit_words = 0
    single_char_words = 0
    words_with_spaces = 0

    for word in unique_words:
        if len(word) == 0:
            empty_words += 1
        if word.isdigit():
            digit_words += 1
        if len(word) == 1:
            single_char_words += 1
        if ' ' in word:
            words_with_spaces += 1
        if '\t' in word:
            words_with_spaces += 1
    
    print("Empty words: ", empty_words)
    print("Digit words: ", digit_words)
    print("Single char words: ", single_char_words)
    print("Words with spaces: ", words_with_spaces)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--new_data', type=bool, default=0, help='New data to be created')
    parser.add_argument('-t', '--test_model', type=bool, default=0, help='Test the model on random and real sentences')

    args = parser.parse_args()
    new_data = args.new_data
        
    if new_data or not os.path.exists(BIGRAM_OUTPUT):
        print("File does not exist, NEED TO CREATE THE DATA")

        reader = WikiReader()
        validate_unique_words(reader.unique_words)

        for i in range(0, 10):
            real_sentence = random.choice(reader.sentence_list)
            random_sentence = np.random.choice(reader.unique_words, size=len(real_sentence))
            
            print("Random sentence: ", random_sentence)
            print("Real sentence: ", real_sentence)
            print("Probability of random sentence: ", reader.sentence_prob(random_sentence))
            print("Probability of real sentence: ", reader.sentence_prob(real_sentence))  
    else:
        reader = WikiReader(BIGRAM_OUTPUT)
        validate_unique_words(reader.unique_words)

    if args.test_model:
        real_sentences , fake_sentences = get_sentences_for_testing()
        real_sentences_probability = []
        fake_sentences_probability = []
        for sentence in real_sentences:
            real_sentences_probability.append(reader.sentence_prob(sentence))
        for sentence in fake_sentences:
            fake_sentences_probability.append(reader.sentence_prob(sentence))
        
        print("Mean probability of real sentences: ", np.mean(real_sentences_probability))
        print("Mean probability of fake sentences: ", np.mean(fake_sentences_probability))

        plot_graph(real_sentences_probability, fake_sentences_probability)

    while True:
        sentence = input("Give me a sentence: ")
        if len(sentence) == 0:
            print("Sentence can't be empty!")
            continue
        if sentence == 'q':
            break
        else:
            sentence = reader.process_sentence(sentence)
            print("Probability of sentence: ", reader.sentence_prob(sentence))
