#!/usr/bin/env python



''' ---------------- About the script ----------------

Assignment 6: Text classification using Deep Learning: TV series Game of Thrones season classification

This script builds and trains a deep learning model using convolutional neural networks which classify TV Series Game of Thrones seasons based on lines spoken. The script prints the evaluation metrics to the terminal, and saves classification report and a performance graph in a directory. 

Preprocessing of the data involves:
- tokenizing training and test data using ```tensorflow.keras.Tokenizer()``` which quickly and efficiently convert text to numbers
- to make the Tokenizer output workable, the documents are padded to be of equal length (maxlen = 100)
- labels transformed to binarized vectors


Arguments:
    
    -dir,    --data_dir:           Directory of the CSV file
    -test,   --test_size:          The size of the test data as a percentage, where the default = 0.2 (20%)
    -optim,  --optimizer:          Method to update the weight parameters to minimize the loss function. Default = Adam
    -ep,     --epochs:             Defines how many times the learning algorithm will work through the entire training dataset. Default = 20




Example:    
    
    with default arguments:
        $ python GoT_deep.py -dir ../data/Game_of_Thrones_Script.csv 
        
    with optional arguments:
        $ python GoT_deep.py -dir ../data/Game_of_Thrones_Script.csv -test 0.3 -optim SGD -ep 50


'''




"""---------------- Importing libraries ----------------
"""


# system tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, numpy, gensim
import pandas as pd
import numpy as np
import gensim.downloader

# import my classifier utility functions - see the Github repo!
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D, Dropout)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2
from tensorflow.keras.utils import plot_model

#from tensorflow.keras.constraints import maxnorm

# matplotlib
import matplotlib.pyplot as plt

# Command-line interface
import argparse



"""---------------- Main script ----------------
"""


def main():
    
    """------ Argparse parameters ------
    """
    # Instantiating the ArgumentParser  object as parser 
    parser = argparse.ArgumentParser(description = "[INFO] Classify Game of Thrones seasons and print out performance accuracy report")
    
    # Adding arguments
    parser.add_argument("-dir", "--data_dir", required = True, help = "Directory of the CSV file")
    parser.add_argument("-test", "--test_size", required=False, default = 0.2, type = float, help = "The size of the test data as a percentage, where the default = 0.2 (20%)")
    parser.add_argument("-optim", "--optimizer", required = False, default = 'adam', help = "Method to update the weight parameters to minimize the loss function. Default = Adam")
    parser.add_argument("-ep", "--epochs", required=False, default = 20, type = int, help = "Defines how many times the learning algorithm will work through the entire training dataset. Default = 20")
    
                                          
    # Parsing the arguments
    args = vars(parser.parse_args())
    
    # Saving parameters as variables
    data = args["data_dir"] # Directory of the CSV file
    test = args["test_size"] # The size of the test data set
    optim = args["optimizer"] # Optimizer
    ep = args["epochs"] # epochs
     
     
    

    """------ Loading data and preprocessing ------
    """

    # Message to a user
    print("\n[INFO] Loading data and preparing for training a Deep Learning model...")
    
    # Create ouput folder, if it doesn??t exist already, for saving the classification report, performance graph and model??s architecture 
    if not os.path.exists("../out"):
        os.makedirs("../out")
    
    # Loading and reading data
    filename = os.path.join(data)
    data = pd.read_csv(filename)
        
    # Extracting sentences and seasons for creating training and test data sets
    sentences = data['Sentence'].values
    season = data['Season'].values

    # train and test split using sklearn
    X_train, X_test, y_train, y_test = train_test_split(sentences, 
                                                        season, 
                                                        test_size=test, 
                                                        random_state=42)

    

    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=5000)
    # Fit to training data
    tokenizer.fit_on_texts(X_train)

    # Tokenized training and test data
    X_train_toks = tokenizer.texts_to_sequences(X_train)
    X_test_toks = tokenizer.texts_to_sequences(X_test)

    # Overall vocabulary size
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    
    # Max length for a doc
    maxlen = 100
    
    
    # Pad training data to maxlen
    X_train_pad = pad_sequences(X_train_toks, 
                                padding='post', # sequences can be padded "pre" or "post"
                                maxlen=maxlen)
    # Pad testing data to maxlen
    X_test_pad = pad_sequences(X_test_toks, 
                               padding='post', 
                               maxlen=maxlen)

    # Transform labels to binarized vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(y_train)
    testY = lb.fit_transform(y_test)


        
    """------ Defining Deep Learning model ------
    """
    
    # Define regularizer
    l2 = L2(0.0001)
    
    # Define embedding size we want to work with
    embedding_dim = 100

    # Define model
    model = Sequential()

    # Embedding -> CONV+ReLU -> MaxPool -> FC+ReLU -> Out
    model.add(Embedding(vocab_size,                  # vocab size from Tokenizer()
                        embedding_dim,               # embedding input layer size
                        input_length=maxlen,         # maxlen of padded doc      
                        trainable=True))             # trainable embeddings                 
                                           
   
    model.add(Conv1D(256, 5, 
                    activation='relu',
                    kernel_regularizer=l2))          # L2 regularization 
    model.add(GlobalMaxPool1D())

    model.add(Dense(128, activation='relu', kernel_regularizer=l2))


    model.add(Dense(8, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', # adding 'sparse_' solved error with incompatible shape using only 'categorical_crossentropy' 
                  optimizer=optim,
                  metrics=['accuracy'])

    # Print summary
    model.summary()
    
    # Ploting and saving model??s architecture
    plot_model(model, to_file='../out/CNN_Model??s_architecture.png',
               show_shapes=True,
               show_dtype=True,
               show_layer_names=True)
    
    # Printing that model??s architecture graph has been saved
    print(f"\n[INFO] Deep learning model??s architecture graph has been saved")
    
    
            
    """------ Training and evaluating Deep Learning model ------
    """
    
    print("[INFO] training and evaluating Deep Learning model ...")
    history = model.fit(X_train_pad, trainY,
                    epochs=ep,
                    verbose=True,
                    validation_data=(X_test_pad, testY))
                 

    # Evaluate 
    loss, accuracy = model.evaluate(X_train_pad, trainY, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test_pad, testY, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    # Plot
    plot_history(history, epochs = ep) 

    # Printing that performance graph has been saved
    print(f"\n[INFO] Deep Learning model??s performance graph has been saved")
    
    # labels
    labels = ["Season_1", "Season_2", "Season_3", "Season_4", "Season_5", "Season_6", "Season_7", "Season_8"]
    
    # Classification report
    predictions = model.predict(X_test_pad, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labels))
    
    
    
    # defining full filepath to save .csv file 
    outfile = os.path.join("../", "out", "GoT_CNN_classifier_report.csv")
    
    # turning report into dataframe and saving as .csv
    report = pd.DataFrame(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labels, output_dict = True)).transpose()
    report.to_csv(outfile)
    print(f"\n[INFO] Classification report has been saved")

    
    print("\nScript was executed successfully! Have a nice day")
        
    

"""---------------- Functions ----------------
"""

# this function was developed for use in class and has been adapted for this project

def plot_history(H, epochs):
    """
    Utility function for plotting model history using matplotlib
    
    H: model history 
    epochs: number of epochs for which the model was trained
    """
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig('../out/GoT_CNN_classification_performance_graph.png')
    
    
    
         
# Define behaviour when called from command line
if __name__=="__main__":
    main()

