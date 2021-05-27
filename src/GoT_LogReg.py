#!/usr/bin/env python


''' ---------------- About the script ----------------

Assignment 6: Text classification using Deep Learning: TV series Game of Thrones season classification

This scripts trains a Logistic regression model which tries to predict which season of Game of Thrones a spoken line comes from.  
It tries to answer a question 'is dialogue a good predictor of season?' Script prints out the evaluation of how well the model performs to the terminal, saves a classification report as CSV file and confusion matrix as PNG file in a created folder 'out'.

Used methods: CountVectorization + LogisticRegression.



Arguments:
    -dir,      --data_dir:             Directory of the .csv file
    -test,     --test_size:            The size of the testing data as a percentage. Default = 0.25 (25%)

Example:
    with default arguments
        $ python GoT_LogReg.py -dir ../data/Game_of_Thrones_Script.csv
    
    with all optional arguments:
        $ python GoT_LogReg.py -dir ../data/Game_of_Thrones_Script.csv -test 0.2
    


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
from sklearn.preprocessing import LabelEncoder

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dropout
from tensorflow.keras import constraints
from tensorflow.keras.constraints import max_norm

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
    parser = argparse.ArgumentParser()
    
    # Adding optional arguments                                                  
    parser.add_argument("-dir", "--data_dir", required = True, help = "Directory of the CSV file")
    parser.add_argument("-test", "--test_size", required = False, default = 0.25, type = float, help = "The size of the testing data as a percentage. Default = 0.25 (25%)")
    
                                     
    # Parsing the arguments
    args = vars(parser.parse_args())
                                     
    # Saving parameters as variables
    data = args["data_dir"] # Directory of the CSV file
    test = args["test_size"] # test data set size

    
    
    """------ Loading and preprocessing data ------
    """
    
    # Message to a user
    print("\n[INFO] Loading data and preparing for a logistic regression model...")
    
    # Loading data
    filename = os.path.join(data)
    data = pd.read_csv(filename)
        
    # Extracting sentences and seasons for creating training and test data
    sentences = data['Sentence'].values
    season = data['Season'].values

    # train and test split using sklearn
    X_train, X_test, y_train, y_test = train_test_split(sentences, 
                                                        season, 
                                                        test_size=test, 
                                                        random_state=42)
    
    # Initializing count vectorizer
    vectorizer = CountVectorizer()
    
    # 'Fitting' vectorizer and the data
    # First for the training data...
    X_train_feats = vectorizer.fit_transform(X_train)
    #... then for the test data
    X_test_feats = vectorizer.transform(X_test) 
    
 

    
    """------ Logistic regression classifier: training and evaluation ------
    """
    
    # Message to a user
    print("\n[INFO] Training and evaluating a logistic regression model...")
    
    # Definining LR model and fitting it to the data
    classifier = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_feats, y_train)
    # Predicting on the test data
    y_pred = classifier.predict(X_test_feats)
    
    # Evaluation
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    # Printing measures to the terminal
    print(classifier_metrics)
    
     
        
    
    """------ Saving classification report and confusion matrix ------
    """
    
    # Creating a folder for the outputs if it doen´t already exist
    if not os.path.exists("../out"):
            os.makedirs("../out")
            
        
    ### Saving evaluation as .csv file
    # Turning classifier report into a dataframe
    report_df = pd.DataFrame(metrics.classification_report(y_test, y_pred, output_dict = True)).transpose()    
    # Defining full filepath to save .csv file 
    path_csv = os.path.join("..", "out", "GoT_LR_classification_report.csv")
    # Saving a dataframe as .csv
    report_df.to_csv(path_csv) 
    # Printing that .csv file has been saved
    print(f"\n[INFO] classifier report is saved in directory {path_csv}")
    
    ### Saving confusion matrix as .png file
    # Defining full filepath to save .png file
    path_png = os.path.join("..", "out", "GoT_LR_confusion_matrix.png")
    # Creating confusion matrix
    clf.plot_cm(y_test, y_pred, normalized=True)
    # Saving as .png file
    plt.savefig(path_png)
    # Printing that .png file has been saved
    print(f"\n[INFO] confusion matrix is saved in directory {path_png}")
    
    # Message to a user
    print("\nScript was executed successfully! Have a nice day")

    
         
# Define behaviour when called from command line
if __name__=="__main__":
    main()

    
    
    
