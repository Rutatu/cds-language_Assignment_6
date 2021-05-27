# cds-language_Assignment_6

***Class assignment for language analytics class at Aarhus University.***

***2021-04-19***


# Text classification using Deep Learning: TV series Game of Thrones season classification

## About the script

This assignment is Class Assignment 6. The task was to create two scripts for text classification: one for Logistic Regression (LR) Classifier, another for Deep Learning Convolutional Neural Network model (DL CNN). Both classifiers try to model the relationship between each season of Game of Thrones and the lines spoken. That is to say - can it predict which season a line comes from? Or to phrase that another way, is dialogue a good predictor of season?

LR model takes TV series Game of Thrones script as an input, trains a Logistic Regression (LR) Classifier, prints the evaluation metrics to the terminal and saves classification report and confusion matrix in a directory. DL CNN model takes the same input, trains a DL CNN classifier, prints the evaluation metrics to the terminal, and saves classification report and a performance graph in a directory. 

## Methods

The problem of the task relates to classifying seasons of TV series Game of Thrones. To address this problem, firstly, I have used a 'classical' machine learning solution such as CountVectorization + LogisticRegression to establish a baseline model performance. Afterwards, I have employed a Deep Learning CNN model using a neural network framework TensorFlow 2.4. CNNs are useful in text classification tasks, because they can model local structure in text (with their immediate context) and are more sophisticated that LR models, therefore suitable for complex cultural data. The CNN´s architecture consists of Embedding layer, Convolutional Layer (CONV) with ReLU activation function,  Global Max Pooling layer (GlobalMAXPOOL) and a Fully Connected Layer (FC). The output layer (OUT) uses softmax activation function and has 8 possible classes. 

Model´s architecture: Embedding -> CONV+ReLU -> GlobalMAXPOOL -> FC+ReLU -> OUT(softmax)

CNNs are prone to overfitting, therefore I applied a weight regularization method to CONV and FC layers to minimize the overfitting. I have used L2 regularization to constrain how the model performs (l2 = L2(0.0001)).
  
Depiction of the full model´s architecture can be found in folder called 'out'.
 
   
 
## Repository contents

| File | Description |
| --- | --- |
| data/ | Folder containing files input data for the script |
| data/Game_of_Thrones_Script.csv| CSV file used as input for the script |
| out/ | Folder containing files produced by the scripts |
| out/CNN_Model´s_architecture.png | CNN model´s architecture |
| out/GoT_CNN_classification_performance_graph.png | Performance graph of CNN classifier |
| out/GoT_CNN_classifier_report.csv| Classification metrics of the CNN classifier |
| out/GoT_LR_classification_report.csv | Classification metrics of the LR classifier |
| out/GoT_LR_confusion_matrix.png | Confusion matrix of LR classifier |
| src/ | Folder containing the scripts |
| src/GoT_deep.py | CNN classifier script |
| src/GoT_LogReg.py | Logistic Regression classifier script |
| utils/ | Folder containing utility script for the project  |
| utils/classifier_utils.py | utility script used in LR classifier script |
| LICENSE | A software license defining what other users can and can't do with the source code |
| README.md | Description of the assignment and the instructions |
| create_GoT_venv.bash | bash file for creating a virtual environmment |
| kill_GoT_venv.bash | bash file for removing a virtual environment |
| requirements.txt | list of python packages required to run the script |



## Data

Data used is a complete set of Game of Thrones script for all seasons (8 seasons in total, 10 episodes first 6 seasons, 7 and 6 episodes for the last two seasons) in form of a table containing 6 columns with different data types used for various purposes. Description of each column is provided in the data description part in a link below. For this assignment, columns 'Season' and 'Sentence' were used.


Link to data: https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons


___Data preprocessing___

The preprocessing of data for LR model included the following step:
- vecorizing training and test data using ```sklearn```  ``` CountVectorizer()```, which transformed text into a vector on the basis of the frequency (count) of each word that occurs in the entire text.

The preprocessing of data for DL CNN model included the following steps:
- tokenizing training and test data using ```tensorflow.keras.Tokenizer()``` which quickly and efficiently convert text to numbers
- to make the Tokenizer output workable, the documents are padded to be of equal length (maxlen = 100)
- labels transformed to binarized vectors




## Intructions to run the code

Both codes were tested on an HP computer with Windows 10 operating system. They were executed on Jupyter worker02.

__Codes parameters__


```Logistic Regression classifier```       

| Parameter | Description |                                              
| --- | --- |                                                                    
| data_dir (dir) | Directory of the CSV input file |                                       
| test_size (tes) | The size of the testing data as a percentage. Default = 0.25 (25%) | 
                                     

```DL CNN classifier```
 
| Parameter | Description |                                              
| --- | --- |                                                                    
| data_dir (dir) | Directory of the CSV input file |                                       
| test_size (tes) | The size of the testing data as a percentage. Default = 0.2 (20%) | 
| optimizer (optim) | Method to update the weight parameters to minimize the loss function. Default = Adam |
| epochs (ep) | Defines how many times the learning algorithm will work through the entire training dataset. Default = 20 |                               



__Steps__

Set-up:
```
#1 Open terminal on worker02 or locally
#2 Navigate to the environment where you want to clone this repository
#3 Clone the repository
$ git clone https://github.com/Rutatu/cds-language_Assignment_6.git 

#4 Navigate to the newly cloned repo
$ cd cds-language_Assignment_6

#5 Create virtual environment with its dependencies and activate it
$ bash create_GoT_venv.sh
$ source ./GoT/bin/activate

``` 

Run the code:

```
#6 Navigate to the directory of the scripts
$ cd src

#7 Run each code with default parameters
$ python GoT_LogReg.py -dir ../data/Game_of_Thrones_Script.csv
$ python GoT_deep.py -dir ../data/Game_of_Thrones_Script.csv 

#8 Run each code with self-chosen parameters
$ python GoT_LogReg.py -dir ../data/Game_of_Thrones_Script.csv -test 0.2
$ python GoT_deep.py -dir ../data/Game_of_Thrones_Script.csv -test 0.3 -optim SGD -ep 50

#9 To remove the newly created virtual environment
$ bash kill_GoT_venv.sh

#11 To find out possible optional arguments for both scripts
$ python GoT_LogReg.py --help
$ python GoT_deep.py --help


 ```

I hope it worked!


## Results

LR classifier achieved a weighted average accuracy of 26% for correctly classifying TV series Game of Thrones seasons. DL CNN classifier achieved a weighted average accuracy of 24%, which is slightly worse than LR classifier. Such results can indicate that it was a very challenging task to classify TV series seasons according to lines spoken, and as simple approach as LR can perform better. This might have happened for various reasons: not enough data (only 10 episodes per season, 8 seasons in total), seasons of TV series Game of Thrones might be specifically hard to classify due to the nature of conversations, shallow or not relevant data preprocessing steps.

GoT_CNN_classification_performance_graph suggest that overfitting might have been a problem - validation loss curve was increasing during the whole training which created a huge gap with training loss curve, while training accuracy reached an accuracy close to 100%. We cannot yet conclude that scripts of TV series are not suitable for text classification tasks, such as classifying seasons. More experimentation with different datasets and hyperparameters is needed.



