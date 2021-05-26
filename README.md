# cds-language_Assignment_6

***Class assignment for language analytics class at Aarhus University.***

***2021-04-19**


# Text classification using Deep Learning: TV series Game of Thrones season classification

## About the script

This assignment is Class Assignment 6. The task was to create two scripts for text classification: one for Logistic Regression (LR) Classifier, another for Deep Learning CNN model (DL CNN). Both classifiers try to model the relationship between each season of Game of Thrones and the lines spoken. That is to say - can it predict which season a line comes from? Or to phrase that another way, is dialogue a good predictor of season?

LR model takes TV series Game of Thrones script as an input, trains a Logistic Regression (LR) Classifier, prints the evaluation metrics to the terminal and saves classification report and confusion matrix in a directory. DL CNN model takes the same input, trains a DL CNN classifier, prints the evaluation metrics to the terminal, and saves classification report and a performance graph in a directory. 

## Methods

The problem of the task relates to classifying seasons of TV series Game of Thrones. To address this problem, firstly, I have used a 'classical' machine learning solution such as CountVectorization + LogisticRegression to establish a baseline model performance. Afterwards, I have employed a Deep Learning CNN model using a neural network framework TensorFlow 2.4. CNNs are useful in text classification tasks, because they can model local structure in text (with their immediate context) and are more sophisticated that LR models, therefore suitable for complex cultural data. The CNN´s architecture consists of Embedding layer, Convolutional Layer (CONV) with ReLU activation function,  Max Pooling layer (MAXPOOL) and a Fully Connected Layer (FC). The output layer (OUT) uses softmax activation function and has 8 possible classes. 

Model´s architecture: Embedding -> CONV+ReLU -> MAXPOOL -> FC+ReLU -> OUT(softmax)

CNNs are prone to overfitting, therefore I applied a weight regularization method to CONV and FC layers to minimize the overfitting. I have used L2 regularization to constrain how the model performs (l2 = L2(0.0001)).
  
Depiction of the full model´s architecture can be found in folder called 'out'.
 
   
 
## Repository contents

| File | Description |
| --- | --- |
| out | Folder containing files produced by the scripts |
| out/logReg_confusion_matrix.png | Confusion matrix of LR classifier |
| out/logReg_report.csv | Classification metrics of the LR classifier |
| out/NN_report.csv | Classification metrics of the NN classifier |
| src | Folder containing the scripts |
| src/Logistic_Regression.py | Logistic Regression classifier script |
| src/Neural_Network.py | Neural Network classifier script |
| utils/ | Folder containing utility scripts for the project  |
| utils/classifier_utils.py | utility script used in LR classifier script |
| utils/neuralnetwork.py | utility script used in NN classifier script |
| README.md | Description of the assignment and the instructions |
| create_classification_venv.bash | bash file for creating a virtual environmment |
| kill_classification_venv.bash | bash file for removing a virtual environment |
| requirements.txt | list of python packages required to run the script |



## Data

Data used is a complete set of Game of Thrones script for all seasons in form of a table containing 6 columns with different data types used for various purposes. Description on each columns are provided on the data description part in a link below. For this assignment, columns 'Season' and 'Sentence' were used.


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
$ git clone https://github.com/Rutatu/cds-visual_Assignment_4.git 

#4 Navigate to the newly cloned repo
$ cd cds-visual_Assignment_4

#5 Create virtual environment with its dependencies and activate it
$ bash create_classification_venv.sh
$ source ./classification/bin/activate

``` 

Run the code:

```
#6 Navigate to the directory of the script
$ cd src

#7 Run each code with default parameters
$ python Logistic_Regression.py
$ python Neural_Network.py

#8 Run each code with self-chosen parameters
$ python Logistic_Regression.py -trs 0.9 -tes 0.1 -n lr_cm
$ python Neural_Network.py -trs 0.7 -tes 0.3 -hl1 30 -hl2 15 -hl3 5 -ep 500 -n classification_report

#9 Run the NN script only with hidden_layer_1:
$ python Neural_Network.py -hl1 30 -hl2 0

#10 To remove the newly created virtual environment
$ bash kill_classification_venv.sh

#11 To find out possible optional arguments for both scripts
$ python Logistic_Regression.py --help
$ python Neural_Network.py --help


 ```

I hope it worked!




## Results
