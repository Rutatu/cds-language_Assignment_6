# cds-language_Assignment_6

***Class assignment for language analytics class at Aarhus University.***

***2021-0***


# Text classification using Deep Learning: TV series Game of Thrones season classification

## About the script


## Methods

1) baseline using a 'classical' ML solution such as CountVectorization + LogisticRegression and use this as a means of evaluating how well your model performs. 

2) DeepLearning model - CNN model. CNNs are useful in text classification tasks, because they can model local structure in text (with their immediate context).
  
  CNNs are prone to overfitting, therefore I applied a weight regularization method to minimize the overfitting. I have used L2 regularization to constrain how the model performs (l2 = L2(0.0001)).
   
 Define embedding size we want to work with
  embedding_dim = 100


  Embeding layer
  model.add(Embedding(vocab_size,                  # vocab size from Tokenizer()
                        embedding_dim,               # embedding input layer size
                        input_length=maxlen,         # maxlen of padded doc      
                        trainable=True))             # trainable embeddings   
                       
   Word eneddings are used to create dense representations of words in a high dimensional space which encodes some kind of linguistic information and is used as an input for CNN model. However, embedding layer is different from word embeddings.

## Repository contents



## Data
preprocessing:
tokenize training and test data
Pad training and test data to maxlen
Transform labels to binarized vectors


## Intructions to run the code



## Results
