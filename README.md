# Automatic Legal Normchain Generation For German Legal Verdicts
Master thesis code

Seperate the labelled and unlabelled files using the code SeperateFiles.py.

## Rule-based approach

The rule-based algorithm follows the following steps:

1. Extract legal norms from the raw ruling text
2. Assigning scores to each norm based on their frequencies and positions 
3. Pick the candidate norms with scores above a specific threshold

The functions are implemented in RulebasedApproach.py and the run code RuleBasedResults.py to generate the results.

## Text classification with MLP model

1. MLP model with numeric vector representation of legal norms as input

* Numeric vector representation of norms as input for each document
* Entries are the frequencies of the norms
* Train different classification models and fine-tuned the best performer

The code is implemented in MLP.py.

2. MLP model with numeric vector representation of legal norms and their positions as input

* Numeric vector representation of norms as input for each document
* Each index of the vector represent the norm and its position
* Entries are the frequencies of the norms in a specific section
* Train different classification models and fine-tuned the best performer

The code is implemented in MLP_with_position.py.

## Text classification with DL model

Text classification with BIGRU-ATT and BIGRU-LWAN neural network models. Code for this two models are implemented in Glove_BiGRUAtt.py, BiGRU_LWAN.py, LabelwiseAttention.py. The process includes:

1. Preprocess input text
2. Embed input text with GloVe as the input data
3. Train the neural networks for large-scale text classification problem

## Text Summarization with BERT-Transformer model
The pretrained BERT-Transformer model is saved at https://drive.google.com/drive/folders/1hWVE7GfsTpk7g2P6KoEklvfofuGQclwV?usp=sharing. Building and training the model include the following steps: 

1. Preprocess input text

   1.1 Clean the text data
   
   1.2 Select fact of case and reasoning sections as input text
   
   1.3 For applying BERT encoder, truncate and post-pad the input text to the fixed length of 512
   
2. Encoding part of the model is the pre-trained BERT encoder for German language 
3. The hidden state vectors are then fed into the Transformer decoder part
4. Train the whole model to predict the norm chains as summarization
