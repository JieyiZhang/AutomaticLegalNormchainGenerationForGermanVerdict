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
