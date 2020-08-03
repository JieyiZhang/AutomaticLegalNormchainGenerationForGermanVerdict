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

MLP model with numeric vector representation of legal norms as input

* Numeric vector representation of norms as input for each document
* Entries are the frequencies of the norms
* Train different classification models and fine-tuned the best performer

MLP model with numeric vector representation of legal norms and their positions as input

* Numeric vector representation of norms as input for each document
* Each index of the vector represent the norm and its position
* Entries are the frequencies of the norms in a specific section
* Train different classification models and fine-tuned the best performer

## Text classification with DL model

## Text Summarization with BERT-Transformer model
