from keras.preprocessing.text import Tokenizer 
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np 
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Layer, Input, Embedding, Dropout, SpatialDropout1D, GlobalAveragePooling1D, Dense, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from sklearn.metrics import classification_report


from keras_self_attention import SeqSelfAttention
import keras.backend as K
from keras.callbacks import Callback
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Model
from keras.layers import Layer, Input, Embedding, Dropout, SpatialDropout1D, GlobalAveragePooling1D
from keras.layers import GlobalMaxPooling1D, Bidirectional, GRU, CuDNNGRU, Activation, Dense
from keras.layers import Dot, Reshape, TimeDistributed, concatenate, BatchNormalization
from keras import initializers, regularizers, constraints
from keras.optimizers import Adam

def compute_precision_score(label_array, generate_array):

#     label_array = np.array(label_tuple)
#     generate_array = np.array(generate_tuple)

    matches = 0
    total = 0

    for label in label_array:
        for pred in generate_array:
            if pred == label:
                matches = matches + 1
        total = total + 1

    if total>0:
        score = matches/total
    else:
        score = -1.0

    return score

def compute_recall_score(label_array, generate_array):

#     label_array = np.array(label_tuple)
#     generate_array = np.array(generate_tuple)

    matches = 0
    total = 0

    for pred in generate_array:
        for label in label_array:
            if pred == label:
                matches = matches + 1
        total = total + 1

    if total>0:
        score = matches/total
    else:
        score = 0.0

    return score

print('Loading data...')
data = pd.read_pickle('input_shorten_AbbrArt.pkl')

docs = data.text.tolist()

print('MultiLabelBinarizer...')
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(data.label_abbrs)

# transform target variable
labels = multilabel_binarizer.transform(data.label_abbrs)

print('Tokenizing...')
tokenize = Tokenizer(num_words = 1000000, oov_token='UNK')
tokenize.fit_on_texts(docs)

tokenize.word_index = {e:i for e,i in tokenize.word_index.items() if i <= 1000000} # <= because tokenizer is 1 indexed
tokenize.word_index[tokenize.oov_token] = 1000000 + 1

print('Train-test split...')
xtrain, xval, ytrain, yval = train_test_split(docs, labels, test_size=0.2, random_state=9)

xtrain = tokenize.texts_to_sequences(xtrain)
xval = tokenize.texts_to_sequences(xval)

print('Padding sequences...')
max_len = 1024
# pad documents to a max length
xtrain = pad_sequences(xtrain, maxlen=max_len, padding='post', truncating = 'post')
xval = pad_sequences(xval, maxlen=max_len, padding='post', truncating = 'post')

print('Generating embedding matrix...')
# load the whole embedding into memory
embeddings_index = dict()
f = open('vectors.txt')
for line in f:
   values = line.split()
   word = values[0]
   coefs = np.asarray(values[1:], dtype='float32')
   embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))

vocab_size = 1000000 + 2

print(vocab_size)

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenize.word_index.items():
   embedding_vector = embeddings_index.get(word)
   if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)

print('Build network...')
# Define input tensor
inp = Input(shape=(xtrain.shape[1],), dtype='int32')

# Word embedding layer
embedded_inputs = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                            weights=[embedding_matrix], trainable = False)(inp)

# Apply dropout to prevent overfitting
embedded_inputs = SpatialDropout1D(0.2)(embedded_inputs)

# Apply Bidirectional GRU over embedded inputs
rnn_outs = Bidirectional(GRU(32, return_sequences=True))(embedded_inputs)
rnn_outs = Dropout(0.2)(rnn_outs) # Apply dropout to GRU outputs to prevent overfitting

# Attention Mechanism - Generate attention vectors
scores = SeqSelfAttention(return_attention=False, attention_activation = 'exponential')(rnn_outs)

# Dense layers
#fc = Dense(14892)(sentence)
#fc = Dropout(0.5)(fc)
fc = Flatten()(scores)
output = Dense(10119, activation='sigmoid')(fc)

# Finally building model
model = Model(inputs=inp, outputs=output)
model.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer='adam')

# Print model summary
model.summary()

print('Start fitting model...')
# fit the model
model.fit(xtrain, ytrain, epochs=20, verbose=1)
# evaluate the model
print('Evaluating')
loss, accuracy = model.evaluate(xval, yval, verbose=0)
print('Accuracy: %f' % (accuracy*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

print('Predicting...')
# predict probabilities for test set
yhat_probs = model.predict(xval, verbose=0)
yhat_probs[yhat_probs>0.5]=1
yhat_probs[yhat_probs<=0.5]=0
# predict crisp classes for test set
#yhat_classes = model.predict_classes(xval, verbose=0)

print('Inverse transforming...')

y_pred_label = multilabel_binarizer.inverse_transform(yhat_probs)

y_val_label = multilabel_binarizer.inverse_transform(yval)

print('Computing scores...')

diction = {'y_val_label': y_val_label, 'y_pred_label': y_pred_label}
assess_df = pd.DataFrame(diction)

assess_df.y_val_label = assess_df.y_val_label.apply(np.array)

assess_df.y_pred_label = assess_df.y_pred_label.apply(np.array)

assess_df['precision_score'] = assess_df.apply(lambda x: compute_precision_score(x['y_val_label'],
                                                                                 x['y_pred_label']), axis=1)

assess_df['recall_score'] = assess_df.apply(lambda x: compute_recall_score(x['y_val_label'],
                                                                                 x['y_pred_label']), axis=1)

assess_df = assess_df[assess_df.precision_score!=-1]

print('Average precision score: ')

print(assess_df.precision_score.mean())

print('Average recall score: ')

print(assess_df.recall_score.mean())
