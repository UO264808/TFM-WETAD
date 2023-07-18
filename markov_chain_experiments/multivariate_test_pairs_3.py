import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.metrics import confusion_matrix, roc_auc_score
keras = tf.keras # small issue with Pylance and Tensorflow
from keras.preprocessing import text
from keras.preprocessing.sequence import skipgrams
from keras.layers import Dense, Embedding, Reshape, dot
from keras.models import Model, Sequential
from markov_utils import *

def build_nn(vocab_size, embed_size):
    word_model = Sequential()
    word_model.add(Embedding(vocab_size, embed_size,
                                embeddings_initializer="glorot_uniform",
                                input_length=1))
    word_model.add(Reshape((embed_size, )))

    context_model = Sequential()
    context_model.add(Embedding(vocab_size, embed_size,
                                embeddings_initializer="glorot_uniform",
                                input_length=1))
    context_model.add(Reshape((embed_size,)))

    merged_output = dot([word_model.output, context_model.output], axes=1)
    model_combined = Sequential()
    model_combined.add(
        Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid"))
    final_model = Model(
        [word_model.input, context_model.input], model_combined(merged_output))
    final_model.compile(loss="mean_squared_error", optimizer="rmsprop")

    # print summary of the model
    final_model.summary()

    return final_model

def train(skip_grams, model):
    first_elem = list(zip(*skip_grams[0][0]))[0]
    second_elem = list(zip(*skip_grams[0][0]))[1]
    X = [np.array(first_elem, dtype='int32'), np.array(second_elem, dtype='int32')]
    Y = np.array(skip_grams[0][1])
    model.fit(X, Y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, validation_split=0.2)

    # calculate cutoff (anomaly_threshold)
    X_diag = [
        np.array(wids[0][1:], dtype='int32'),
        np.array(wids[0][:-1], dtype='int32')]

    pred = model.predict(X_diag)
    pred = estimate_perplexity(pred)

    anomaly_threshold = find_anomaly_threshold(
        pred, THRESHOLD_MODE, THRESHOLD_VALUE, pred.shape[0])

    return _, anomaly_threshold


def predict(test_data, model):
    # fill Wids array only with test data
    test_wids = [[]]

    # some test words do not seem to have appeared in train
    for w in text.text_to_word_sequence(test_data[0]):
        if not w in word2id:
            word2id[w] = len(word2id)+1
        test_wids[0].append(word2id[w])

    # the context is the previous point
    X_diag = [
        np.array(test_wids[0][1:], dtype='int32'),
        np.array(test_wids[0][:-1], dtype='int32')]

    pred = model.predict(X_diag)

    return pred, test_wids


############################### main ###############################

# generate one transition seqsuecence for Markov's chain
#transitions = generate_random_transitions(N_E, VOCAB_SIZE)
#transitions = [0,0,1,1,2,3,2,3,2,1,1,0,0,0,0,1,1,2,3,4,4,3,3,2,1,0,0,0]
#transitions = [0,0,1,1,2,3,4,5,5,5,4,4,4,3,3,2,1,1,1,1,2,3,4,5,6,6,7,7,8,8,8,8,9,9,9,8,7,6,6,7,7,8,7,6,6,5,5,5,5,4,4,3,2,2,1,2,2,3,2,1,1,0,0,0,0,0]
#transitions = [0,0,1,1,2,3,4,5,5,5,4,4,4,3,3,2,1,1,1,1,2,3,4,5,6,6,7,7,8,8,8,8,9,10,9,10,11,12,13,12,11,10,10,9,8,7,6,6,7,7,8,7,6,6,5,5,5,5,4,4,3,2,2,1,2,2,3,2,1,1,0,0,0,0,0]
#transitions = [0,0,1,1,2,3,4,5,5,5,4,4,4,3,3,2,1,1,1,1,2,3,4,5,6,6,7,7,8,8,8,8,9,10,9,10,11,12,13,14,14,13,14,15,16,17,17,16,15,14,13,12,11,10,10,9,8,7,6,6,7,7,8,7,6,6,5,5,5,5,4,4,3,2,2,1,2,2,3,2,1,1,0,0,0,0,0]

transitions_1 = [0,0,1,1,2,3,2,3,2,1,1,0,0,0,0,1,1,2,3,4,4,3,3,2,1,0,0,0]
transitions_2 = [0,1,1,1,2,3,4,3,3,4,4,4,3,2,2,1,0,1,2,3,3,4,4,3,2,1,0,0]
transitions_3 = [0,0,0,1,2,3,4,4,3,4,4,4,3,2,2,1,0,1,2,3,3,3,3,3,2,1,0,0]

#transitions_1 = [0,0,1,1,2,3,4,5,5,5,4,4,4,3,3,2,1,1,1,1,2,3,4,5,6,6,7,7,8,8,8,8,9,9,9,8,7,6,6,7,7,8,7,6,6,5,5,5,5,4,4,3,2,2,1,2,2,3,2,1,1,0,0,0,0,0]
#transitions_2 = [0,0,1,2,2,2,3,4,4,5,5,5,4,3,3,3,3,3,3,3,2,3,4,5,6,6,7,7,7,7,7,8,8,9,9,8,7,7,6,7,7,8,7,6,6,5,5,5,5,5,4,3,2,2,1,2,2,3,2,2,2,1,2,1,0,0]
#transitions_3 = [0,0,1,1,1,2,3,4,4,4,4,5,4,3,4,4,4,4,4,3,2,3,4,5,6,6,6,6,7,6,7,8,9,9,9,8,7,7,6,7,7,8,7,6,6,5,5,5,5,5,4,3,2,2,1,2,2,3,3,3,2,1,2,1,1,1]

#transitions_1 = [0,0,1,1,2,3,4,5,5,5,4,4,4,3,3,2,1,1,1,1,2,3,4,5,6,6,7,7,8,8,8,8,9,10,9,10,11,12,13,12,11,10,10,9,8,7,6,6,7,7,8,7,6,6,5,5,5,5,4,4,3,2,2,1,2,2,3,2,1,1,0,0,0,0,0]
#transitions_2 = [0,0,0,1,2,3,4,5,6,5,5,5,4,4,3,3,3,3,2,1,2,3,4,5,6,7,7,7,8,7,7,8,9,10,9,10,11,12,13,12,12,11,10,9,8,7,8,9,9,9,8,7,6,5,5,6,5,4,4,3,2,1,1,1,2,2,3,2,2,2,1,1,0,0,0]
#transitions_3 = [0,0,0,1,2,3,4,5,6,7,6,5,4,4,3,3,2,1,0,1,2,3,4,5,6,5,6,7,8,9,10,9,9,10,9,10,11,12,13,13,13,12,11,10,10,10,9,9,9,9,8,7,6,5,5,6,5,4,4,3,2,1,1,1,2,2,3,2,2,2,1,1,0,0,0]

#transitions_1 = [0,0,1,1,2,3,4,5,5,5,4,4,4,3,3,2,1,1,1,1,2,3,4,5,6,6,7,7,8,8,8,8,9,10,9,10,11,12,13,14,14,13,14,15,16,17,17,16,15,14,13,12,11,10,10,9,8,7,6,6,7,7,8,7,6,6,5,5,5,5,4,4,3,2,2,1,2,2,3,2,1,1,0,0,0,0,0]
#transitions_2 = [0,0,1,2,3,3,3,4,5,5,4,3,3,3,3,2,2,3,2,3,2,3,4,5,6,7,8,7,8,9,9,8,9,10,11,10,11,12,13,14,15,14,15,16,17,17,17,16,15,14,14,13,12,11,10,9,8,7,8,9,8,7,8,7,6,6,5,4,3,2,3,4,3,3,2,1,1,2,3,2,2,3,2,1,0,0,0]
#transitions_3 = [0,0,0,0,1,1,1,2,3,4,3,2,2,3,3,2,1,1,1,1,2,3,4,5,6,6,7,7,8,8,8,8,9,10,11,11,11,12,13,14,14,13,14,15,16,17,17,17,16,17,17,17,16,15,14,13,12,12,12,11,10,9,8,7,7,7,7,6,7,6,5,4,3,2,2,1,2,2,3,2,1,0,0,0,0,0,0]

VOCAB_SIZE = (len(set(transitions_1)))**3

# generate train sequence (fraction_outliers=0)
data_train_1, labels_train, _, _ = generate_sequence_multivariate(N_TRAIN, transitions_1, 0, 1)
data_train_2, _, _, _ = generate_sequence_multivariate(N_TRAIN, transitions_2, 0, 3)
data_train_3, _, _, _ = generate_sequence_multivariate(N_TRAIN, transitions_3, 0, 6)

data_train_1 = data_train_1[:N_TRAIN]
data_train_2 = data_train_2[:N_TRAIN]
data_train_3 = data_train_3[:N_TRAIN]

# fix random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# concatenate words
data_train = []
for elem in range(len(data_train_1)):
    data_train.append(data_train_1[elem] + data_train_2[elem] + data_train_3[elem])

data_train_corpus = dataframe_to_corpus(data_train)
skip_grams, word2id, wids, id2word = generate_skipgrams(
            data_train_corpus, WINDOW_SIZE, skipgrams)

# check full data
#full_data_corpus = dataframe_to_corpus(full_data)
#_, full_word2id, _ = generate_skipgrams(full_data_corpus, False)

# build neural network model
model = build_nn(VOCAB_SIZE+1, EMBED_SIZE)

# train model
train_loss, threshold_val = train(skip_grams, model)
print('Calculated threshold: {:.4f}'.format(threshold_val))

# generate test sequence (fraction_outlier=0.10)
data_test_1, labels_test_1, perplexity_1, _ = generate_sequence_multivariate(N_TEST, transitions_1, 0.10, 1)
data_test_2, labels_test_2, perplexity_2, _ = generate_sequence_multivariate(N_TEST, transitions_2, 0.10, 3)
data_test_3, labels_test_3, perplexity_3, _ = generate_sequence_multivariate(N_TEST, transitions_3, 0.10, 6)

data_test_1 = data_test_1[:N_TEST]
data_test_2 = data_test_2[:N_TEST]
data_test_3 = data_test_3[:N_TEST]
labels_test_1 = labels_test_1[:N_TEST]
labels_test_2 = labels_test_2[:N_TEST]  
labels_test_3 = labels_test_3[:N_TEST]

# concatenate words and average labels
data_test = []
labels_test = []
for elem in range(len(data_test_1)):
    data_test.append(data_test_1[elem] + data_test_2[elem] + data_test_3[elem])
    labels_test.append(labels_test_1[elem] or labels_test_2[elem] or labels_test_3[elem])

# evaluate model over test set
data_test_corpus = dataframe_to_corpus(data_test)
y_pred, test_wids = predict(data_test_corpus, model)

# estimate perplexity and detect anomalies
y_pred_perplex = estimate_perplexity(y_pred)
pred = y_pred_perplex > threshold_val
# aux contains the points with anomalies
aux = np.where(pred==True)
# change_points will contain the area
change_points = np.zeros((aux[0].shape[0], 2))
for i in range(change_points.shape[0]):
    change_points[i] = [aux[0][i]-1, aux[0][i]+1]

# plot predicted probabilities
#plt.plot(y_pred_perplex, color='black')
#plt.axhline(y=threshold_val, color='red', linestyle='-')
#plt.show()

# plot detected anomalies
fig, axs = plt.subplots(3)
axs[0].plot(data_test, color='black')
axs[1].plot(labels_test, color='green')
axs[2].plot(y_pred_perplex, color='blue')
for pairs in change_points:
    axs[2].axvspan(pairs[0], pairs[1], facecolor='orange')
axs[2].axhline(y=threshold_val, color='red', linestyle='-')
plt.show()

# delete first label 
labels_test_sk = labels_test[1:]

# calculate metrics
tn, fp, fn, tp = confusion_matrix(labels_test_sk, pred).ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn) 

accuracy = (tp+tn)/(tp+fp+fn+tn)

f1 = (2*precision*recall)/(precision+recall)

print('++++++++ Metrics of Skip Gram NS ++++++++')
print('TP -> {} | FP -> {}'.format(tp, fp))
print('FN -> {} | TN -> {}'.format(fn,tn))
print('+++++++++++++++++++++++++++++++++++++++++++')
print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))
print('Accuracy: {:.4f}'.format(accuracy))
print('F1_score: {:.4f}'.format(f1))
print('AUC_score: {:.4f}'.format(roc_auc_score(labels_test_sk, pred)))

print('[+] Vocabulary size = ' + str(VOCAB_SIZE))

pred_table = pd.DataFrame({'labels_test': np.array(labels_test_sk), 
                           'an_prediction': np.squeeze(pred), 
                           'predicted_perplex': np.squeeze(y_pred_perplex)})
pred_table.to_csv('results_multi_three.csv', index=True)
