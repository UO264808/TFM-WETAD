import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
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
    final_model.compile(loss='mse', optimizer=keras.optimizers.Adam())

    # print summary of the model
    final_model.summary()

    return final_model

def train(skip_grams, model):
    first_elem = list(zip(*skip_grams[0][0]))[0]
    second_elem = list(zip(*skip_grams[0][0]))[1]
    X = [np.array(first_elem, dtype='int32'), np.array(second_elem, dtype='int32')]
    Y = np.array(skip_grams[0][1])
    model.fit(X, Y, batch_size=32, epochs=EPOCHS, verbose=2, validation_split=0.2)

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
transitions = [0,0,1,1,2,3,4,5,5,5,4,4,4,3,3,2,1,1,1,1,2,3,4,5,6,6,7,7,8,8,8,8,9,9,9,8,7,6,6,7,7,8,7,6,6,5,5,5,5,4,4,3,2,2,1,2,2,3,2,1,1,0,0,0,0,0]
#transitions = [0,0,1,1,2,3,4,5,5,5,4,4,4,3,3,2,1,1,1,1,2,3,4,5,6,6,7,7,8,8,8,8,9,10,9,10,11,12,13,12,11,10,10,9,8,7,6,6,7,7,8,7,6,6,5,5,5,5,4,4,3,2,2,1,2,2,3,2,1,1,0,0,0,0,0]
#transitions = [0,0,1,1,2,3,4,5,5,5,4,4,4,3,3,2,1,1,1,1,2,3,4,5,6,6,7,7,8,8,8,8,9,10,9,10,11,12,13,14,14,13,14,15,16,17,17,16,15,14,13,12,11,10,10,9,8,7,6,6,7,7,8,7,6,6,5,5,5,5,4,4,3,2,2,1,2,2,3,2,1,1,0,0,0,0,0]

VOCAB_SIZE = len(set(transitions))+1

# fix random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# generate train sequence (fraction_outliers=0)
data_train, labels_train, _, _ = generate_sequence(N_TRAIN, transitions, 0)

data_train_corpus = dataframe_to_corpus(data_train)
skip_grams, word2id, wids, id2word = generate_skipgrams(
            data_train_corpus, WINDOW_SIZE, skipgrams)

# check full data
#full_data_corpus = dataframe_to_corpus(full_data)
#_, full_word2id, _ = generate_skipgrams(full_data_corpus, False)

# build neural network model
model = build_nn(VOCAB_SIZE, EMBED_SIZE)

# train model
train_loss, threshold_val = train(skip_grams, model)
print('Calculated threshold: {:.4f}'.format(threshold_val))

# obtain and save embeddings
weights = model.layers[3].get_weights()[0]
#embeddings = np.array(weights)
#np.save('embeddings.npy', embeddings)
embedding_df = pd.DataFrame(weights[1:], index=id2word.values())
print(embedding_df)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# PCA
pca = PCA(n_components=2)
emb = pca.fit(embedding_df).transform(embedding_df)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=emb[:, 0], y=emb[:, 1], s=70, hue=id2word.values())
plt.title("Gráfica PCA - Tamaño vocabulario=10", pad=15)
plt.xlabel("Primera componente principal")
plt.ylabel("Segunda componente principal")
plt.savefig("Markov_2d_PCA.png")

# TSNE
tsne = TSNE(n_components=2, n_iter=10000, perplexity=3)
T = tsne.fit_transform(embedding_df)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=T[:, 0], y=T[:, 1], s=70, hue=id2word.values())
plt.title("Gráfica T-SNE - Tamaño vocabulario=10", pad=15)
plt.savefig("Markov_2d_TSNE.png")
