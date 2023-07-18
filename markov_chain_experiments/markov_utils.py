import random
import numpy as np
import tensorflow as tf
from scipy.stats import chi2
keras = tf.keras # small issue with Pylance and Tensorflow
from keras.preprocessing import text
from keras.preprocessing.sequence import skipgrams
from markov_conf import *

def transition_matrix(transitions):
    """
    This method generates a matrix to describe the transitions of a
    Markov chain.
    Source: https://stackoverflow.com/questions/46657221/generating-markov-transition-matrix-in-python

    Parameters
    ----------
    transitions: list, transitions of the Markov chain.

    Returns
    -------
    M: numpy.array, contains the calculated transition matrix for the given Markov chain.

    """
    n = 1 + max(transitions) # number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    # now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]

    # sum 1 per columns
    return np.array(M).transpose()

def generate_sequence(n, transitions_sequence, fraction_outliers):
    """
    This method generates a sequence of symbols based of a given transition matrix. It also creates
    a determinate number of "detectable" and "undetectable" anomalies in the sequence.
    On one hand, a "detectable" anomaly is a jump of two or more steps in the next letter of the
    sequence. e.g.:  ... A A C B ...
    On the other hand, a "undetectable" anomaly is a consecutive step wich is labelled as anormal.

    Parameters
    ----------
    n: int, size of the sequence about to be generated.
    transitions_sequence: list, Markov chain to determinate the transitions matrix.
    fraction_outliers: float, number betwenn 0.0 and 1.0to determinate the proportion of anomalies
                       in the generated sequence.

    Returns
    -------
    string_output:
    labels: list, contains the truth value of each symbol. L in the sequence to determinate if exists an
            anomaly or not.
    perplex: list, contains the calculated perplexity of each symbol in the sequence. "Detectable"
             will have infinite perplexity.
    """
    m = np.array(transition_matrix(transitions_sequence))
    #print(m)
    n_states = m.shape[0]
    uniform_prob = np.ones(n_states) / n_states
    initial_state = np.zeros(n_states)
    initial_state[0] = 1

    current_state = initial_state
    string_output = []
    labels = []
    perplex = []

    for _ in range(n):
        outlier = np.random.rand()<=fraction_outliers
        last_state = current_state.copy()
        if outlier:
            # choose the next state with uniform probability
            next = np.random.choice(np.arange(n_states), p=uniform_prob)
            current_state = np.zeros(n_states)
            current_state[next] = 1
            string_output.append(chr(65+next))
            labels.append(np.array(True))
        else:
            prob_states = m @ current_state
            # Choose the nex state with transition probability
            next = np.random.choice(np.arange(n_states), p=prob_states)
            current_state = np.zeros(n_states)
            current_state[next] = 1
            string_output.append(chr(65+next))
            labels.append(np.array(False))

        # perplexity is the inverse of the probability of a symbol appearance
        perplex.append(1.0/m[np.where(current_state)[0],np.where(last_state)[0]])

    return string_output, labels, perplex, m

def generate_sequence_multivariate(n, transitions_sequence, fraction_outliers, repeats):
    """
    Generates a sequence of symbols based on a transition matrix. This method with the repeats
    parameter can determinate the number of times each symbol will be repeared in the sequence.
    Example with repeats = 3: A A A B B B C C C ...
    This can be useful to form a multivariate sequence by joining some of these sequences.

    Parameters
    ----------
    n: int, size of the sequence about to be generated.
    transitions_sequence: list, Markov chain to determinate the transitions matrix.
    fraction_outliers: float, number betwenn 0.0 and 1.0to determinate the proportion of anomalies
                       in the generated sequence.
    repeats: int, number of times each symbol in the sequence will be repeated.

    Returns
    ------- 
    string_output:
    labels: list, contains the truth value of each symbol in the sequence to determinate if exists an
            anomaly or not.
    perplex: list, contains the calculated perplexity of each symbol in the sequence. "Detectable"
             will have infinite perplexity.
    m: numpy.array, contains the calculated transition matrix for the given Markov chain.
    """
    m = np.array(transition_matrix(transitions_sequence))
    #print(m)
    n_states = m.shape[0]
    uniform_prob = np.ones(n_states) / n_states
    initial_state = np.zeros(n_states)
    initial_state[0] = 1

    current_state = initial_state
    string_output = []
    labels = []
    perplex = []

    for _ in range(round(n/repeats)+1):
        outlier = np.random.rand()<=fraction_outliers
        last_state = current_state.copy()
        if outlier:
            # choose the next state with uniform probability
            next = np.random.choice(np.arange(n_states), p=uniform_prob)
            current_state = np.zeros(n_states)
            current_state[next] = 1
            for i in range(repeats):
                string_output.append(chr(65+next))
                if i == 0:
                    labels.append(np.array(True)) 
                else:
                    labels.append(np.array(False))  
        else:
            prob_states = m @ current_state
            # Choose the nex state with transition probability
            next = np.random.choice(np.arange(n_states), p=prob_states)
            current_state = np.zeros(n_states)
            current_state[next] = 1
            for _ in range(repeats):
                string_output.append(chr(65+next))
                labels.append(np.array(False))

        # perplexity is the inverse of the probability of a symbol appearance
        perplex.append(1.0/m[np.where(current_state)[0],np.where(last_state)[0]])

    return string_output, labels, perplex, m

def generate_sequence_not_random(n, transitions_sequence, fraction_outliers):
    m = np.array(transition_matrix(transitions_sequence))
    #print(m)
    n_states = m.shape[0]
    uniform_prob = np.ones(n_states) / n_states
    initial_state = np.zeros(n_states)
    initial_state[0] = 1

    current_state = initial_state
    string_output = []
    labels = []
    perplex = []

    for _ in range(n):
        outlier = np.random.rand()<=fraction_outliers
        last_state = current_state.copy()
        if outlier:
            # choose the next state with uniform probability
            probs = np.zeros(n_states)
            if last_state[0] == 1:
                single_prob = (np.ones(n_states-2)/(n_states-2))[0]
                probs[2:] = single_prob
            elif last_state[len(last_state)-1] == 1:
                single_prob = (np.ones(n_states-2)/(n_states-2))[0]
                probs[:len(last_state)-2] = single_prob
            else:
                single_prob = (np.ones(n_states-3)/(n_states-3))[0]
                idx = np.where(last_state == 1)[0][0]
                probs[:idx-1] = single_prob
                probs[idx+2:] = single_prob

            next = np.random.choice(np.arange(n_states), p=probs)
            current_state = np.zeros(n_states)
            current_state[next] = 1
            string_output.append(chr(65+next))
            labels.append(np.array(True))
        else:
            prob_states = m @ current_state
            # Choose the nex state with transition probability
            next = np.random.choice(np.arange(n_states), p=prob_states)
            current_state = np.zeros(n_states)
            current_state[next] = 1
            string_output.append(chr(65+next))
            labels.append(np.array(False))

        # perplexity is the inverse of the probability of a symbol appearance
        perplex.append(1.0/m[np.where(current_state)[0],np.where(last_state)[0]])

    return string_output, labels, perplex, m

def dataframe_to_corpus(data):
    """
    This method performs a transformation of an input numpy array to an output string containing
    each element (letter) of the input array.

    Parameters:
    -----------
    data: numpy.array, large list of symbols (in this case letters).

    Returns:
    --------
    corpus: list, element [0] of this list is string with all symbols concatenated.
    """
    # convert Dataframe to a large string containing all words
    corpus = []
    smd_text = ""  # empty string
    # fill corpus
    for i in data:
        smd_text += i
        smd_text += " "

    corpus.append(smd_text)

    return corpus

def generate_skipgrams(corpus, window_size, func, debug=False):
    """
    Generates skipgrams froma given corpus.

    Parameters:
    -----------
    corpus: string, the skipgrams will be generated with this string.
    window_size: int, size of the selected window to generate the skipgrams.
    debug: bool, variable to determine if after generate the skipgramss someof the will be
           printed in the terminal.

    Returns:
    --------
    skip_grams: list, generated skipgrams.
    word2id: dictionary, provides information about the relatioship between words and IDs.
    wids: 
    id2word: dictionary, provides informationabput the relationshipbetween words and IDs.
    """
    # create and fit tokenizer with corpus
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(corpus)

    # create dictionaries with relationship between Ids and words
    word2id = tokenizer.word_index
    id2word = {v: k for k, v in word2id.items()}

    vocab_size = len(word2id) 

    wids = [[word2id[w]
                for w in text.text_to_word_sequence(doc)] for doc in corpus]

    print('Vocabulary size:', vocab_size)
    print('Most frequent words:', list(word2id.items())[-5:])

    # generate skip-grams
    skip_grams = [
        func(wid, vocabulary_size=vocab_size, window_size=window_size) for wid in wids]

    # show some skip-grams
    #if debug:
    #    pairs, labels = skip_grams[0][0], skip_grams[0][1]
    #    for i in range(10):
    #        print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
    #            id2word[pairs[i][0]], pairs[i][0],
    #            id2word[pairs[i][1]], pairs[i][1],
    #            labels[i]))

    return skip_grams, word2id, wids, id2word

def skipgrams_beaf_closest(sequence, vocabulary_size, window_size, negative_samples=1.0):
    """
    Generates skipgram word tuples.
    It takes into account the word before the target word and the word after it.

    This function transforms a sequence of word indexes (list of integers)
    into tuples of words of the form:
    
    - (word, (word just before in the window, word just after in the window)), with label 1 (positive samples).
    - (word, (random word from the vocabulary, random word from the vocabulary)), with label 0 (negative samples).

    """
    couples = []
    labels = []
    for i, wi in enumerate(sequence):
        #if i == 0 or i == len(sequence)-1:
        #    continue
        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)

        wj = []
        for j in range(window_start, window_end):
            if j == i-1 or j == i+1:
                # if wj is the closest word by the left or the right it will be the context
                wj.append(sequence[j])
                if not wj:
                    continue
            if len(wj) == 2:
                # the context tuple contains the previous and next word
                couples.append([wi, wj])
                labels.append(1)
    
    # generate random tuples for negative samples
    if negative_samples > 0:
        num_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)

        couples += [
            [words[i % len(words)], [random.randint(1, vocabulary_size - 1), random.randint(1, vocabulary_size - 1)]]
            for i in range(num_negative_samples)
        ]
        labels += [0] * num_negative_samples
    
    # shuffle time
    random.seed(RANDOM_SEED)
    random.shuffle(couples)
    random.seed(RANDOM_SEED)
    random.shuffle(labels)

    return couples, labels

def skipgrams_beaf(sequence, vocabulary_size, window_size, negative_samples=1.0):
    """
    Generates skipgram word pairs.

    This function transforms a sequence of word indexes (list of integers)
    into tuples of words of the form:

    - (word, (word before in the same window, word after in the same window)), with label 1 (positive samples).
    - (word, (random word from the vocabulary, random word from the vocabulary)), with label 0 (negative samples).

    """
    couples = []
    labels = []
    for i, wi in enumerate(sequence):
        #if i == 0 or i == len(sequence)-1:
        #    continue
        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)

        for j in range(window_start, i):
            for z in range (i+1, window_end):
                wj = [sequence[j], sequence[z]]
                # the context tuple contains the previous and next word
                couples.append([wi, wj])
                labels.append(1)
                
    # generate random tuples for negative samples
    if negative_samples > 0:
        num_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)

        couples += [
            [words[i % len(words)], [random.randint(1, vocabulary_size - 1), random.randint(1, vocabulary_size - 1)]]
            for i in range(num_negative_samples)
        ]
        labels += [0] * num_negative_samples
    
    # shuffle time
    random.seed(RANDOM_SEED)
    random.shuffle(couples)
    random.seed(RANDOM_SEED)
    random.shuffle(labels)

    return couples, labels

def skipgramns_n_bef(sequence, vocabulary_size, window_size, negative_samples=1.0):
    """
    Generates skipgram word tuples.
    Here the window_size param corresponds to the length of words of the context.

    This function transforms a sequence of word indexes (list of integers)
    into tuples of words of the form:

    - (word, (n-preceding words)), with label 1 (positive samples).
    - (word, (n-random word from the vocabulary)), with label 0 (negative samples).

    """
    couples = []
    labels = []
    i = window_size + 1
    while i < len(sequence):
        wi = sequence[i-1]
        wj = []
        for j in range(i-1-window_size, i-1):
            wj.append(sequence[j])
        if len(wj) == window_size:
            couples.append([wi, wj])
            labels.append(1)
        i += 1

    # generate random tuples for negative samples
    if negative_samples > 0:
        num_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)

        for y in range(num_negative_samples):
            wrd = words[y % len(words)]
            context = []
            for j in range(window_size):
                context.append(random.randint(1, vocabulary_size - 1))
            couples.append([wrd, context])
        labels += [0] * num_negative_samples

    # shuffle time
    random.seed(RANDOM_SEED)
    random.shuffle(couples)
    random.seed(RANDOM_SEED)
    random.shuffle(labels)

    return couples, labels

def estimate_perplexity(y_pred):
    """
    This method calculates the perplexity of symbols based on a probability.

    Parameters:
    -----------
    y_pred: numpy.array, output of the skipgram model. It consists in an array with probabilities.

    Returns:
    --------
    perplex: numpy.array, calculated perplexity foreach symbol.
    """
    # Since we are working with pairs of words, this is a bigram
    # High probability will be translated into less perplex
    return (1.0/y_pred)

def find_anomaly_threshold(values: np.ndarray, threshold_mode: str, threshold_value: float, n_features: int=0) -> float:
    """
    Estimate anomaly threshold based on train values and a given mode.
    Parameters:
    -----------

    Returns:
    --------
    """
    if threshold_mode == 'chi2':
        return chi2.ppf(threshold_value, df=n_features)
    if threshold_mode == 'percentile':
        return np.percentile(values, threshold_value)
    elif threshold_mode == 'sigma':
        return np.array(values).std() * threshold_value
    elif threshold_mode == 'max':
        return values.max()
    else:
        raise RuntimeError("Invalid threshold_mode configuration.")

def detect_anomalies(y_pred):
    """
    Apply binary mask to predictions:
         -1 --> outlier (true)
          1 --> inlier  (false)
    Only for classic anomaly detection methods like LOF or One Class SVM.

    Parameters:
    -----------
    y_pred: numpy.array, output predictions of the algorithm.

    Returns:
    --------
    y_pred: numpy.array, masked output.

    """
    y_pred = y_pred == -1
    return y_pred