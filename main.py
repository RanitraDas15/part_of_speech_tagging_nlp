from utils_pos import get_word_tag, preprocess
import pandas as pd
from collections import defaultdict
import math
import numpy as np

# load in the training corpus
with open("WSJ_02-21.pos", "r") as f:
    training_corpus = f.readlines()

# read the vocabulary data, split by each line of text, and save the list
with open("hmm_vocab.txt", "r") as f:
    voc_l = f.read().split("\n")

# vocab: dictionary that has the index of the corresponding words
vocab = {}
# Get the index of the corresponding words
for i, word in enumerate(sorted(voc_l)):
    vocab[word] = i

# load in the test corpus
with open("WSJ_24.pos", "r") as f:
    y = f.readlines()

# corpus without tags, preprocessed
_, prep = preprocess(vocab, "test.words")


def create_dictionaries(training_corpus, vocab):
    # initialize the dictionaries using defaultdict
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    # Initialize "prev_tag" (previous tag) with the start state, denoted by '--s--'
    prev_tag = "--s--"
    # use 'i' to track the line number in the corpus
    i = 0
    # Each item in the training corpus contains a word and its POS tag
    # Go through each word and its tag in the training corpus
    for word_tag in training_corpus:
        # get the word and tag using the get_word_tag helper function (imported from utils_pos.py)
        word, tag = get_word_tag(word_tag, vocab)
        # Increment the transition count for the previous word and tag
        transition_counts[(prev_tag, tag)] += 1
        # Increment the emission count for the tag and word
        emission_counts[(tag, word)] += 1
        # Increment the tag count
        tag_counts[tag] += 1
        # Set the previous tag to this tag (for the next iteration of the loop)
        prev_tag = tag
    return emission_counts, transition_counts, tag_counts


emission_counts, transition_counts, tag_counts = create_dictionaries(
    training_corpus, vocab
)


# get all the POS states
states = sorted(tag_counts.keys())


def predict_pos(prep, y, emission_counts, vocab, states):
    # Initialize the number of correct predictions to zero
    num_correct = 0
    # Get the (tag, word) tuples, stored as a set
    all_words = set(emission_counts.keys())
    # Get the number of (word, POS) tuples in the corpus 'y'
    total = len(y)
    for word, y_tup in zip(prep, y):
        # Split the (word, POS) string into a list of two items
        y_tup_l = y_tup.split()
        # Verify that y_tup contain both word and POS
        if len(y_tup_l) == 2:
            # Set the true POS label for this word
            true_label = y_tup_l[1]
        else:
            # If the y_tup didn't contain word and POS, go to next word
            continue
        count_final = 0
        pos_final = ""
        if word in vocab:
            for pos in states:
                # define the key as the tuple containing the POS and word
                key = (pos, word)
                if key in emission_counts:
                    # get the emission count of the (pos,word) tuple
                    count = emission_counts[key]
                    # keep track of the POS with the largest count
                    if count > count_final:
                        # update the final count (largest count)
                        count_final = count
                        # update the final POS
                        pos_final = pos
            if pos_final == true_label:
                num_correct += 1
    accuracy = num_correct / total
    return accuracy


def create_transition_matrix(alpha, tag_counts, transition_counts):
    # Get a sorted list of unique POS tags
    all_tags = sorted(tag_counts.keys())
    # Count the number of unique POS tags
    num_tags = len(all_tags)
    # Initialize the transition matrix 'A'
    A = np.zeros((num_tags, num_tags))
    # Get the unique transition tuples (previous POS, current POS)
    trans_keys = set(transition_counts.keys())
    # Go through each row of the transition matrix A
    for i in range(num_tags):
        # Go through each column of the transition matrix A
        for j in range(num_tags):
            # Initialize the count of the (prev POS, current POS) to zero
            count = 0
            # Define the tuple (prev POS, current POS)
            # Get the tag at position i and tag at position j (from the all_tags list)
            key = (all_tags[i], all_tags[j])
            # Check if the (prev POS, current POS) tuple exists in the transition counts dictionaory
            if key in transition_counts:
                # Get count from the transition_counts dictionary for the (prev POS, current POS) tuple
                count = transition_counts[key]
            # Get the count of the previous tag (index position i) from tag_counts
            count_prev_tag = tag_counts[all_tags[i]]
            # Apply smoothing using count of the tuple, alpha, count of previous tag, alpha, and number of total tags
            A[i, j] = (count + alpha) / (count_prev_tag + alpha * num_tags)
    return A


# creating Transition probability matrix.
alpha = 0.001
A = create_transition_matrix(alpha, tag_counts, transition_counts)


def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    # get the number of POS tag
    num_tags = len(tag_counts)
    # Get a list of all POS tags
    all_tags = sorted(tag_counts.keys())
    # Get the total number of unique words in the vocabulary
    num_words = len(vocab)
    # Initialize the emission matrix B with places for tags in the rows and words in the columns
    B = np.zeros((num_tags, num_words))
    # Get a set of all (POS, word) tuples from the keys of the emission_counts dictionary
    emis_keys = set(list(emission_counts.keys()))
    # Go through each row (POS tags)
    for i in range(num_tags):
        # Go through each column (words)
        for j in range(num_words):
            # Initialize the emission count for the (POS tag, word) to zero
            count = 0
            # Define the (POS tag, word) tuple for this row and column
            key = (all_tags[i], vocab[j])
            # check if the (POS tag, word) tuple exists as a key in emission counts
            if key in emission_counts.keys():
                # Get the count of (POS tag, word) from the emission_counts d
                count = emission_counts[key]
            # Get the count of the POS tag
            count_tag = tag_counts[all_tags[i]]
            # Apply smoothing and store the smoothed value into the emission matrix B for this row and column
            B[i, j] = (count + alpha) / (count_tag + alpha * num_words)
    return B


# creating emission probability matrix.
B = create_emission_matrix(alpha, tag_counts, emission_counts, list(vocab))


def initialize(states, tag_counts, A, B, corpus, vocab):
    # Get the total number of unique POS tags
    num_tags = len(tag_counts)
    # Initialize best_probs matrix
    # POS tags in the rows, number of words in the corpus as the columns
    best_probs = np.zeros((num_tags, len(corpus)))
    # Initialize best_paths matrix
    # POS tags in the rows, number of words in the corpus as columns
    best_paths = np.zeros((num_tags, len(corpus)), dtype=int)
    # Define the start token
    s_idx = states.index("--s--")
    # Go through each of the POS tags
    for i in range(num_tags):
        # Handle the special case when the transition from start token to POS tag i is zero
        if A[s_idx, i] == 0:
            # Initialize best_probs at POS tag 'i', column 0, to negative infinity
            best_probs[i, 0] = float("-inf")
        # For all other cases when transition from start token to POS tag i is non-zero:
        else:
            # Initialize best_probs at POS tag 'i', column 0
            # Check the formula in the instructions above
            best_probs[i, 0] = math.log(A[s_idx, i]) + math.log(B[i, vocab[corpus[0]]])
    return best_probs, best_paths


best_probs, best_paths = initialize(states, tag_counts, A, B, prep, vocab)


def viterbi_forward(A, B, test_corpus, best_probs, best_paths, vocab):
    # Get the number of unique POS tags (which is the num of rows in best_probs)
    num_tags = best_probs.shape[0]
    # Go through every word in the corpus starting from word 1
    # Recall that word 0 was initialized in `initialize()`
    for i in range(1, len(test_corpus)):
        # For each unique POS tag that the current word can be
        for j in range(num_tags):
            # Initialize best_prob for word i to negative infinity
            best_prob_i = float("-inf")
            # Initialize best_path for current word i to None
            best_path_i = None
            # For each POS tag that the previous word can be:
            for k in range(num_tags):
                # Calculate the probability = best probs of POS tag k, previous word i-1 + log(prob of transition from POS k to POS j) + log(prob that emission of POS j is word i)
                prob = (
                    best_probs[k, i - 1]
                    + math.log(A[k, j])
                    + math.log(B[j, vocab[test_corpus[i]]])
                )
                # check if this path's probability is greater than the best probability up to and before this point
                if prob > best_prob_i:
                    # Keep track of the best probability
                    best_prob_i = prob
                    # keep track of the POS tag of the previous word that is part of the best path.
                    # Save the index (integer) associated with that previous word's POS tag
                    best_path_i = k
            # Save the best probability for the given current word's POS tag and the position of the current word inside the corpus
            best_probs[j, i] = best_prob_i
            # Save the unique integer ID of the previous POS tag into best_paths matrix, for the POS tag of the current word and the position of the current word inside the corpus.
            best_paths[j, i] = best_path_i
    return best_probs, best_paths


best_probs, best_paths = viterbi_forward(A, B, prep, best_probs, best_paths, vocab)


def viterbi_backward(best_probs, best_paths, corpus, states):
    # Get the number of words in the corpus which is also the number of columns in best_probs, best_paths
    m = best_paths.shape[1]
    # Initialize array z, same length as the corpus
    z = [None] * m
    # Get the number of unique POS tags
    num_tags = best_probs.shape[0]
    # Initialize the best probability for the last word
    best_prob_for_last_word = float("-inf")
    # Initialize pred array, same length as corpus
    pred = [None] * m
    # Go through each POS tag for the last word (last column of best_probs) in order to find the row (POS tag integer ID) with highest probability for the last word
    for k in range(num_tags):
        # If the probability of POS tag at row k is better than the previosly best probability for the last word:
        if best_probs[k, -1] > best_prob_for_last_word:
            # Store the new best probability for the last word
            best_prob_for_last_word = best_probs[k, -1]
            # Store the unique integer ID of the POS tag which is also the row number in best_probs
            z[m - 1] = k
    # Convert the last word's predicted POS tag from its unique integer ID into the string representation using the 'states' dictionary store this in the 'pred' array for the last word
    pred[m - 1] = states[k]
    # Find the best POS tags by walking backward through the best_paths
    # From the last word in the corpus to the 0th word in the corpus
    for i in range(len(corpus) - 1, -1, -1):
        # Retrieve the unique integer ID of the POS tag for the word at position 'i' in the corpus
        pos_tag_for_word_i = best_paths[np.argmax(best_probs[:, i]), i]
        # In best_paths, go to the row representing the POS tag of word i and the column representing the word's position in the corpus to retrieve the predicted POS for the word at position i-1 in the corpus
        z[i - 1] = best_paths[pos_tag_for_word_i, i]
        # Get the previous word's POS tag in string form
        # Use the 'states' dictionary, where the key is the unique integer ID of the POS tag, and the value is the string representation of that POS tag
        pred[i - 1] = states[pos_tag_for_word_i]
    return pred


pred = viterbi_backward(best_probs, best_paths, prep, states)


def compute_accuracy(pred, y):
    num_correct = 0
    total = 0
    for prediction, y in zip(pred, y):
        # Split the label into the word and the POS tag
        word_tag_tuple = y.split()
        # Check that there is actually a word and a tag no more and no less than 2 items
        if len(word_tag_tuple) != 2:
            continue
        # store the word and tag separately
        word, tag = word_tag_tuple
        # Check if the POS tag label matches the prediction
        if prediction == tag:
            # count the number of times that the prediction and label match
            num_correct += 1
        # keep track of the total number of examples (that have valid labels)
        total += 1
    return num_correct / total


print(f"Accuracy of prediction using predict_pos is {predict_pos(prep, y, emission_counts, vocab, states):.4f}")
print(f"Accuracy of the Viterbi algorithm is {compute_accuracy(pred, y):.4f}")