# Part-of-Speech-Tagging_NLP
This project specializes in Natural Language Processing and is based on Part-of-Speech (POS) tagging, the process of assigning a part-of-speech tag (Noun, Verb, Adjective...) to each word in an input text.


<a name='0'></a>
# A: Data Sources
This project will use two tagged data sets collected from the **Wall Street Journal (WSJ)**. 
 
- One data set (**WSJ-2_21.pos**) will be used for **training**.
- The other (**WSJ-24.pos**) for **testing**. 
- The tagged training data has been preprocessed to form a vocabulary (**hmm_vocab.txt**). 
- The words in the vocabulary are words from the training set that were used two or more times. 
- The vocabulary is augmented with a set of 'unknown word tokens', described below. 

The training set will be used to create the emission, transmission and tag counts. 

The test set (WSJ-24.pos) is read in to create `y`. 
- This contains both the test text and the true tag. 
- The test set has also been preprocessed to remove the tags to form **test_words.txt**. 
- This is read in and further processed to identify the end of sentences and handle words not in the vocabulary using functions provided in **utils_pos.py**. 
- This forms the list `prep`, the preprocessed text used to test our POS taggers.

A POS tagger will necessarily encounter words that are not in its datasets. 
- To improve accuracy, these words are further analyzed during preprocessing to extract available hints as to their appropriate tag. 
- For example, the suffix 'ize' is a hint that the word is a verb, as in 'final-ize' or 'character-ize'. 
- A set of unknown-tokens, such as '--unk-verb--' or '--unk-noun--' will replace the unknown words in both the training and test corpus and will appear in the emission, transmission and tag data structures.


<a name='1'></a>
# Part 1: Parts-of-speech tagging 

<a name='1.1'></a>
## Part 1.1 - Training

In this project, I will start with the simplest possible parts-of-speech tagger and then will build up to the state of the art. 

Note that, there are words that are not ambiguous. 
- For example, the word `is` is a verb and it is not ambiguous. 
- In the `WSJ` corpus, 86% of the token are unambiguous (meaning they have only one tag) 
- About 14\% are ambiguous (meaning that they have more than one tag)

Before I start predicting the tags of each word, there is a need to compute a few dictionaries that will help us to generate the tables. 


#### Transition counts
- The first dictionary is the `transition_counts` dictionary which computes the number of times each tag happened next to another tag. 

This dictionary will be used to compute: 
***P(t_i |t_{i-1})***

This is the probability of a tag at position ***i*** given the tag at position ***i-1***.

In order to compute equation 1, I will create a `transition_counts` dictionary where 
- The keys are `(prev_tag, tag)`
- The values are the number of times those two tags appeared in that order. 


#### Emission counts
The second dictionary I will compute is the `emission_counts` dictionary. This dictionary will be used to compute:

***P(w_i|t_i)***

In other words, I will use it to compute the probability of a word given its tag. 
In order to compute equation 2, I will create an `emission_counts` dictionary where 
- The keys are `(tag, word)` 
- The values are the number of times that pair showed up in the training set. 


#### Tag counts
The last dictionary that I will create is the `tag_counts` dictionary. 
- The key is the tag 
- The value is the number of times each tag appeared.


<a name='1.2'></a>
### Part 1.2 - Testing

I will test the accuracy of the parts-of-speech tagger using the `emission_counts` dictionary. 
- Given my preprocessed test corpus `prep`, I will assign a parts-of-speech tag to every word in that corpus. 
- Using the original tagged test corpus `y`, I will then compute what percent of the tags I got correct. 

I will implement the function `predict_pos` that computes the accuracy of this model.

##### Expected Output
```CPP
Accuracy of prediction using predict_pos is 0.8889
```


<a name='2'></a>
# Part 2: Hidden Markov Models for POS

Now I will build something more context specific. Concretely, I will be implementing a Hidden Markov Model (HMM) with a Viterbi decoder
- The HMM is one of the most commonly used algorithms in Natural Language Processing, and is a foundation to many deep learning techniques.
- In addition to parts-of-speech tagging, HMM is used in speech recognition, speech synthesis, etc. 
- By completing this part, I am expecting to get a 95% accuracy on the same dataset I used in Part 1.

The Markov Model contains a number of states and the probability of transition between those states. 
- In this case, the states are the parts-of-speech. 
- A Markov Model utilizes a transition matrix, `A`. 
- A Hidden Markov Model adds an observation or emission matrix `B` which describes the probability of a visible observation when we are in a particular state. 
- In this case, the emissions are the words in the corpus
- The state, which is hidden, is the POS tag of that word.


<a name='3'></a>
# Part 3: Viterbi Algorithm and Dynamic Programming

In this part of the project I will implement the Viterbi algorithm which makes use of dynamic programming. Specifically, I will use the two matrices, `A` and `B` to compute the Viterbi algorithm. The process have been decomposed into three main steps. 

* **Initialization** - In this part I initialize the `best_paths` and `best_probabilities` matrices that I will be populating in `feed_forward`.
* **Feed forward** - At each step, I calculate the probability of each path happening and the best paths up to that point. 
* **Feed backward**: This allows me to find the best path with the highest probabilities. 

<a name='3.1'></a>
## Part 3.1:  Initialization 

I will start by initializing two matrices of the same dimension. 
- best_probs: Each cell contains the probability of going from one POS tag to a word in the corpus.
- best_paths: A matrix that helps me trace through the best possible path in the corpus. 

<a name='3.2'></a>
## Part 3.2 Viterbi Forward

In this part of the project, I will implement the `viterbi_forward` segment. In other words, I will populate the `best_probs` and `best_paths` matrices.
- Walk forward through the corpus.
- For each word, compute a probability for each possible tag. 
- Unlike the previous algorithm `predict_pos`, this will include the path up to that (word,tag) combination. 


<a name='3.3'></a>
## Part 3.3 Viterbi backward

Now I will implement the Viterbi backward algorithm.
- The Viterbi backward algorithm gets the predictions of the POS tags for each word in the corpus using the `best_paths` and the `best_probs` matrices.


<a name='4'></a>
# Part 4: Predicting on a data set

Compute the accuracy of the prediction by comparing it with the true `y` labels. 
- `pred` is a list of predicted POS tags corresponding to the words of the `test_corpus`. 


##### Expected Output
```CPP
Accuracy of the Viterbi algorithm is 0.9531
```
