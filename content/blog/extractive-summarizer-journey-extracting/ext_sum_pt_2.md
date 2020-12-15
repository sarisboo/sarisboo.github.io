---
title: "Extractive Summarizer Journey: Extracting Features"
date: "2020-11-14T12:09:28.121Z"
description: How to preprocess text data for NLP.
---
This is the second article of a series that aims to explore how to preprocess textual data and extract features to ultimately build a classic extractive summarizer using a machine learning algorithm.

Since we approach the extractive summarisation here as a classification problem. We need to determine which features make a sentence summary worthy. Depending on the type of text you want to summarise you might want to select particular features. Here is an example of features you can use for experimentation (This is only a preselection since we will later perform a more thorough feature selection step to determine which ones are relevant). The features can be statistical (tf-idf, length of a sentence, sentence position) or linguistic (presence of proper noun, presence of specific words, presence of verbs, pronouns, punctuation etc...). As a result, each sentence of the document will be represented by an attribute vector of features. For some features, we will make use of the `nltk` library.

## Proper Noun Feature
The importance of a sentence can also be measured by the number of proper nouns (named entities) present. The more proper nouns, the more important the sentence and thus the more summary worthy it will be.
```
# Extract Singular and plural proper nouns
def proper_noun_count_extractor(word_list):
    """
    Accepts a list of words as input and returns
    counts of NNP and NNPS words present
    """
    tagged_sentence = pos_tag(literal_eval(word_list))

    # Count Proper Nouns
    proper_noun_count = len(
        [word for word, pos in tagged_sentence if pos in ["NNP", "NNPS"]]
    )
    return proper_noun_count
```

## Verb Count Feature
Verbs characterise events. The higher the verb count the more important the sentence.
```
def verb_count_extractor(sentence):
    """Accepts a string as input and tokenizes and
    counts the number of verbs in the string"""
    tagged_sentence = pos_tag(word_tokenize(sentence))

    # Count Verbs
    verb_count = len(
        [
            word
            for word, pos in tagged_sentence
            if pos in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        ]
    )
    return verb_count
```

## Sentence Length
The intuition behind this feature is that short sentences like author names as well as unusually long, descriptive sentences are not not supposed to be part of the summary. The method employed here is a simple word count.
For this example, we will use the `nltk` library.
```
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

nltk.download("averaged_perceptron_tagger")

def sentence_length_extractor(sentence):
    """
    Accepts string as input, tokenizes it excluding puntuation and counts the words
    """
    # splits without punctuatiom
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    only_words_count = len(tokenizer.tokenize(sentence))

    return only_words_count
``` 

## Punctuation Feature
This is based on the intuition that sentences ending in question or exclamation marks, or sentences containing quotes are less important.
```
def question_mark_finder(sentence):
    """
    Returns 1 if sentence contains question mark, 0 otherwise
    """
    if "?" in sentence:
        return 1
    else:
        return 0


def exclamation_mark_finder(sentence):
    """
    Returns 1 if sentence contains question mark, 0 otherwise
    """
    if "!" in sentence:
        return 1
    else:
        return 0


def quotes_finder(sentence):
    """
    Returns 1 if sentence contains question mark, 0 otherwise
    """
    if "'" or "`" in sentence:
        return 1
    else:
        return 0
```

## Pronoun Count Feature
Sentences with high pronoun counts are less likely to be included in summaries unless the pronouns are expanded into corresponding nouns.
```
def pronoun_count(sentence):
    """Accepts string as input and counts teh number of pronouns present"""

    tagged_sentence = pos_tag(word_tokenize(sentence))

    # Count Pronouns
    pron_count = len([word for word, pos in tagged_sentence if pos in ["PRON"]])
    return pron_count
```

## TF-IDF
Tf-idf stands for term frequency inverse document frequency. It measures the importance of a word in a document, collection or corpus. Here we use it at the sentence level but only apply it to the training set to avoid potential data leakage. The word frequencies are subsequently stored as sparse matrices and the `vectorizer` is pickled for later use on the other splits for feature generation.
```
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pickle


# Read the data
# Training set rediced size
train = pd.read_csv(
    "data/interim/cleaned/train_cleaned_step_1.csv.gz",
    compression="gzip",
    nrows=1800000,
)
test = pd.read_csv(
    "data/interim/cleaned/test_cleaned_step_1.csv.gz", compression="gzip"
)
val = pd.read_csv("data/interim/cleaned/val_cleaned_step_1.csv.gz", compression="gzip")

# Drop Nas
train.dropna(inplace=True)
test.dropna(inplace=True)
val.dropna(inplace=True)

# Use training set to compute tfidf as part of feature engineering
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1), analyzer="word")
corpus_train = train["sentence"]
corpus_test = test["sentence"]
corpus_val = val["sentence"]

# Fit transform train and transform test and val set using the same vectorizer
train_sparse_matrix = vectorizer.fit_transform(corpus_train)
test_sparse_matrix = vectorizer.transform(corpus_test)
val_sparse_matrix = vectorizer.transform(corpus_val)

if __name__ == "__main__":
    # save cropped dataset
    train.to_csv(
        "src/features/cropped/train_cropped.csv.gz",
        compression="gzip",
        index=False,
    )
    # Save tfidf sparse matrix for each dataset
    sparse.save_npz(
        "src/features/cropped/tf_idf_feature/train_sparse_matrix.npz",
        train_sparse_matrix,
    )

    sparse.save_npz(
        "src/features/cropped/tf_idf_feature/test_sparse_matrix.npz", test_sparse_matrix
    )

    sparse.save_npz(
        "src/features/cropped/tf_idf_feature/val_sparse_matrix.npz", val_sparse_matrix
    )

    # Pickle vectorizer
    pickle.dump(
        vectorizer,
        open("src/features/cropped/tf_idf_feature/tfidf_vectorizer.pickle", "wb"),
    )
```

## Other features worth exploring
There are many feature worth exploring to improve your model. 
They can be location features such as sentence position in a document or paragraph, similarity features, presence of cue-phrases (such as "in conclusion", "this report", "summary"...), presence of non-essential information ("because", "additionally", etc...) and many others.