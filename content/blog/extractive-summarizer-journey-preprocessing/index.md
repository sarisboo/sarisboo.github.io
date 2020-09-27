---
title: "Extractive Summarizer Journey: Preprocessing"
date: "2020-09-07T11:00:37.121Z"
description: How to preprocess text data for NLP.
---

## Get the data 
This example makes use the `cnn_dailymail` dataset, it contains 2 features news `articles` and their corresponding `highlights`. Since you need to make sure you have enough instances to train, validate and test our model you can load the `train`, `validation` and `test` sets separately from tensorflow datasets.

```
ds_train = tfds.load(name="cnn_dailymail", split="train")
ds_val = tfds.load(name="cnn_dailymail", split="validation")
ds_test = tfds.load(name="cnn_dailymail", split="test")
```

For easier data wrangling and preprocessing, you need to transform these datasets to pandas data frames. You can either use a custom function or use the `as_dataframe()` method (nightly version). Either way, you might need to convert the returned data frame rows from bytes to string before applying further transformations.

## Preliminary Cleanup
This can be the most time-consuming step. Once you obtain your dataset, you need to wrangle the data to see what it looks like and apply the necessary cleaning steps. 
These steps can depend on the original input data. In this specific case, to extract raw text from the data, you need to remove all non-textual information by making use of regex expressions. Here is an example of text clean up for the input data using a concatenated regex expression.

```
# Cleanup the datasets
# Compose concatenated regex expression to clean data faster
start_of_string = "^\s*"
remove_cnn = ".*\((CNN|EW.com)\)?(\s+--\s+)?"
remove_by_1 = (
    "By \.([^.*]+\.)?([^.]*\. PUBLISHED: \.[^|]*\| \.)? UPDATED:[^.]+\.[^.]+\.\s*"
)
remove_by_2 = "By \.([^.*]+\.)?(\sand[^.*]+\.\s*)?(UPDATED[^.*]+\.[^.*]+\.\s*)?(\slast[^.*]+\.\s*)?"
remove_last_updated = "Last[^.*]+\.\s"
remove_twitter_link = "By \.([^.*]+\.)\s*Follow\s@@[^.*]+\.\s+"
remove_published = "(PUBLISHED[^.*]+\.[^.*]+\.[^.*]+\.\s*)(UPDATED[^.*]+\.[^.*]+\.\s*)?"
# end_of_string = '[\'"]*\s*$'

r_cleanup_source = (
    start_of_string
    + "("
    + "|".join(
        [
            remove_cnn,
            remove_by_1,
            remove_by_2,
            remove_last_updated,
            remove_twitter_link,
            remove_published,
        ]
    )
    + ")"
)

r_cleanup = re.compile(r_cleanup_source)
```
You can then apply these clean up steps on the input data using the following function.
```
def cleanup(text):
    return r_cleanup.sub("", text)
```

After you get rid of all the irrelevant information for your analysis, you need to proceed to pre-processing the plain text you get to make it NLP software friendly.

##  Preprocessing
During the preprocessing steps, you need to regularly check for and deal with missing values before you pass them into nltk's preprocessing steps because your code will not work when you pass in missing values.

### Sentence Tokens
It is the process of breaking up text into sentences, the simplest way to do it would be to split at full stops, but there might be abbreviations like E.U, Ms. or Dr. that make this step less obvious. 
Fortunately you can use libraries like nltk( `sent_tokenize` )or Spacy in most cases to take care of this. In the case of the `cnn_dailymail` the nltk sentence tokenizer inconsistently splits the sentences in cases like for example:

```
sent_tokenize('Bishop John Folda, of North Dakota, is taking time off after being diagnosed .He contracted the infection through contaminated food in Italy .Church members in Fargo, Grand Forks and Jamestown could have been exposed .')
```

Outputs:

```
['Bishop John Folda, of North Dakota, is taking time off after being diagnosed .He contracted the infection through contaminated food in Italy .Church members in Fargo, Grand Forks and Jamestown could have been exposed .']
```

The expected output would be:
```
['Bishop John Folda, of North Dakota, is taking time off after being diagnosed','He contracted the infection through contaminated food in Italy', 'Church members in Fargo, Grand Forks and Jamestown could have been exposed']
```

This might be because of the trailing whitespace between the last word of the sentence so for now, so you might want to use a custom sentence tokenizer to work around this limitation.

One approach to this is to create a `text_id` out of the `index` column to be later be able to identify which sentence belongs to each text. 

You then split sentences the naive way by using a simple sentence boundary regex and apply it to the data.

```
def split_sentences(text):
    # Segment texts into sentences
    r_sentence_boundary = re.compile(
        r"\s?[.!?]\s?"
    )  # Modify this to not include abbreviations and other exceptions
    return r_sentence_boundary.split(text)[:-1]


# Split text by sentences
def split_by_sentence(df):
    df["sentences"] = df["article"].apply(lambda x: split_sentences(str(x)))
```

Next, you make a list of tuples to keep track of `text_id` by `sentence`.
```
# Make a list of (text_id, sentence_list) pairs
def tup_list_maker(tup_list):
    """
    Takes a list of tuples with index 0 being the text_id and index 1 being a
    list of sentences and broadcasts the text_id to each sentence
    """
    final_list = []
    for item in tup_list:
        index = item[0]
        sentences = item[1]
        for sentence in sentences:
            pair = (index, sentence)
            final_list.append(pair)
    return final_list
```

Then use this function to get each sentence with its own `text_id`
```
def create_full_tuple(df):
    tuples = list(zip(df["text_id"], [sentence for sentence in df["sentences"]]))
    tup_list = tup_list_maker(tuples)
    # Converting the tuples list into a dataframe
    sentences = pd.DataFrame(tup_list, columns=["text_id", "sentence"])
    return sentences
```

And finally you assemble the full data frame with split sentences, text ids and the `is_summary` columns which is the labels columns indicating which sentence belongs to the summary.
```
def create_full_final_dataframe(df):

    """
    Creates the final segmented dataframe with the `is_summary` column
    """

    dataframe = make_text_id(df)
    df_article, df_highlights = split_into_2_dfs(dataframe)

    df_article["sentences"] = df_article["article"].apply(
        lambda x: split_sentences(str(x))
    )
    df_highlights["sentences"] = df_highlights["highlights"].apply(
        lambda x: split_sentences(str(x))
    )
    segmented_df_articles = create_full_tuple(df_article)
    segmented_df_highlights = create_full_tuple(df_highlights)

    # Create targets for dataframes
    segmented_df_articles["is_summary_sentence"] = 0
    segmented_df_highlights["is_summary_sentence"] = 1

    # Stack the 2 dataframes and order by `text_id` column
    return segmented_df_articles.append(
        segmented_df_highlights, ignore_index=True
    ).sort_values(by=["text_id"])
```

### Word Tokens
For this step, it's convenient to use nltk's `word_tokenize` splitter to split each sentence into word tokens. This tokenizer splits words based on punctuation marks. Although it is not always splitting words correctly (for instance $10,000 is identified as two separate tokens ['$', '10,000] and €1000 is identified as a single token ['€1000']),it is convenient to make use of it as a preliminary step for now.
```
from nltk.tokenize import word_tokenize
def tokenizer(df, column):
    df[column].dropna(inplace=True)
    df["tokens"] = df[column].apply(word_tokenize)
    return df
```

### Stopwords Removal 
Next you need to remove stop words, stop words are words like as, the, of, is. They carry no real meaning for the sentence so they're not really relevant to your analysis. The nltk library provides a list of common English stop words but it is by no means standard and stop words may vary depending on the problem you're trying to solve.
In this case, you could use the nltk list for simplicity.
```
from nltk.corpus import stopwords
from ast import literal_eval 

def stop_words_remover(tokenized_sent):
    """
    Removes stop words from a tokenized sentence
    """
    # Convert string back to list

    filtered_sentence = []
    stop_words = set(stopwords.words("english"))
    for word in literal_eval(tokenized_sent):
        if word not in stop_words:
            filtered_sentence.append(word)
    return filtered_sentence
```

### Stemming (Porter Stemmer)
Stemming is the process of suffix stripping, it reduces the word to a base form that is representative of all the variants of that word.
 
A stemmer uses a set of fixed rules to decide how the word should be stripped (it might not always end up in a linguistically correct base form).
For this analysis, you can use nltk's Porter Stemmer for its simplicity and speed.

```
from nltk.stem import PorterStemmer
from ast import literal_eval
import pandas as pd

porter = PorterStemmer()

def stemmer(stemmed_sent):
    """
    Removes stop words from a tokenized sentence
    """
    porter = PorterStemmer()
    stemmed_sentence = []
    for word in literal_eval(stemmed_sent):
        stemmed_word = porter.stem(word)
        stemmed_sentence.append(stemmed_word)
    return stemmed_sentence
```

Voilà ! You just cleaned and preprocessed text data for text summarisation. The next step is feature engineering to extract important characteristics of the text to then feed it to our machine learning algorithm.



