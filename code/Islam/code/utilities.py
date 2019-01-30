import io
import os
import re
import nltk
import json

import pandas as pd

from os import listdir, makedirs
from os.path import exists, join
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

class Utils():

    def makedirs(self, directory):
        """
            Checks if directory doesn't exist, then creates it.

            Paras:
                directory: name of directory

            Returns:
                Boolean
        """
        if not exists(directory): 
            makedirs(directory)
            return True
        else:
            return False

    def pd_csv(self, path):
        """
            Converts CSV to pandas Dataframe

            Paras:
                path: path to csv sheet
            
            Return:
                df: extracted dataframe
        """
        return pd.read_csv(path)

    def remove_columns(self, dataframe, column_names):
        """
            Filters out unwanted columns

            Paras:
                dataframe: dataframe to filter
                column_names: columns to keep

            Return:
                df: filtered dataframe
        """
        try:
            dataframe = dataframe[column_names]
            dataframe = dataframe.dropna()
            return dataframe.reset_index(drop = True)
        except (ValueError, KeyError):
            pass

    def concate_columns(self, column_1, column_2):
        """
            Concates columns and adds the strings together.

            Paras:
                column_1, column_2: columns to concatenate
            Returns:
                returns concatenated list of strings
        """
        return [str(column_1[i] + ". " + column_2[i]) for i in range(len(column_1))]

    def merge_workbooks(self, dataframes, column_names):
        """
            Merges multiple workbooks based on column names

            Paras:
                dataframes: dfs to merge
                Column_names: column names to merge on
            Returns:
                df: concatenated dataframes
        """
        return pd.concat([self.remove_columns(df, column_names) for df in dataframes])

    def download_nltk_tools(self, path):
        """
            Downloads NLTK tools

            Paras: 
                path: path to download tools
            Returns:
            None
        """
        if self.makedirs(path) == True:
            nltk.download("punkt", path)
            nltk.download("stopwords", path)
            nltk.download("wordnet", path)
            nltk.data.path.append(path)
            os.remove(join(path, "corpora/stopwords.zip"))
            os.remove(join(path, "corpora/wordnet.zip"))
            os.remove(join(path, "tokenizers/punkt.zip"))

    def remove_contractions(self, raw):
        """
            Removes contractions to clean sentences
            
            Paras:
                raw: raw text data
            Returns:
                raw: cleaned text
        """
        contractions = { 
                        "ain't": "is not",
                        "aren't": "are not",
                        "can't": "cannot",
                        "could've": "could have",
                        "couldn't": "could not",
                        "didn't": "did not",
                        "doesn't": "does not",
                        "don't": "do not",
                        "hadn't": "had not",
                        "hasn't": "has not",
                        "haven't": "have not",
                        "he'd": "he would",
                        "he'll": "he will",
                        "he's": "he is",
                        "how'd": "how did",
                        "how'll": "how will",
                        "how's": "how is",
                        "I'd": "I would",
                        "I'll": "I will",
                        "I'm": "I am",
                        "I've": "I have",
                        "isn't": "is not",
                        "it'd": "it would",
                        "it'll": "it will",
                        "it's": "it is",
                        "let's": "let us",
                        "ma'am": "madam",
                        "mayn't": "may not",
                        "might've": "might have",
                        "mightn't": "might not",
                        "must've": "must have",
                        "mustn't": "must not",
                        "needn't": "need not",
                        "o'clock": "of the clock",
                        "oughtn't": "ought not",
                        "shan't": "shall not",
                        "sha'n't": "shall not",
                        "she'd": "she would",
                        "she'll": "she will",
                        "she's": "she is",
                        "should've": "should have",
                        "shouldn't": "should not",
                        "shouldn't've": "should not have",
                        "so've": "so have",
                        "so's": "so as",
                        "that'd": "that would",
                        "that's": "that is",
                        "there'd": "there had",
                        "there's": "there is",
                        "they'd": "they would",
                        "they'll": "they will",
                        "they're": "they are",
                        "they've": "they have",
                        "to've": "to have",
                        "wasn't": "was not",
                        "we'd": "we would",
                        "we'll": "we will",
                        "we're": "we are",
                        "we've": "we have",
                        "weren't": "were not",
                        "what'll": "what will",
                        "what're": "what are",
                        "what's": "what is",
                        "what've": "what have",
                        "when's": "when is",
                        "when've": "when have",
                        "where'd": "where did",
                        "where's": "where is",
                        "where've": "where have",
                        "who'll": "who will",
                        "who'll've": "who will have",
                        "who's": "who is",
                        "who've": "who have",
                        "why's": "why has",
                        "why've": "why have",
                        "will've": "will have",
                        "won't": "will not",
                        "won't've": "will not have",
                        "would've": "would have",
                        "wouldn't": "would not",
                        "y'all": "you all",
                        "you'd": "you had / you would",
                        "you'll": "you will",
                        "you'll've": "you will have",
                        "you're": "you are",
                        "you've": "you have"
                    }
        if raw in contractions:
            for key, value in contractions.items():
                raw = re.sub(key, value, raw)
                return raw
        else:
            return raw

    def clean_text(self, text, remove_stopwords = True):
        """
            Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings

            Paras:
                text: text data to clean
                remove_stopwords: if true, remove stop words from text to reduce noise
            Returns:
                text: cleaned text data
        """
        text = [self.remove_contractions(word) for word in sent_tokenize(text.lower())]
        text = " ".join(text)

        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'&amp;', '', text) 
        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'\'', ' ', text)
        text = re.sub(r'[^a-zA-Z]', " ", text)

        if remove_stopwords:
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)

        return text

    def save_data(self, directory, name, docs, mode = "w"):
        """
            Saves data to directory

            Paras:
                directory: directory to save data
                name: name of file
            Returns:
                None
        """
        self.makedirs(directory)
        with open(join(directory, name), mode, encoding = "utf-8") as file:
            file.write(docs)

    def filterReview(self, df, name, value):
        """
            Filter dataframe by returning rows with column criteria
            Paras:
                df: dataframe
                name: name of column
                value: value to filter based on
            Return:
                df: data frame
        """
        return df.loc[df[name] == value]

    def loadData(self, path, column_names):
        """
            Loads the csvs in prepation for processing
            Paras:
                path: path
                column_names: columns to merge data on
            Return:
                df: merged dataframes
        """
        files = [doc for doc in listdir(path) if doc.endswith(".csv")]
        dataframes = [self.pd_csv(join(path, sheet)) for sheet in files]
        if len(dataframes)>1: return self.merge_workbooks(dataframes, column_names)
        else: return dataframes[0]

    def saveResults(self, df):
        df.to_csv("TrainingResults.csv", encoding = 'utf-8', index = False)

    def performance(self, df):
        tp = len(self.filterReview(df, "status", "correct"))
        fp = len(self.filterReview(df, "status", "incorrect"))
        precision = round(float(tp/(tp+fp)), 2)*100
        accuracy = round(float(tp/len(df)), 2)*100
        print("Precision: ", precision)
        print("Accuracy: ", accuracy)