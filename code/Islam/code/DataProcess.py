import io
import re
import pandas as pd

from os.path import join
from os import listdir, makedirs
from pathos.multiprocessing import ProcessingPool as Pool

from utilities import Utils
from pprint import pprint

class DataProcessing():

    def __init__(self):
        """
            Initalizes DataProcessing class with utilities and parallel processing
            Paras:
                None
            Returns:
                None
        """
        self.utls = Utils()
        self.pool = Pool()
    
    def getSummaries(self, summaries):
        """
            Retuns list of cleaned summaries from dataframe list column
            Paras:
                summaries: list of summaries
            Returns:
                summaries: list of cleaned summaries
        """
        return [re.sub(r"[^a-zA-Z]", " ", summaries[i].lower()) for i in range(len(summaries))]

    def getReview(self, review):
        """
            Retuns list of cleaned revies from dataframe list column
            Paras:
                review: list of reviews
            Returns:
                review: list of cleaned reviews
        """
        review = [re.sub(r"[^a-zA-Z]", " ", review[i].lower()) for i in range(len(review))]
        return list(self.pool.map(self.utls.clean_text, review))

    def createDataframe(self, reviews, rating):
        """
            Creates dataframe class of cleaned concatenated reviews & summaries, and ratings.
            Paras:
                reviews: cleaned concated reviews
            Returns:
                ratings: ratings of the reviews
        """
        return pd.DataFrame({"reviews": reviews, "ratings": rating})

    def ProcessData(self, column_names = ["summary", "reviewText", "overall"]):
        """
            Runs DataProcessing class and creates dataframe of cleaned reviews and associated rating labels
            Paras:
                None
            Returns:
                None
        """
        dataframe = self.utls.loadData("./Dataset/raw_training", column_names)
        summaries = self.getSummaries(list(dataframe["summary"]))
        review = self.getReview(list(dataframe["reviewText"]))
        reviews = self.utls.concate_columns(summaries, review)
        rating = list(dataframe["overall"])
        return self.createDataframe(reviews, rating)

if __name__ == "__main__":
    Preprocessing = DataProcessing()
    dataframe = Preprocessing.ProcessData()
