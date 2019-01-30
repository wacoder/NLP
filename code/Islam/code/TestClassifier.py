import re
import pandas as pd

from os import remove
from subprocess import Popen, PIPE

from utilities import Utils

class Test():

    def __init__(self):
        """
            Initalizes Test class with utilities
            Paras:
                None
            Returns:
                None
        """
        self.utls = Utils()

    def testClassifier(self):
        """
            Classifies datapoint

            Paras:
                None
            Return:
                None
        """
        command = "./fastText/fasttext predict ./fastTextModels/model.bin ./temp_test.txt"
        pred = Popen(command, shell = True, stdout=PIPE).stdout
        pred = re.sub(r"b\'|\\n\'", "", str(pred.read()))
        return int(pred[-1])

    def constructDataframe(self):
        """
            Constructs dataframe with test resutls

            Paras:
                None
            Return:
                None
        """
        status = []
        predictedRating = []

        df = self.utls.pd_csv("./Dataset/testing/test_with_ratings.csv")
        df = self.utls.remove_columns(df, ["summary", "reviewText"])

        for i, temp in df.iterrows():
            data = temp.summary + ". " + temp.reviewText
            data = self.utls.clean_text(data)
            self.utls.save_data("./", "temp_test.txt", data)
            rating = self.testClassifier()
            predictedRating.append(rating)
            if i == rating: 
                status.append("correct")
            else: 
                status.append("incorrect")
        df = pd.DataFrame({"predictedRating":predictedRating, "status":status})
        remove("./temp_test.txt")
        return df      

    def runTest(self):
        """
            Run Test class, and creates dataframe with processed results.

            Paras:
                None
            Returns:
                None
        """
        df = self.constructDataframe()
        self.utls.saveResults(df)
        self.utls.performance(df)

if __name__ == "__main__":
    classify = Test()
    classify.runTest()