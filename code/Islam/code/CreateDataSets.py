import re
import pandas as pd

from os.path import join

from nltk.tokenize import word_tokenize
from utilities import Utils
from DataProcess import DataProcessing

class DataSets():

	def __init__(self):
		"""
			Initalizes DataSets class with utilities and preprocessing
			Paras:
				None
			Returns:
				None
		"""
		self.utls = Utils()
		self.preprocess = DataProcessing()

	def createTrainingCorpus(self, df):
		"""
			Creates training data set with labels appended to beginging of each label

			Paras:
				df: datafframe
			Returns:
				None
		"""
		length = []
		for i, doc in enumerate(df["reviews"]):
			doc =  "__label__%i" %(df["ratings"][i]) + " " + doc.rstrip()
			doc = " ".join([word for word in word_tokenize(doc) if len(word)>1])
			length.append(doc)
			self.utls.save_data("./Dataset/training_processed", "training.txt", doc + "\n", mode = "a")

	def createSet(self):
		"""
			Runs DataSets class and creates training set

			Paras: 
				None
			Returns:
				None
		"""	
		df = self.preprocess.ProcessData()
		self.createTrainingCorpus(df)
		
if __name__ == "__main__":
	data = DataSets()
	data.createSet()