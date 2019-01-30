from os import system

from utilities import Utils

class Classify():

    def __init__(self):
        """
            Initalizes Classify class with utilities
            Paras:
                None
            Returns:
                None
        """
        self.utls = Utils()

    def setParameters(self, lr = None, lrUpdateRate = None, dim = None, ws = None, epoch = None, neg = None, loss = None, thread = None, saveOutput = None):
        """
            Sets parameters to train NN

            Paras:
                -lr                 learning rate [0.05]
                -lrUpdateRate       change the rate of updates for the learning rate [100]
                -dim                size of word vectors [100]
                -ws                 size of the context window [5]
                -epoch              number of epochs [5]
                -neg                number of negatives sampled [5]
                -loss               loss function {ns, hs, softmax} [ns]
                -thread             number of threads [12]
                -pretrainedVectors  pretrained word vectors for supervised learning []
                -saveOutput         whether output params should be saved [0]
            Returns:
                training parameters
        """
        if lr == None: lr = " "
        else: lr = "-lr %s " %lr
        
        if lrUpdateRate == None: lrUpdateRate = " "
        else: lrUpdateRate = "-lrUpdateRate %s " %lrUpdateRate
        
        if dim == None: dim = " "
        else: dim = "-dim %s " %dim
        
        if ws == None: ws = " "
        else: ws = "-ws %s " %ws
        
        if epoch == None: epoch = " "
        else: epoch = "-epoch %s " %epoch
        
        if neg == None: neg = " "
        else: neg = "-neg %s " %neg
        
        if loss == None: loss = " "
        else: loss = "-loss %s " %loss
        
        if thread == None: thread = " "
        else: thread = "-thread %s " %thread
        
        if saveOutput == None: saveOutput = " "
        else: saveOutput = "-saveOutput %s " %saveOutput
        
        return lr + lrUpdateRate + dim + ws + epoch + neg + loss + thread + saveOutput

    def trainClassifier(self, hyper_parameters):
        """
            Trains supervised classifier
            Paras:
                hyper_parameters: parameters to train neural net
            Returns:
                None
        """
        self.utls.makedirs("./fastTextModels")
        system("./fastText/fasttext supervised -input ./Dataset/training_processed/training.txt -output ./fastTextModels/model_1 -label __label__ {}").format(hyper_parameters)
       
if __name__ == "__main__":
    classify = Classify()
    hyper_parameters = classify.setParameters(lr = 0.3, epoch = 5, dim = 300)
    classify.trainClassifier(hyper_parameters)