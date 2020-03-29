class hyperparam:
    """
    Class to define the hyperparameters of the model.
    to call any value, syntax would be as follows
    $ vocab = hyperparam.vocab
    """
    def __init__(self, vocab, nGram, smoothingDelta, trainingFile, testingFile):
        self.vocav = vocab
        self.nGram = nGram
        self.smoothingDelta = smoothingDelta
        self.trainingFile = trainingFile
        self.testingFile = testingFile
