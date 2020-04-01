class Classifier:
    """
    This class trains a model based on the paramteres.
    """
    def __init__(self, vocab, ngram, delta, train, test):
        """
        constructor for the trainer class
        """
        self.vocab = vocab
        self.ngram = ngram
        self.delta = delta
        self.training_file = train
        self.testing_file = test
        self.model = ()  # list to contain 6 models, one for each language
        self.data = {}  # we will have the data stored as a dictionary of language:tweet pair

    def read_data(self):
        """
        This method read the data based on the vocab value provided by the user
        0 Fold the corpus to lowercase and use only the 26 letters of the alphabet [a-z]
        1 Distinguish up and low cases and use only the 26 letters of the alphabet [a-z, A-Z]
        2 Distinguish up and low cases and use all characters accepted by the built-in isalpha() method
        :return:
        """
        return

    def create_model(self):
        """
        Once the data is read, next step is to create the model. This method creates the model based
        on the value provided for ngram. For example, given the string "abc*def" (* is out of language)
        1 character unigrams creates 6 unigrams: a, b, c, d, e, f (1D array)
        2 character bigrams creates 4 bigrams: ab, bc, de, ef (2D array)
        3 character trigrams creates 2 trigrams: abc, def (3D array)
        The attribute self.model is a list which will contain 6 intances of the same array, one for each language.
        :return:
        """
        return

    def train_model(self):
        """
        Once the model is created based on the ngram value, we start training the model.
        To train a unigarm model we keep a counter for each individual character of the language and a counter
        for total number of character of the language, which will give the probability of a certain character in
        the language.
        To train a bigram model,
        :return:
        """
        return

    def save_model(self):
        """
        Save the model with parameter description to keep track of the model performance.
        :return:
        """
        return

    def test_model(self):
        """
        In this method, we need to run the test case through the model; we need to calculate the metrics
        for the model; output those metrics to a file.
        :return:
        """
        return
