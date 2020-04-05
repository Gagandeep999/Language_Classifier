import sys
import re
import math
import pandas as pd
import numpy as np
from decimal import Decimal
from collections import defaultdict


class Classifier:
    """
    This class trains a model based on the paramteres and prints the result of the test to an output file.
    """
    def __init__(self, vocab, ngram, delta, train, test):
        """
        constructor for the trainer class
        """
        self.vocab = vocab
        self.ngram = ngram
        self.delta = delta
        self.pattern = re
        self.training_file = train
        self.testing_file = test
        self.data = defaultdict(list)  # we will have the data stored as a dictionary of language:tweet pair
        self.languages = ['eu', 'ca', 'gl', 'es', 'en', 'pt']
        self.defaultSmoothing = 10e-10
        for language in self.languages:
            exec("self.%sAlphabets={}" % language)
            exec('self.%sSize = 0' % language)
            exec('self.%sModel = np.array([])' % language)

    def read_data(self, return_data):
        """
        This method read the data based on the vocab value provided by the user
        0 Fold the corpus to lowercase and use only the 26 letters of the alphabet [a-z]
        1 Distinguish up and low cases and use only the 26 letters of the alphabet [a-z, A-Z]
        2 Distinguish up and low cases and use all characters accepted by the built-in isalpha() method
        :return:
        """
        df = pd.read_csv(self.training_file, encoding='utf-8', error_bad_lines=False, sep='\t', warn_bad_lines=False) # nrows=5000,
        df.columns = ['TweetID', 'UserID', 'Language', "Tweet"]
        _df = df[['Language', 'Tweet']].copy()
        train_dict = defaultdict(list)
        if self.vocab == '0':
            self.pattern = re.compile('[a-z]')
            for index, row in _df.iterrows():
                sentence = ''
                tweet = row['Tweet']
                tweet = tweet.lower()
                language = row['Language']
                for letter in tweet:
                    if self.pattern.match(letter):
                        exec('if \'{let}\' not in self.{L}Alphabets.keys():\n\
                                 self.{L}Alphabets[letter] = self.{L}Size\n\
                                 self.{L}Size += 1'.format(let=letter, L=language))
                        sentence = sentence + letter
                    else:
                        sentence = sentence + ' '
                train_dict[row['Language']].append(sentence)
        elif self.vocab == '1':
            self.pattern = re.compile('[a-zA-Z]')
            for index, row in _df.iterrows():
                sentence = ''
                tweet = row['Tweet']
                language = row['Language']
                for letter in tweet:
                    if self.pattern.match(letter):
                        exec('if \'{let}\' not in self.{L}Alphabets.keys():\n\
                                             self.{L}Alphabets[letter] = self.{L}Size\n\
                                             self.{L}Size += 1'.format(let=letter, L=language))
                        sentence = sentence + letter
                    else:
                        sentence = sentence + ' '
                train_dict[row['Language']].append(sentence)
        elif self.vocab == '2':
            for index, row in _df.iterrows():
                sentence = ''
                tweet = row['Tweet']
                language = row['Language']
                for letter in tweet:
                    if letter.isalpha():
                        exec('if \'{let}\' not in self.{L}Alphabets.keys():\n\
                                             self.{L}Alphabets[letter] = self.{L}Size\n\
                                             self.{L}Size += 1'.format(let=letter, L=language))
                        sentence = sentence + letter
                    else:
                        sentence = sentence + ' '
                train_dict[row['Language']].append(sentence)
        else:
            print('Invalid input for vocab parameter.')
            sys.exit(1)

        if return_data:
            return train_dict
        else:
            self.data = train_dict

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
        if self.ngram == '1':
            for language in self.languages:
                exec("self.{L}Model = np.resize(self.{L}Model, (self.{L}Size+1))".format(L=language))
                exec('self.{L}Model = np.add(self.{L}Model, self.defaultSmoothing)'.format(L=language))
        elif self.ngram == '2':
            for language in self.languages:
                exec("self.{L}Model = np.resize(self.{L}Model, ((self.{L}Size+1),(self.{L}Size+1)))".format(L=language))
                exec('self.{L}Model = np.add(self.{L}Model, self.defaultSmoothing)'.format(L=language))
        elif self.ngram == '3':
            for language in self.languages:
                exec('self.{L}Model = np.resize(self.{L}Model, ((self.{L}Size+1),(self.{L}Size+1),(self.{L}Size+1)))'.format(L=language))
                exec('self.{L}Model = np.add(self.{L}Model, self.defaultSmoothing)'.format(L=language))
        else:
            print('Invalid input for ngram parameter.')
            sys.exit(1)

    def train_model(self):
        """
        Once the model is created based on the ngram value, we start training the model.
        To train a unigarm model we keep a counter for each individual character of the language and a counter
        for total number of character of the language, which will give the probability of a certain character in
        the language.
        To train a bigram model,
        :return:
        """
        for language, tweets in self.data.items():
            for tweet in tweets:
                if self.ngram == '1':
                    for i in range(len(tweet) - 1):
                        first = tweet[i] # get the first character
                        if not first.isspace():
                            exec('index = self.%sAlphabets[first]' % language)  # get index of the character from the language dictionary
                            exec('self.%sModel[index] += 1' % language)  # increment that index in the language model
                elif self.ngram == '2':
                    for i in range(len(tweet) - 2):
                        first = tweet[i]  # get first character
                        second = tweet[i + 1]  # get second character
                        if (not first.isspace()) and (not second.isspace()):
                            exec('firstIndex = self.%sAlphabets[first]' % language)  # get index of the character from the language dictionary
                            exec('secondIndex = self.%sAlphabets[second]' % language)  # get index of the character from the language dictionary
                            exec('self.%sModel[firstIndex][secondIndex] += 1' % language)  # increment that index in the language model
                else:
                    for i in range(len(tweet) - 2):
                        first = tweet[i]
                        second = tweet[i + 1]
                        third = tweet[i + 2]
                        if (not first.isspace()) and (not second.isspace()) and (not third.isspace()):
                            exec('firstIndex = self.%sAlphabets[first]' % language)
                            exec('secondIndex = self.%sAlphabets[second]' % language)
                            exec('thirdIndex = self.%sAlphabets[third]' % language)
                            exec('self.%sModel[firstIndex][secondIndex][thirdIndex] += 1' % language)

        for language in self.languages:
            if self.ngram == '1':
                exec('self.{L}Model = np.add(self.{L}Model, self.delta)'.format(L=language))  # this is where smoothing happens
                exec('self.{L}Model = np.divide(self.{L}Model, self.{L}Model.sum(axis=0))'.format(L=language))  # divide all the values by the sum of the row
                exec('self.{L}Model = np.log10(self.{L}Model)'.format(L=language))
            elif self.ngram == '2':
                exec('self.{L}Model = np.add(self.{L}Model, self.delta)'.format(L=language))  # this is where smoothing happens
                exec('self.{L}Model = np.divide(self.{L}Model, self.{L}Model.sum(axis=1))'.format(L=language))  # divide all the values by the sum of the row
                exec('self.{L}Model = np.log10(self.{L}Model)'.format(L=language))
            else:
                exec('for x in range(self.{L}Model.shape[0]):\n\
                    for y in range(self.{L}Model.shape[1]):\n\
                        {L}ModelTemp = self.{L}Model[x,y,:]\n\
                        {L}ModelTemp = np.add({L}ModelTemp, self.delta)\n\
                        {L}ModelTemp = np.divide({L}ModelTemp, {L}ModelTemp.sum(axis=0))\n\
                        {L}ModelTemp = np.log10({L}ModelTemp)\n\
                        self.{L}Model[x,y,:] = {L}ModelTemp\n'.format(L=language))

    def save_model(self):
        """
        Save the model with parameter description to keep track of the model performance.
        :return:
        """
        for language in self.languages:
            if self.ngram == '1':
                exec('np.savetxt(\'Models/{L}ModelUnigram.model\', self.{L}Model, delimiter=\',\', fmt=\'%1.2e\')'.format(L=language))
            elif self.ngram == '2':
                exec('np.savetxt(\'Models/{L}ModelBigram.model\', self.{L}Model, delimiter=\',\', fmt=\'%1.2e\')'.format(L=language))
            else:
                exec('outfile = open(\'Models/{L}ModelTrigram.model\', \'w\')\n\
    print(\'# Shape \', self.{L}Model.shape, file=outfile)\n\
    outfile.flush()\n\
    print(\'# To load model - new_data = np.loadtxt(filename)\', file=outfile)\n\
    outfile.flush()\n\
    print(\'# Reshape the data - new_data = new_data.reshape((shape))\', file=outfile)\n\
    outfile.flush()\n\
    for data_slice in self.{L}Model:\n\
        np.savetxt(outfile, data_slice, delimiter=\',\', fmt=\'%1.2e\')'.format(L=language))

    def test_model(self):
        """
        In this method, we need to run the test case through the model; we need to calculate the metrics
        for the model; output those metrics to a file.
        :return:
        """
        filename = 'Outputs/trace_%s_%s_%s.txt' % (self.vocab, self.ngram, str(self.delta))
        print('filename for trace is: ', filename)
        file = open(filename, 'w')
        print('TWEETID', '  ', 'PREDICTEDVALUE', '  ', 'PROBABILITY', '  ', 'ACTUALVALUE', 'RESULT', file=file, end='\n')
        df = pd.read_csv(self.testing_file, encoding='utf-8', error_bad_lines=False, sep='\t')
        df.columns = ['TweetID', 'UserID', 'Language', "Tweet"]
        _df = df[['TweetID', 'Language', 'Tweet']].copy()
        probability = {}
        for index, row in _df.iterrows():
            for language in self.languages:
                exec("%sProb=math.log10(1/6)" % language)
            tweetID = row['TweetID']
            langTweet = row['Language']
            tweet = row['Tweet']
            if self.ngram == '1':
                for i in range(len(tweet) - 2):
                    first = tweet[i]
                    for language in self.languages:
                        # add condition when the "first" does not match the pattern
                        exec('if not self.pattern.match(first):\n\
    prob = 0\n\
elif (first not in self.{lang}Alphabets.keys()):\n\
    prob = self.{lang}Model[-1]\n\
else:\n\
    index = self.{lang}Alphabets[first]\n\
    prob = self.{lang}Model[index]\n\
{lang}Prob = prob + {lang}Prob\n'.format(lang=language))
            elif self.ngram == '2':
                # add condition when the "first" & "second" does not match the pattern
                for i in range(len(tweet) - 2):
                    first = tweet[i]
                    second = tweet[i + 1]
                    for language in self.languages:
                        exec('if ( (not self.pattern.match(first)) and (not self.pattern.match(second)) ):\n\
    prob = 0\n\
elif ((first not in {lang}Alphabets.keys()) and (second not in {lang}Alphabets.keys())):\n\
    prob = {lang}Model[-1][-1]\n\
elif (second not in {lang}Alphabets.keys()):\n\
    index = {lang}Alphabets[first]\n\
    prob = {lang}Model[index][-1]\n\
elif (first not in {lang}Alphabets.keys()):\n\
    index = {lang}Alphabets[second]\n\
    prob = {lang}Model[-1][index]\n\
else:\n\
    firstIndex = {lang}Alphabets[first]\n\
    secondIndex = {lang}Alphabets[second]\n\
    prob = {lang}Model[firstIndex][secondIndex]\n\
{lang}Prob = prob + {lang}Prob\n'.format(lang=language))
            else:
                for i in range(len(tweet) - 2):
                    first = tweet[i]
                    second = tweet[i + 1]
                    third = tweet[i + 2]
                    for language in self.languages:
                        # add condition when the "first" & "second" & "third" does not match the pattern
                        exec('if ( (not first.isalpha()) and (not second.isalpha()) and (not third.isalpha()) ):\n\
    prob = 0 \n\
elif ((first not in {lang}Alphabets.keys()) and (second not in {lang}Alphabets.keys()) and (third not in {lang}Alphabets.keys())):\n\
    prob = {lang}Model[-1][-1][-1]\n\
elif ( (first not in {lang}Alphabets.keys()) and (second not in {lang}Alphabets.keys()) ):\n\
    index = {lang}Alphabets[third]\n\
    prob = {lang}Model[-1][-1][index]\n\
elif ( (first not in {lang}Alphabets.keys()) and (third not in {lang}Alphabets.keys()) ):\n\
    index = {lang}Alphabets[second]\n\
    prob = {lang}Model[-1][index][-1]\n\
elif ( (second not in {lang}Alphabets.keys()) and (third not in {lang}Alphabets.keys()) ):\n\
    index = {lang}Alphabets[first]\n\
    prob = {lang}Model[index][-1][-1]\n\
elif first not in {lang}Alphabets.keys():\n\
    secondIndex = {lang}Alphabets[second]\n\
    thirdIndex = {lang}Alphabets[third]\n\
    prob = {lang}Model[-1][secondIndex][thirdIndex]\n\
elif second not in {lang}Alphabets.keys():\n\
    firstIndex = {lang}Alphabets[first]\n\
    thirdIndex = {lang}Alphabets[third]\n\
    prob = {lang}Model[firstIndex][-1][thirdIndex]\n\
elif third not in {lang}Alphabets.keys():\n\
    firstIndex = {lang}Alphabets[first]\n\
    secondIndex = {lang}Alphabets[second]\n\
    prob = {lang}Model[firstIndex][secondIndex][-1]\n\
else:\n\
    firstIndex = {lang}Alphabets[first]\n\
    secondIndex = {lang}Alphabets[second]\n\
    thirdIndex = {lang}Alphabets[third]\n\
    prob = {lang}Model[firstIndex][secondIndex][thirdIndex]\n\
{lang}Prob = prob + {lang}Prob\n'.format(lang=language))

            for langu in self.languages:
                exec("probability['%s'] = %sProb" % (langu, langu))
            result = max(probability, key=probability.get)
            print(tweetID, '  ', result, '  ', '%.2E' % Decimal(probability[result]), '  ', langTweet, '  ',
                  'correct' if (langTweet == result) else 'wrong', file=file, end='\n')
        return filename
