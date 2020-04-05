import pandas as pd


class Evaluation:
    """
    To calculate the different metrics of the model
    """
    def __init__(self, file):
        self.data = file
        self.wrong_count = 0
        self.correct_count = 0
        self.languages = ['eu', 'ca', 'gl', 'es', 'en', 'pt']
        for language in self.languages:
            exec('self.%sTP = 0' % language)
            exec('self.%sFP = 0' % language)
            exec('self.%sTN = 0' % language)
            exec('self.%sFN = 0' % language)

    def calculate_performance(self):
        df = pd.read_csv(self.data, delim_whitespace=True, header=0)
        for index, row in df.iterrows():
            predicted = row['PREDICTEDVALUE']
            actual = row['ACTUALVALUE']
            result = row['RESULT']
            if 'wrong' in result:
                self.wrong_count += 1
                exec('self.%sFN += 1' % actual)
                exec('self.%sFP += 1' % predicted)
            else:
                self.correct_count += 1
                exec('self.%sTP += 1' % predicted)

    def print_to_file(self):
        eval_filename = self.data.replace('trace', 'eval')
        file = open(eval_filename, 'w')
        print(self.accuracy(), file=file, end='\n')
        print(self.precision(), file=file, end='\n')
        print(self.recall(), file=file, end='\n')
        print(self.macroF1(), "  ", self.weightedF1(), file=file, end='\n')

    def accuracy(self):
        return self.correct_count/(self.wrong_count+self.correct_count)

    def precision(self):
        line = ""
        for language in self.languages:
            exec('line += str(self.%sTP / (self.%sTP + self.%sFP)) + \"  \" ' % (language, language, language))
        return line

    def recall(self):
        line = ""
        for language in self.languages:
            exec('line += str(self.%sTP / (self.%sTP + self.%sFN)) + \"  \" ' % (language, language, language))
        return line

    def f1measure(self):
        line = ""
        for language in self.languages:
            exec('precision = self.%sTP / (self.%sTP + self.%sFP) \n\
    recall = self.%sTP / (self.%sTP + self.%sFN) \n\
    line += str(2 * (precision * recall) / (precision + recall)) + \"  \" ' % (language, language, language, language, language, language))
        return line

    def macroF1(self):
        data = 0
        for language in self.languages:
            exec('precision = self.%sTP / (self.%sTP + self.%sFP) \n\
    recall = self.%sTP / (self.%sTP + self.%sFN) \n\
    data += 2 * (precision * recall) / (precision + recall) ' % (language, language, language, language, language, language))
        return data / 6

    def weightedF1(self):
        data = 0
        for language in self.languages:
            exec('precision = self.%sTP / (self.%sTP + self.%sFP) \n\
    recall = self.%sTP / (self.%sTP + self.%sFN) \n\
    data += ((self.%sTP + self.%sFP) / (self.wrong_count + self.right_count)) * (2 * (precision * recall) / (precision + recall))' % (language, language, language, language, language, language, language, language))
        return data
