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
        for language in self.languages:
            exec('print(self.precision(self.%sTP, self.%sFP), end=\'  \', file=file)' % (language, language))
        print(file=file)
        for language in self.languages:
            exec('print(self.recall(self.%sTP, self.%sFN), end=\'  \', file=file)' % (language, language))
        print(file=file)
        for language in self.languages:
            exec('print(self.f1measure(self.%sTP, self.%sFN, self.%sFP), end=\'  \', file=file)' % (language, language, language))
        print(file=file)
        macro = 0
        for language in self.languages:
            macro_each = self.macroF1(language)
            macro += macro_each
        print(macro / self.languages.__len__(), file=file, end='  ')
        weight = 0
        for language in self.languages:
            weight_each = self.weightedF1(language)
            weight += weight_each
        print(weight, file=file)
        file.flush()
        file.close()

    def accuracy(self):
        return self.correct_count/(self.wrong_count+self.correct_count)

    def precision(self, tp, fp):
        return tp / (tp+fp)

    def recall(self, tp, fn):
        return tp / (tp+fn)

    def f1measure(self, tp, fn, fp):
        precision = self.precision(tp=tp, fp=fp)
        recall = self.recall(tp=tp, fn=fn)
        return 2 * ((precision*recall)/(precision+recall))

    def macroF1(self, L):
        macro = 0
        exec('macro = self.f1measure(self.%sTP, self.%sFN, self.%sFP)' % (L, L, L))
        return macro

    def weightedF1(self, L):
        weight = 0
        exec('weight = (self.%sTP+self.%sFP)/(self.correct_count+self.wrong_count)' % (L, L))
        macro = self.macroF1(L)
        return macro * weight
