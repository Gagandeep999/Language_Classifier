import pandas as pd


class Evaluation:
    """
    To calculate the different metrics of the model
    """
    def __init__(self, file):
        self.data = file
        self.wrong_count = 0
        self.correct_count = 0
        self.macro = 0
        self.weight = 0
        self.languages = ['eu', 'ca', 'gl', 'es', 'en', 'pt']
        for language in self.languages:
            exec('self.%sTP = 0' % language)
            exec('self.%sFP = 0' % language)
            exec('self.%sTN = 0' % language)
            exec('self.%sFN = 0' % language)

    def calculate_performance(self):
        print('in eval reading from: ', self.data)
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
        print('in eval writing to: ', eval_filename)
        file = open(eval_filename, 'w')
        print('%.4f' % self.accuracy(), file=file, end='\n')
        for language in self.languages:
            exec('pre = self.precision(self.%sTP, self.%sFP)' % (language, language))
            exec('print(\'%.4f\' % pre, end=\'  \', file=file)')
        print(file=file)
        for language in self.languages:
            exec('rec = self.recall(self.%sTP, self.%sFN)' % (language, language))
            exec('print(\'%.4f\' % rec, end=\'  \', file=file)')
        print(file=file)
        for language in self.languages:
            exec('f1 = self.f1measure(self.%sTP, self.%sFN, self.%sFP)' % (language, language, language))
            exec('print(\'%.4f\' % f1, end=\'  \', file=file)')
        print(file=file)
        for language in self.languages:
            exec('macro = self.macroF1(self.%sTP, self.%sFN, self.%sFP)' % (language, language, language))
            exec('self.macro += macro')
        self.macro /= self.languages.__len__()
        print('%.4f' % self.macro, file=file, end='  ')
        for language in self.languages:
            exec('weight = self.weightedF1(self.%sTP, self.%sFN, self.%sFP)' % (language, language, language))
            exec('self.weight += weight')
        print('%.4f' % self.weight, file=file)
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
        if precision+recall == 0:
            return 0
        return 2 * ((precision*recall)/(precision+recall))

    def macroF1(self, tp, fn, fp):
        precision = self.precision(tp=tp, fp=fp)
        recall = self.recall(tp=tp, fn=fn)
        if precision + recall == 0:
            return 0
        return 2 * ((precision * recall) / (precision + recall))

    def weightedF1(self, tp, fn, fp):
        precision = self.precision(tp=tp, fp=fp)
        recall = self.recall(tp=tp, fn=fn)
        if precision + recall == 0:
            return 0
        return (tp+fp)/(self.wrong_count+self.correct_count) * ( 2 * ((precision * recall) / (precision + recall)) )
