import pandas as pd


class Evaluation:
    """
    To calculate the different metrics of the model
    """
    def __init__(self, file):
        self.data = file
        self.wrong_count = 0
        self.correct_count = 0
        self.languages = ['eu', 'ca', 'gl', 'es', 'es', 'pt']
        for language in self.languages:
            exec('%sTP = 0' % language)
            exec('%sFP = 0' % language)
            exec('%sTN = 0' % language)
            exec('%sFN = 0' % language)

    def calculate_performance(self):
        df = pd.read_csv('../trace_0_1_0.5.txt', delim_whitespace=True, header=0)
        for index, row in df.iterrows():
            predicted = row['PREDICTEDVALUE']
            actual = row['ACTUALVALUE']
            result = row['RESULT']
            if 'wrong' in result:
                self.wrong_count += 1
                exec('%sFN += 1' % actual)
                exec('%sFP += 1' % predicted)
            else:
                self.correct_count += 1
                exec('%sFP += 1' % predicted)

    def print_to_file(self):
        eval_filename = self.data.replace('trace', 'eval')
        file = open(eval_filename, 'w')
        print(self.accuracy(), file=file, end='\n')
        # print(precision(euData), '  ', ... precision(ptData))
        # print(recall(euData), '  ', ... recall(ptData))
        # print(f1measure(euData) .......... f1measure(ptData))
        # print(macroF1(), '  ', microF1())

    def accuracy(self):
        return self.correct_count/(self.wrong_count+self.correct_count)

    def precision(self):
        return

    def recall(self):
        return

    def f1measure(self):
        return

    def macroF1(self):
        return

    def microF1(self):
        return
