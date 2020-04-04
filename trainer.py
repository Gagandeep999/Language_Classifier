from classifier import Classifier
from evaluation import Evaluation


def main():
    vocab = input('Enter choice for vocabulary: ')
    ngram = input('Enter choice for NGram: ')
    delta = input('Enter smoothing delta value between 0 and 1: ')
    training_file = 'OriginalDataSet/training-tweets.txt'
    # training_file = input('Enter training file: ')
    testing_file = 'OriginalDataSet/test-tweets.txt'
    # testing_file = input('Enter test file: ')

    classifier = Classifier(vocab, ngram, float(delta), training_file, testing_file)
    classifier.read_data(False)
    classifier.create_model()
    classifier.train_model()
    trace_file = classifier.test_model()
    evaluation = Evaluation(trace_file)
    evaluation.calculate_performance()
    evaluation.print_to_file()


if __name__ == '__main__':
    main()
