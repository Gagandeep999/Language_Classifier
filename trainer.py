from classifier import Classifier
from evaluation import Evaluation
import time


def main():
    vocab = input('Enter choice for vocabulary: ')
    # vocab = '0'
    ngram = input('Enter choice for NGram: ')
    # ngram = '1'
    delta = input('Enter smoothing delta value between 0 and 1: ')
    # delta = '0.5'
    training_file = 'OriginalDataSet/training-tweets.txt'
    # training_file = input('Enter training file: ')
    testing_file = 'OriginalDataSet/test-tweets-given.txt'
    # testing_file = input('Enter test file: ')

    classifier = Classifier(vocab, ngram, float(delta), training_file, testing_file)
    start = time.time()
    classifier.read_data(False)
    print('Time taken to read: ', time.time() - start)
    start = time.time()
    classifier.create_model()
    print('Time taken to create model: ', time.time() - start)
    start = time.time()
    classifier.train_model()
    print('Time taken to train model: ', time.time() - start)
    # classifier.save_model()
    start = time.time()
    trace_file = classifier.test_model()
    print('Time taken to test model: ', time.time() - start)
    evaluation = Evaluation(trace_file)
    start = time.time()
    evaluation.calculate_performance()
    evaluation.print_to_file()
    print('Time taken to evaluate model: ', time.time() - start)


if __name__ == '__main__':
    start = time.time()
    main()
    print('Time taken: ', time.time() - start)
