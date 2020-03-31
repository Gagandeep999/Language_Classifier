from classifier import Classifier


def main():
    vocab = input('Enter choice for vocabulary: ')
    ngram = input('Enter choice for NGram: ')
    delta = input('Enter smoothing delta value between 0 and 1: ')
    # training_file = 'OriginalDataSet/training-tweets.txt'
    training_file = input('Enter training file: ')
    # testing_file = 'OriginalDataSet/test-tweets-given.txt'
    testing_file = input('Enter test file: ')

    classifier = Classifier(vocab, ngram, delta, training_file, testing_file)
    classifier.read_data()
    classifier.create_model()
    classifier.train_model()
    classifier.test_model()


if __name__ == '__main__':
    main()
