# Saraswathi Shanmugamoorthy
# CS 6375.003
# Assignment 3 - Perceptron and Neural Networks

import collections
import re
import copy
import os
import sys



class Docs:
    text = ""
    frequency_of_words = {}

    classes_true = ""
    classes_learn = ""

    def __init__(self, text, Counter, classes_true):
        self.text = text
        self.frequency_of_words = Counter
        self.classes_true = classes_true

    def getText(self):
        return self.text

    def getFrequencyWords(self):
        return self.frequency_of_words

    def getClassesTrue(self):
        return self.classes_true

    def getClassesLearned(self):
        return self.classes_learn

    def setLearnedClass(self, predicts):
        self.classes_learn = predicts



# to count the frequencies of the words present in the text files
def wordsCollection(text):
    wordsList = collections.Counter(re.findall(r'\w+', text))
    return dict(wordsList)


# Read all text files in given directory and construct the data set
# the directory path should just be like "train/ham" for example
# storage is the dictionary to store the email in
# True class is the true classification of the email (spam or ham)
def dataSetsMaking(store_dict, directory, classes_true):
    for directory_entry in os.listdir(directory):
        directory_entry_path = os.path.join(directory, directory_entry)
        if os.path.isfile(directory_entry_path):
            with open(directory_entry_path, 'r') as text_file:
                text = text_file.read()
                store_dict.update({directory_entry_path: Docs(text, wordsCollection(text), classes_true)})



def settingStopWords(text_file_for_stop_words):
    stop = []
    with open(text_file_for_stop_words, 'r') as txt:
        stop = (txt.read().splitlines())
    return stop


def vocabWords(data_sets):
    v = []
    for i in data_sets:
        for j in data_sets[i].getFrequencyWords():
            if j not in v:
                v.append(j)
    return v

def stopWordsRemoving(stop, data_sets):
    data_sets_filtered = copy.deepcopy(data_sets)
    for i in stop:
        for j in data_sets_filtered:
            if i in data_sets_filtered[j].getFrequencyWords():
                del data_sets_filtered[j].getFrequencyWords()[i]
    return data_sets_filtered

# For learning weights using perceptron training rule
def weightsLearning(weights, const_for_learning, train_set, iterations_num, classes):
    for i in iterations_num:
        
        for d in train_set:
            weights_sum = weights['weight_zero']
            for f in train_set[d].getFrequencyWords():
                if f not in weights:
                    weights[f] = 0.0
                weights_sum += weights[f] * train_set[d].getFrequencyWords()[f]
            output_perceptron = 0.0
            if weights_sum > 0:
                output_perceptron = 1.0
            values_target = 0.0
            if train_set[d].getClassesTrue() == classes[1]:
                values_target = 1.0
           
            for w in train_set[d].getFrequencyWords():
                weights[w] += float(const_for_learning) * float((values_target - output_perceptron)) * \
                              float(train_set[d].getFrequencyWords()[w])

# to test the accuracy of the algorith on the test test. And also to give the perceptron output

def applying_perceptron(weights, classes, instance):
    weights_sum = weights['weight_zero']
    for i in instance.getFrequencyWords():
        if i not in weights:
            weights[i] = 0.0
        weights_sum += weights[i] * instance.getFrequencyWords()[i]
    if weights_sum > 0:
        # spam
        return 1
    else:
        # ham
        return 0

#to read the training and test data
def main(training_dir, testing_dir, iteration, const_for_learning):
    # to create dictionaries and other lists
    train_set = {}
    test_set = {}
    train_set_filtered = {}
    testing_set_filtered = {}

    # list of stop words
    stop_words = settingStopWords('stop_words.txt')

    classes = ["ham", "spam"]

    # Number of iteration and learning constant (usually around .1 or .01)
    iteration = iteration
    const_for_learning = const_for_learning

    # To make the datasets
    dataSetsMaking(train_set, training_dir + "/spam", classes[1])
    dataSetsMaking(train_set, training_dir + "/ham", classes[0])
    dataSetsMaking(test_set, testing_dir + "/spam", classes[1])
    dataSetsMaking(test_set, testing_dir + "/ham", classes[0])

    train_set_filtered = stopWordsRemoving(stop_words, train_set)
    testing_set_filtered = stopWordsRemoving(stop_words, test_set)

    training_set_vocab = vocabWords(train_set)
    filtered_training_set_vocab = vocabWords(train_set_filtered)

    # to store the weights in dictionary and initialize as 1.0
    weights = {'weight_zero': 1.0}
    filtered_weights = {'weight_zero': 1.0}
    for i in training_set_vocab:
        weights[i] = 0.0
    for i in filtered_training_set_vocab:
        filtered_weights[i] = 0.0

    # For learning the weights
    weightsLearning(weights, const_for_learning, train_set, iteration, classes)
    weightsLearning(filtered_weights, const_for_learning, train_set_filtered, iteration, classes)

    #applying_perceptron algorithm to the test set and then output accuracy
    correct_predicts_count = 0
    for i in test_set:
        predicts = applying_perceptron(weights, classes, test_set[i])
        if predicts == 1:
            test_set[i].setLearnedClass(classes[1])
            if test_set[i].getClassesTrue() == test_set[i].getClassesLearned():
                correct_predicts_count += 1
        if predicts == 0:
            test_set[i].setLearnedClass(classes[0])
            if test_set[i].getClassesTrue() == test_set[i].getClassesLearned():
                correct_predicts_count += 1

    # applying_perceptron algorithm after filtering to the test set and then output accuracy
    filt_correct_predicts = 0
    for i in testing_set_filtered:
        predicts = applying_perceptron(filtered_weights, classes, testing_set_filtered[i])
        if predicts == 1:
            testing_set_filtered[i].setLearnedClass(classes[1])
            if testing_set_filtered[i].getClassesTrue() == testing_set_filtered[i].getClassesLearned():
                filt_correct_predicts += 1
        if predicts == 0:
            testing_set_filtered[i].setLearnedClass(classes[0])
            if testing_set_filtered[i].getClassesTrue() == testing_set_filtered[i].getClassesLearned():
                filt_correct_predicts += 1

    print "****************************************************"
    print " "
    print "Learning constant: %.4f" % float(const_for_learning)
    print "Number of iteration: %d" % int(iteration)

    print " "
    print "Correct Predictions before filtering out stop words: %d/%d" % (correct_predicts_count, len(test_set))
    print "Accuracy before filtering out stop words: %.4f%%" % (float(correct_predicts_count) / float(len(test_set)) * 100.0)

    print " "
    print "Correct predictions after filtering out stop words: %d/%d" % (filt_correct_predicts, len(testing_set_filtered))
    print "Accuracy after filtering out stop words: %.4f%%" % (float(filt_correct_predicts) / float(len(testing_set_filtered)) * 100.0)
    print " "
    print "*******************************************************"

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
