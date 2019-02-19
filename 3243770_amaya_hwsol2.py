"""
    Implementation of n-gram approach (from n = 1 to n = 5) for feature representation.
    Using n-gram represenations, build and test a binary classifier (sentiments/binary labels).
    Then study the impact of n on the performance of the classifier.
"""

import sys
import string
import re
from sklearn import svm

# NOTE: This program assumes that a file parameter will be given from which it will read the input dataset.

n_gram_length = 5
feature_vectors = {key: set() for key in range(1, n_gram_length + 1)}

# List of sentences with their respective label, n_grams
sentences_with_label = []

# Start by reading in the file name and perform text pre-processing
with open(sys.argv[1], 'r') as dataset:
    rows = dataset.readlines()
    for row in rows:
        [words, label] = row.lower().replace('\'', '').split('\t')    
        # remove punctuations from words then leftover empty spaces
        word_array = re.sub('[ ]+', ' ', re.sub('['+string.punctuation+']', ' ', words)).split()
        # # Create the unique n-grams (feature vectors)
        review_n_grams = {key: {} for key in range(1, n_gram_length + 1)}
        for n in range(1, n_gram_length + 1):
            for i in range(len(word_array) - n + 1):
                fv = ' '.join(word_array[i:i+n])
                feature_vectors[n].add(fv)
                review_n_grams[n][fv] = True
        sentences_with_label.append((' '.join(word_array), int(label), review_n_grams))
    
# Create alphabetically sorted feature_vectors
sorted_feature_vectors = {key: [] for key in range(1, n_gram_length + 1)}
for key in range(1, n_gram_length + 1):
    sorted_feature_vectors[key] = list(feature_vectors[key])
    sorted_feature_vectors[key].sort()

# Get the frequency of every n_gram per review per sorted_feature_vector
n_grams_matrix = {key: [] for key in range(1, n_gram_length + 1)}
for key in range(1, n_gram_length + 1):
    for sentence, _, review_n_grams in sentences_with_label:
        frequency_index = []
        for n_gram in sorted_feature_vectors[key]:
            # check if the n_gram is in that review, if not the frequency is automatically 0
            # this is to save computation time
            if review_n_grams[key].get(n_gram):
                frequency_index.append(len(re.findall(r'\b%s\b' %n_gram, sentence)))
            else:
                frequency_index.append(0)
        n_grams_matrix[key].append(frequency_index)

# Get the list of sentences with negative (0) and positive (1) labels for each feature
n_grams_matrix_by_label = {key: ([], []) for key in range(1, n_gram_length + 1)}
for key in range(1, n_gram_length + 1):
    for index, (_, label, _) in enumerate(sentences_with_label):
        if label is 0:
            n_grams_matrix_by_label[key][0].append(n_grams_matrix[key][index])
        elif label is 1:
            n_grams_matrix_by_label[key][1].append(n_grams_matrix[key][index])

# Combine feature vectors
def make_matrix_for_1_to(end, max_length):
    result = ([], [])
    for key in range(1, end + 1):
        negatives, positives = n_grams_matrix_by_label[key]
        # Pad with zeroes to match the largest feature vector length
        for index, x in [(0, negatives), (1, positives)]:
            for y in x:
                [y.append(0) for x in range(len(y), max_length)]
                result[index].append(y)
    return result

# Initialize the SVM classifier
binary_classifier = svm.LinearSVC()

# Train and test the classifier using 10-fold cross validation
def train_and_test(negatives, positives):

    # Initialize the metric values
    false_positive_rate, false_negative_rate, true_positive_rate, true_negative_rate, accuracy, precision, recall = (0.0 for _ in range(7))

    # Array to hold the metrics for each part of the 10-fold cross validation
    fpr, fnr, tpr, tnr, acc, prec, rec = ([] for _ in range(7))

    for start in range(10):

        training_data = ([], [])
        testing_data = ([], [])

        # Seperate the data into 90% training and 10% testing, ensuring that the training includes 90% of the positive labels and 90% of the negative labels
        # negatives, positives = n_grams_matrix_by_label[1]
        for label, vector in [(0, negatives), (1, positives)]:
            for index, item in enumerate(vector):
                if (index - start) % 10:
                    training_data[0].append(item)
                    training_data[1].append(label)
                else:
                    testing_data[0].append(item)
                    testing_data[1].append(label)  

        # train the model with the training data
        binary_classifier.fit(training_data[0], training_data[1])

        # test the model and calculate 7 metrics: 
        # false positive rate, false negative rate, true positive rate, true negative rate, accuracy, precision and recall
        # All formulas from: https://en.wikipedia.org/wiki/Sensitivity_and_specificity
        predicted_labels = binary_classifier.predict(testing_data[0] or [[]])

        # Get basic numbers used for metric calculations
        condition_positives = len(positives)
        condition_negatives = len(negatives)
        true_positives = true_negatives = false_positives = false_negatives = 0

        for (predicted_label, actual_label) in zip(predicted_labels, testing_data[1]):
            if predicted_label == actual_label:
                if predicted_label == 1:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if predicted_label == 1:
                    false_positives += 1
                else:
                    false_negatives += 1
        
        # Calculate false positive rate
        fpr.append(false_positives / condition_negatives if condition_negatives != 0 else 1)

        # Calculate false negative rate
        fnr.append(false_negatives / condition_positives if condition_positives != 0 else 1)

        # Calculate true positive rate
        tpr.append(true_positives / condition_positives if condition_positives != 0 else 1)

        # Calculate true negative rate
        tnr.append(true_negatives / condition_negatives if condition_negatives != 0 else 1)

        # Calculate accuracy
        acc.append((true_positives + true_negatives) / (condition_positives + condition_negatives) if (condition_positives + condition_negatives) != 0 else 1)

        # Calculate precision
        prec.append(true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 1)

        # Calculate recall
        rec.append(true_positives / condition_positives if condition_positives != 0 else 1)

    # Get the average metric value for the feature
    false_positive_rate = sum(fpr) / len(fpr)
    false_negative_rate = sum(fnr) / len(fnr)
    true_positive_rate = sum(tpr) / len(tpr)
    true_negative_rate = sum(tnr) / len(tnr)
    accuracy = sum(acc) / len(acc)
    precision = sum(prec) / len(prec)
    recall = sum(rec) / len(rec)

    print("FPR:{0:.5f}".format(false_positive_rate))
    print("FNR:{0:.5f}".format(false_negative_rate))
    print("TPR:{0:.5f}".format(true_positive_rate))
    print("TNR:{0:.5f}".format(true_negative_rate))
    print("Accuracy:{0:.5f}".format(accuracy))
    print("Precision:{0:.5f}".format(precision))
    print("Recall:{0:.5f}".format(recall))

# Test each feature_vector seperated (f1, f2, f3, f4, f5)
for key in range(1, n_gram_length + 1):
    print("Testing feature {}".format(key))
    train_and_test(*n_grams_matrix_by_label[key])
    print('')

# Test the feature_vectors combined (f1f2, f1f2f3, f1f2f3f4, f1f2f3f4f5)
max_len = max(len(sorted_feature_vectors[x]) for x in range(1, n_gram_length + 1))
for key in range(2, n_gram_length + 1):
    print("Testing features", 1, '-', key)
    train_and_test(*make_matrix_for_1_to(key, max_len))
    print('')