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
    for row in rows[0:10]:
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

# for key in range(1, n_gram_length + 1):
#     print('matrix for key', key, 'of length', len(sorted_feature_vectors[key]))
#     for frequency_index in n_grams_matrix[key]:
#         print(frequency_index)
#     print('\n\n')

# for index, (sentence, label, _) in enumerate(sentences_with_label):
#     print(sentence, label, n_grams_matrix[1][index])

# Get the list of sentences with postive (1) and negative (0) labels for each n_gram
n_grams_matrix_by_label = {key: ([], []) for key in range(1, n_gram_length + 1)}
for key in range(1, n_gram_length + 1):
    for index, (_, label, _) in enumerate(sentences_with_label):
        if label is 0:
            n_grams_matrix_by_label[key][0].append(n_grams_matrix[key][index])
        elif label is 1:
            n_grams_matrix_by_label[key][1].append(n_grams_matrix[key][index])

# Initialize the SVM classifier
binary_classifier = svm.LinearSVC()

# Fit the binary classifier using 1_grams
binary_classifier.fit(n_grams_matrix[1], [label for (_, label, _) in sentences_with_label])







