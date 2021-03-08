from random import random
from nltk import word_tokenize
import numpy as np


def matrix_overall_sim(matrix):
    sum_of_max = 0.0
    count = 0
    while not matrix_all_negative(matrix):
        indexes = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
        sum_of_max = sum_of_max + matrix[indexes[0]][indexes[1]]
        count += 1
        matrix[indexes[0], :] = -1
        matrix[:, indexes[1]] = -1
    return 0 if count == 0 else sum_of_max / count


def sim_matrix_of_tokens(a, b, sim_func):
    words_a = word_tokenize(a)
    words_b = word_tokenize(b)
    matrix = np.zeros((words_a.__len__(), words_b.__len__()))
    for i in range(words_a.__len__()):
        for j in range(words_b.__len__()):
            matrix[i][j] = sim_func(words_a[i], words_b[j])
    return matrix


def matrix_all_negative(matrix):
    return np.all(matrix == -1)


def sim_rand(word_a, word_b):
    return random()


if __name__ == "__main__":
    a = 'editor edittext title editor edittext title pocket note'
    b = 'edit note title title title edit note'
    sample_matrix = sim_matrix_of_tokens(a, b, sim_rand)
    sim = matrix_overall_sim(sample_matrix)
    print(sim)
