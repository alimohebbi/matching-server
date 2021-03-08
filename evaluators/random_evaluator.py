import random

import pandas as pd

from evaluators.abstract_evaluator import AbstractEvaluator


class RandomEvaluator(AbstractEvaluator):
    def get_potential_matches(self, descriptors_data):
        return descriptors_data

    def make_descriptors_compatible(self, row):
        pass

    def assign_score(self, descriptors):
        pass

    def run_algorithm(self, row):
        return random.random()

    def save(self, data, save_file_name):
        pass
