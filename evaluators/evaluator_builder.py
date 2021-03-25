from evaluators.adaptrdoid_evaluator import AdaptdroidEvaluator
from evaluators.atm_evaluator import ATMEvaluator
from evaluators.craftdroid_evaluator import CraftdroidEvaluator
from evaluators.abstract_evaluator import AbstractEvaluator
from evaluators.custom_evaluator import CustomEvaluator
from evaluators.random_evaluator import RandomEvaluator


def extract_threshold(algorithm: str):
    number = int(algorithm.split('_')[1])
    return number / 100


class EvaluatorBuilder:
    def __init__(self):
        self._semantic_config = {'word_embedding': None, 'training_set': None, 'algorithm': None,
                                 'descriptors': None}

    def build(self) -> AbstractEvaluator:
        if not self.all_set():
            raise Exception('All four elements of configuration should be set')
        evaluator = self.find_evaluator(self.events_list)
        evaluator.set_models(self._semantic_config['word_embedding'], self._semantic_config['training_set'])
        evaluator.process_descriptors()
        return evaluator

    def find_evaluator(self, events_list):
        algorithm = self._semantic_config['algorithm']
        if algorithm.startswith('adaptdroid'):
            evaluator = AdaptdroidEvaluator(events_list, self._semantic_config['descriptors'])
            evaluator.set_threshold(extract_threshold(algorithm))
        elif algorithm == 'craftdroid':
            evaluator = CraftdroidEvaluator(events_list, self._semantic_config['descriptors'])
        elif algorithm.startswith('atm'):
            evaluator = ATMEvaluator(events_list, self._semantic_config['descriptors'])
            evaluator.set_threshold(extract_threshold(algorithm))
        elif algorithm == 'custom':
            evaluator = CustomEvaluator(events_list, self._semantic_config['descriptors'])
        elif algorithm == 'random':
            evaluator = RandomEvaluator(events_list, self._semantic_config['descriptors'])
        else:
            raise Exception('Algorithm ' + self._semantic_config['algorithm'] + ' do not exist')
        return evaluator

    def set_word_embedding(self, embedding):
        self._semantic_config['word_embedding'] = embedding

    def set_algorithm(self, algorithm):
        self._semantic_config['algorithm'] = algorithm

    def set_train_set(self, train_set):
        self._semantic_config['training_set'] = train_set

    def set_descriptors(self, descriptors):
        self._semantic_config['descriptors'] = descriptors

    def all_set(self):
        if self._semantic_config['word_embedding'] is None:
            return False
        if self._semantic_config['algorithm'] is None:
            return False
        if self._semantic_config['training_set'] is None:
            return False
        if self._semantic_config['descriptors'] is None:
            return False
        return True

    def set_semantic_config(self, semantic_config):
        self._semantic_config = semantic_config

    def set_events_list(self, events_list):
        self.events_list = events_list
