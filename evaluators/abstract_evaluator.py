import logging
from abc import ABC, abstractmethod

import pandas as pd

from config import Config
from descriptor_processes.load_data import get_map
from descriptor_processes.text_pre_process import space_cleaner
from embedding.embedding_factory import SimpleEmbeddingFactory, ClustersEmbeddingFactory, EmbedTypes
from threshold.threshold_get import get_threshold

clustering_train_set = ['topics', 'category',
                        'hierarchy_category_edit',
                        'hierarchy_topics_edit',
                        'hierarchy_category_jaccard',
                        'hierarchy_topics_jaccard']


class AbstractEvaluator(ABC):
    config = Config()

    def __init__(self, events_list, evaluation_config):
        self.train_set = None
        self.events_list = events_list
        self.descriptors_type = evaluation_config['descriptors']
        self.correct_number = None
        self.potential_matches = None
        self.ranking_potential_matches = None
        self.active_technique = None
        self.technique = None
        self.threshold = evaluation_config['threshold']

    def process_descriptors(self):
        mapping = get_map(self.events_list, self.descriptors_type)
        descriptors_data = self.create_sim(mapping)
        self.potential_matches = self.get_potential_matches(descriptors_data)
        if self.config.threshold_use:
            self.potential_matches = self.apply_threshold(self.potential_matches)

    @abstractmethod
    def get_potential_matches(self, descriptors_data):
        pass

    def get_ranking_potential_matches(self, descriptors_data):
        return self.get_potential_matches(descriptors_data)

    def set_models(self, active_technique, train_set, source_app=None):
        self.train_set = train_set
        self.active_technique = active_technique
        if train_set in clustering_train_set and not source_app:
            logging.debug('Skip set model')
            pass
        elif train_set in clustering_train_set:
            self.technique = self.set_cluster_model(source_app)
            logging.debug('set model for clusters')
        else:
            self.technique = SimpleEmbeddingFactory \
                .get_instance() \
                .get_model(EmbedTypes[active_technique], train_set)
            logging.debug('set model for simple train set')

    def set_cluster_model(self, source_app):
        zeros = self.technique.zeros if self.technique else 0
        new_model = ClustersEmbeddingFactory \
            .get_instance() \
            .get_model(EmbedTypes[self.active_technique], self.train_set, source_app)
        if self.technique:
            new_model.zeros += zeros
        return new_model

    def add_sim_to_df(self, data):
        if self.technique is None and self.train_set not in clustering_train_set:
            raise Exception("A Model should have been set")
        data['score'] = data.apply(lambda row: self.run_algorithm(row), axis=1)

    def create_sim(self, data):
        self.add_sim_to_df(data)
        return data

    def run_algorithm(self, row):
        if self.train_set in clustering_train_set:
            self.set_models(self.active_technique, self.train_set, row['src_app'])
            logging.debug('set model for cluster train set inside run algorithm')
        descriptors = self.make_descriptors_compatible(row)
        descriptors = self.remove_white_space(descriptors)
        score = self.assign_score(descriptors)
        return score

    @abstractmethod
    def make_descriptors_compatible(self, row):
        pass

    @abstractmethod
    def assign_score(self, descriptors):
        pass

    @staticmethod
    def remove_white_space(descriptors):
        if type(descriptors) == pd.Series:
            return descriptors.map(lambda x: space_cleaner(x))
        for key, value in descriptors.__dict__.items():
            descriptors.__dict__[key] = space_cleaner(value)
        return descriptors

    def apply_threshold(self, potential_matches):
        return potential_matches[potential_matches['score'] >= self.threshold]
