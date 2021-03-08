import logging
import os
import time
from abc import ABC, abstractmethod

import pandas as pd
from pandas import read_csv

from config import Config
from descriptor_processes.load_data import get_map
from descriptor_processes.text_pre_process import space_cleaner
from embedding.embedding_factory import SimpleEmbeddingFactory, ClustersEmbeddingFactory, EmbedTypes

clustering_train_set = ['topics', 'category',
                        'hierarchy_category_edit',
                        'hierarchy_topics_edit',
                        'hierarchy_category_jaccard',
                        'hierarchy_topics_jaccard']


class AbstractEvaluator(ABC):
    config = Config()

    def __init__(self, save_path, map_type):
        self.train_set = None
        self.save_path = save_path
        self.descriptors_type = map_type
        self.correct_number = None
        self.potential_matches = None
        self.ranking_potential_matches = None
        self.active_technique = None
        self.technique = None

    def process_descriptors(self):
        descriptors_data = self.load_descriptors(self.save_path, self.descriptors_type)
        self.correct_number = descriptors_data[descriptors_data['target_label'] == 'correct'].shape[0]
        self.potential_matches = self.get_potential_matches(descriptors_data)
        self.ranking_potential_matches = self.get_ranking_potential_matches(descriptors_data)

    @abstractmethod
    def get_potential_matches(self, descriptors_data):
        pass

    def get_ranking_potential_matches(self, descriptors_data):
        return self.get_potential_matches(descriptors_data)

    def set_models(self, active_technique, train_set, source_app=None):
        self.train_set = train_set
        self.active_technique = active_technique
        if self.map_exit() or (train_set in clustering_train_set and not source_app):
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
        data[self.active_technique] = data.apply(lambda row: self.run_algorithm(row), axis=1)

    def save(self, data, save_file_name):
        data.to_csv(self.config.save_score_dir + save_file_name, index=False)

    def create_sim(self, data, save_file_name='sim_score.csv'):
        self.add_sim_to_df(data)
        self.save(data, save_file_name)
        return data

    def calc_tp(self, matches):
        tp = matches[self.active_technique].loc[matches[self.active_technique]['target_label'] == 'correct'].shape[0]
        return tp

    def calc_fp(self, matches):
        fp = matches[self.active_technique].loc[matches[self.active_technique]['target_label'] == 'wrong'].shape[0]
        return fp

    def calc_metrics(self, matches, correct_number):
        tp = self.calc_tp(matches)
        precision = tp / (matches[self.active_technique].shape[0] + .000001)
        recall = tp / correct_number
        f = (2 * precision * recall) / (precision + recall + .000001)
        return {'precision': precision, 'recall': recall, 'f': f}

    @staticmethod
    def calc_f(precision, recall):
        try:
            return 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            return 0

    def evaluate(self):
        data = self.potential_matches.sample(frac=1, random_state=self.get_seed()).reset_index(drop=True)
        matches = {}
        group_by = ['src_app', 'target_app', 'src_event_index']

        matches[self.active_technique] = data.iloc[data.reset_index().groupby(group_by)[self.active_technique].idxmax()]
        if self.correct_number == 0:
            self.correct_number = matches[self.active_technique].shape[0]
        return self.calc_metrics(matches, self.correct_number)

    @staticmethod
    def get_rank_of_correct_match(group, field):
        data = group.copy()
        data['rank'] = data[field].rank(ascending=False)
        correct_index = data.loc[data['target_label'] == 'correct'].index.values.astype(int)
        if correct_index.size == 0:
            return 0
        return data[data['target_label'] == 'correct'].reset_index()['rank'][0]

    def is_between_top_n(self, group, field, top_n):
        rank = self.get_rank_of_correct_match(group, field)
        if rank == 0:
            return 0
        return 1 if rank <= top_n else 0

    def evaluate_by_top_rank(self):
        data = self.ranking_potential_matches
        result = {}
        group_by = ['src_app', 'target_app', 'src_event_index']

        groups = data.reset_index().groupby(group_by)
        top_n_number = 0
        for name, group in groups:
            top_n_number += self.is_between_top_n(group, self.active_technique, self.config.top_n)
        result[self.active_technique] = top_n_number
        return result

    def evaluate_by_mrr(self):
        data = self.ranking_potential_matches
        result = {}
        group_by = ['src_app', 'target_app', 'src_event_index']
        groups = data.reset_index().groupby(group_by)

        sum_reveres_rank = 0
        for name, group in groups:
            rank = self.get_rank_of_correct_match(group, self.active_technique)
            if rank == 0:
                print('Warning: for an event there was not match!!!')
                continue
            sum_reveres_rank += 1.0 / rank
        result[self.active_technique] = sum_reveres_rank / self.correct_number
        return result

    def map_exit(self):
        return os.path.exists(self.config.save_score_dir + self.save_path)

    def load_descriptors(self, save_path, map_type):
        if self.map_exit():
            descriptors_data = read_csv(self.config.save_score_dir + save_path, encoding='latin-1')
        else:
            mapping = get_map(map_type)
            descriptors_data = self.create_sim(mapping, save_path)
        return descriptors_data

    def get_seed(self):
        return round(time.time() * 10) % 1000000

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

    def add_rank_to_df(self, data):
        group_by = ['src_app', 'target_app', 'src_event_index']
        data['rank'] = data.reset_index().groupby(group_by)[self.active_technique].rank(ascending=False)
