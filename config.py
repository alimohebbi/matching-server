import os
import yaml


class Config(object):

    def __init__(self):
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(THIS_FOLDER, 'config.yml')
        with open(path, 'r') as ymlfile:
            self._config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        paths = self._get_property('model_path')
        for key in paths.keys():
            paths[key] = self._get_property('model_dir') + paths[key]
        self._config['model_path'] = paths

    @property
    def train_sets(self):
        return self._get_property('train_set')

    @property
    def clusters(self):
        return self._get_property('clusters')

    @property
    def cache_size(self):
        return self._get_property('cache_size')

    @property
    def eval_repeat(self):
        return self._get_property('eval_repeat')

    @property
    def save_p_results_path(self):
        return self._get_property('save_p_results_path')

    @property
    def save_rank_results_path(self):
        return self._get_property('save_rank_results_path')

    @property
    def train_data_path(self):
        return self._get_property('train_data_path')

    @property
    def algorithm(self):
        return self._get_property('algorithm')

    @property
    def descriptors(self):
        return self._get_property('descriptors')

    @property
    def top_n(self):
        return self._get_property('top_n')

    @property
    def atm_threshold(self):
        return self._get_property('atm_threshold')

    @property
    def adaptdroid_lower_threshold(self):
        return self._get_property('adaptdroid_lower_threshold')

    @property
    def adaptdroid_higher_threshold(self):
        return self._get_property('adaptdroid_higher_threshold')

    @property
    def model_path(self):
        return self._get_property('model_path')

    @property
    def src_event_dir(self):
        return self._get_property('src_event_dir')

    @property
    def target_event_dir(self):
        return self._get_property('target_event_dir')

    @property
    def index_path(self):
        return self._get_property('index_path')

    @property
    def save_score_dir(self):
        return self._get_property('save_score_dir')

    @property
    def save_map_dir(self):
        return self._get_property('save_map_dir')

    @property
    def active_techniques(self):
        return self._get_property('active_techniques')

    def _get_property(self, property_name):
        if property_name not in self._config.keys():  # we don't want KeyError
            return None  # just return None if not found
        return self._config[property_name]

    @atm_threshold.setter
    def atm_threshold(self, value):
        self._config['atm_threshold'] = value

    @top_n.setter
    def top_n(self, value):
        self._config['top_n'] = value

    @adaptdroid_lower_threshold.setter
    def adaptdroid_lower_threshold(self, value):
        self._config['adaptdroid_lower_threshold'] = value

    @adaptdroid_higher_threshold.setter
    def adaptdroid_higher_threshold(self, value):
        self._config['adaptdroid_higher_threshold'] = value
