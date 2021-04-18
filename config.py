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
    def threshold_use(self):
        return self._get_property('threshold_use')

    @property
    def thresholds(self):
        return self._get_property('thresholds')

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
    def model_path(self):
        return self._get_property('model_path')



    @property
    def active_techniques(self):
        return self._get_property('active_techniques')

    def _get_property(self, property_name):
        if property_name not in self._config.keys():  # we don't want KeyError
            return None  # just return None if not found
        return self._config[property_name]

