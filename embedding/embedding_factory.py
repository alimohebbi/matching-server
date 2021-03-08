import enum
from collections import OrderedDict

from embedding.word_embedding import *
from exception.embedding_exceptions import ChangeOfLevel


class EmbedTypes(enum.Enum):
    wm = 1
    w2v = 2
    nnlm = 3
    use = 4
    bert = 5
    random = 6
    fast = 7
    glove = 8
    jaccard = 9
    edit_distance = 10


model_cache = OrderedDict()

embeddings = {
    EmbedTypes.wm: WordMovers,
    EmbedTypes.w2v: Word2VecS,
    EmbedTypes.glove: Glove,
    EmbedTypes.nnlm: TensorFlowEmbed,
    EmbedTypes.use: TensorFlowEmbed,
    EmbedTypes.bert: Bert,
    EmbedTypes.random: RandomEmbed,
    EmbedTypes.fast: Fast,
    EmbedTypes.jaccard: Jaccard,
    EmbedTypes.edit_distance: EditDistance
}

name_to_type = {
    'WordMovers': EmbedTypes.wm,
    'Word2VecS': EmbedTypes.w2v,
    'Glove': EmbedTypes.glove,
    'Fast': EmbedTypes.fast,
}


class HierarchyDecorator(ABC):
    train_set_hierarchy = ['topics', 'category', 'googleplay', 'standard', 'empty']

    def train_set_trials(self, a, b):
        current_train_set_index = self.train_set_hierarchy.index(self.train_set)
        class_name = type(self.domain_model).__name__
        score = 0
        while score == 0 and current_train_set_index != self.train_set_hierarchy.index('empty'):
            temp_model = self.choose_temp_model(class_name, current_train_set_index)
            score = self.undecorated_score(temp_model, a, b)
            current_train_set_index += 1
        return score

    def choose_temp_model(self, class_name, current_train_set_index):
        next_train_set = self.train_set_hierarchy[current_train_set_index + 1]
        if next_train_set == 'category':
            temp_model = ClustersEmbeddingFactory.get_instance().get_model(name_to_type[class_name], next_train_set,
                                                                           self.source_app)
        elif next_train_set != 'empty':
            temp_model = SimpleEmbeddingFactory.get_instance().get_model(name_to_type[class_name], next_train_set)
        elif self.syntactic == 'edit':
            temp_model = SimpleEmbeddingFactory.get_instance().get_model(EmbedTypes.edit_distance, 'empty')
        elif self.syntactic == 'jaccard' and isinstance(self, SentenceLevelDecorator):
            temp_model = SimpleEmbeddingFactory.get_instance().get_model(EmbedTypes.jaccard, 'empty')
        else:
            raise ChangeOfLevel
        return temp_model

    def __init__(self, domain_model: WordEmbedding, hierarchy_info, source_app):
        self.domain_model = domain_model
        self.train_set = hierarchy_info.split('_')[1]
        self.syntactic = hierarchy_info.split('_')[2]
        self.source_app = source_app

    @abstractmethod
    def undecorated_score(self, temp_model, a, b):
        pass


class WordLevelDecorator(HierarchyDecorator, Word2VecS):
    def undecorated_score(self, temp_model, a, b):
        return temp_model.two_word_sim(a, b)

    def calc_sim_by_model(self, a, b):
        try:
            matrix = sim_matrix_of_tokens(a, b, self._two_word_sim_decorator)
            score = matrix_overall_sim(matrix)
            if score != 0:
                return score
        except ChangeOfLevel:
            self.zeros += 1
            temp_model = SimpleEmbeddingFactory.get_instance().get_model(EmbedTypes.jaccard, 'empty')
            score = temp_model.calc_sim_by_model(a, b)
        return score

    def two_word_sim(self, a, b):
        score = self.domain_model.two_word_sim(a, b)
        if score != 0:
            return score
        return self.train_set_trials(a, b)


class SentenceLevelDecorator(HierarchyDecorator, WordEmbedding):
    sentence_level = True

    def undecorated_score(self, temp_model, a, b):
        return temp_model.calc_sim_by_model(a, b)

    def calc_sim_by_model(self, a, b):
        score = self.domain_model.calc_sim_by_model(a, b)
        if score != 0:
            return score
        return self.train_set_trials(a, b)


class EmbeddingFactory(ABC):

    def __init__(self):
        self.embedding_type = None
        self.train_set = None
        self.train_set = None
        self.source_app = None

    __instance = None

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def check_cache(self):
        key = self.get_key()
        if key in model_cache.keys():
            model_cache[key].zeros = 0
            return model_cache[key]
        else:
            self.check_cache_size()
            model_path = self.get_model_path()
            model = embeddings[self.embedding_type](model_path)
            model_cache[key] = model
        return model

    @staticmethod
    def check_cache_size():
        if model_cache.__len__() >= config.cache_size:
            model_cache.popitem(last=False)

    def get_model(self, embedding_type: EmbedTypes, train_set, source_app=None):
        self.embedding_type = embedding_type
        self.train_set = train_set
        self.source_app = source_app
        if train_set.startswith("hierarchy"):
            return self.hierarchical_model_getter(train_set)
        return self.check_cache()

    def hierarchical_model_getter(self, train_set):
        self.train_set = train_set.split('_')[1]
        model = self.check_cache()
        if self.embedding_type == EmbedTypes.wm:
            return SentenceLevelDecorator(model, train_set, self.source_app)
        else:
            return WordLevelDecorator(model, train_set, self.source_app)

    @abstractmethod
    def get_key(self):
        pass

    @abstractmethod
    def get_model_path(self):
        pass


class SimpleEmbeddingFactory(EmbeddingFactory):

    def get_key(self):
        return self.embedding_type.name + self.train_set

    def get_model_path(self):
        no_model_techniques = [EmbedTypes.jaccard, EmbedTypes.edit_distance, EmbedTypes.random]
        if self.embedding_type in no_model_techniques:
            return ''
        name = self.embedding_type.name
        if name == 'wm':
            name = 'w2v'

        model_name = name + '_' + self.train_set
        return config.model_path[model_name]


class ClustersEmbeddingFactory(EmbeddingFactory):

    def get_key(self):
        model_id = self.get_model_id()
        return self.embedding_type.name + self.train_set + model_id

    def get_model_id(self):
        map_file_path = config.clusters['app_to_cluster']
        app_name_to_cluster = pd.read_csv(map_file_path)
        model_id = list(app_name_to_cluster.loc[app_name_to_cluster['app_name'] == self.source_app][self.train_set])[0]
        return str(model_id)

    def get_model_path(self):
        map_file_path = config.clusters['app_to_cluster']
        app_name_to_cluster = pd.read_csv(map_file_path)
        name = self.embedding_type.name
        if name == 'wm':
            name = 'w2v'
        name = self.train_set + '_' + name
        return list(app_name_to_cluster.loc[app_name_to_cluster['app_name'] == self.source_app][name])[0]
