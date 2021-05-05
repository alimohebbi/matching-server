import json
import random
from pprint import pprint

from config import Config
from evaluators.evaluator_builder import EvaluatorBuilder

config = Config()
embedding = config.active_techniques[0]
train_set = config.train_sets[0]
algorithm = config.algorithm[0]
descriptor = config.descriptors[0]
semantic_config = {"algorithm": algorithm, "descriptors": descriptor, "training_set": train_set,
                   'word_embedding': embedding}


def forbidden_config(semantic_config):
    if semantic_config['word_embedding'] in ['jaccard', 'edit_distance', 'random']:
        return semantic_config['training_set'] != 'empty'
    if semantic_config['word_embedding'] in ['use', 'nnlm', 'bert']:
        return semantic_config['training_set'] != 'standard'
    if semantic_config['word_embedding'] not in ['jaccard', 'edit_distance', 'random']:
        return semantic_config['training_set'] == 'empty'


def score_descriptors(events_list):
    semantic_config = json.loads(events_list['smConfig'])
    if forbidden_config(semantic_config):
        raise Exception("Config is forbidden")
    if not events_list['sourceEvent']:
        print('Source event is missing')
        return '{}'
    builder = EvaluatorBuilder()
    builder.set_evaluation_config(semantic_config)
    builder.set_events_list(events_list)
    evaluator = builder.build()
    scored_events = evaluator.potential_matches.sort_values(by=['score'], ascending=False)
    return scored_events['score'].to_json()


def score_descriptors2(descriptors):
    candidates = descriptors["candidates"]
    result_keys = list(candidates.keys())
    pprint(descriptors["targetLabels"])
    random.shuffle(result_keys)
    result = {}
    for i in result_keys:
        result[i] = random.random()
    pprint(result)
    return result


def delete_non_related_keys(candidates):
    for k, v in candidates.items():
        to_del = []
        for k2 in v.keys():
            if k2 not in ['text', 'resource-id']:
                to_del.append(k2)
        for i in to_del:
            v.pop(i)


if __name__ == '__main__':
    # sample = open('input_sample-old.txt').read()
    sample = open('syntetic_input.txt').read()
    list_events = json.loads(sample)
    results = score_descriptors(list_events)
    pprint(results)
