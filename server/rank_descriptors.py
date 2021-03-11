import ast
from pprint import pprint
import random


def rank_descriptors(descriptors):
    candidates = descriptors["candidates"]
    result_keys = list(candidates.keys())

    delete_non_related_keys(candidates)

    pprint(candidates)
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
    event1 = {'text': 'bah', 'resource-id': 'ah', 'xx': 'll'}
    event2 = {'text': 'bah2', 'resource-id': 'ah2', 'xx': 'll'}
    a = {'a': event1, 'b': event2}
    input_json = {'candidates': a}
    rank_descriptors(input_json)
