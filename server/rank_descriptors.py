import ast
from pprint import pprint
import random
import json

import pandas as pd

from server.add_neighbors import add_neighbors
from server.reshape_data import reformat_df, create_source_target_map


def rank_descriptors(descriptors):
    candidates = descriptors["candidates"]
    result_keys = list(candidates.keys())
    pprint(descriptors["labels"])
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
    sample = open('input_sample.txt').read()
    list_events = json.loads(sample)
    create_source_target_map(list_events)
