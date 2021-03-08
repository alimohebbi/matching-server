import logging
import re

import numpy as np
import pickle
import time

import pandas as pd
from gensim.models.ldamulticore import LdaMulticore
from play_scraper import details
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError

from config import Config
from descriptor_processes.text_pre_process import pre_process

config = Config()
log = logging.getLogger("training")
log.setLevel(logging.WARNING)


def app_details(app_id: str) -> dict:
    for i in range(3):
        try:
            return details(app_id)
        except (ReadTimeout, ConnectionError):
            print(f"ReadTimeout error, waiting for {str(i ** 3)} seconds.")
        except (HTTPError, ValueError):
            print("url for %s not found" % app_id)
            return None
        except AttributeError:
            print("AttributeError")
        time.sleep(i ** 3)


def get_dominant_topic(doc):
    if not doc:
        return ''
    vec_bow = dictionary.doc2bow(doc.split())
    if len(topic_model[vec_bow]) == 0:
        print("can't find the model")
        return
    else:
        topic_distribution = dict(topic_model[vec_bow])
        dominant_topic = max(topic_distribution, key=topic_distribution.get)
    return str(dominant_topic)


def get_required_models(path):
    with open(path + "dictionary", 'rb') as pickle_file:
        dictionary = pickle.load(pickle_file)
    with open(path + "tfidf_model", 'rb') as pickle_file:
        tfidf = pickle.load(pickle_file)
    topic_model = LdaMulticore.load(path + "model")
    return dictionary, tfidf, topic_model


def set_app_details(app_cluster_map: pd.DataFrame, detail_name):
    if detail_name not in app_cluster_map.columns:
        app_cluster_map[detail_name] = ''
    for index, row in app_cluster_map.iterrows():
        if row[detail_name] == '':
            details = app_details(row['app_id'])
            app_cluster_map.at[index, detail_name] = details[detail_name] if details else ''


def set_models_path(app_cluster_map: pd.DataFrame, cluster_name):
    for index, row in app_cluster_map.iterrows():
        if str(row[cluster_name]) == '':
            continue
        models_path = config.model_path[cluster_name]
        for i in ['glove', 'w2v', 'fast']:
            j = cluster_name + '_' + i
            model_name = str(row[cluster_name])
            model_name = re.sub('([\'\[\]])', '', model_name)
            app_cluster_map.at[index, j] = models_path + i + '/model_' + model_name


dictionary, tfidf, topic_model = [None, None, None]


def set_topics(app_cluster_map):
    global dictionary, tfidf, topic_model
    app_cluster_map["description"] = pre_process(pd.DataFrame(app_cluster_map['description']), train=True)[
        "description"]
    if 'topics' not in app_cluster_map.columns:
        dictionary, tfidf, topic_model = get_required_models(config.clusters['topic_model'])
        app_cluster_map['topics'] = app_cluster_map.apply(lambda row: get_dominant_topic(row['description']), axis=1)


def set_category_base_models(app_cluster_map):
    for cluster_name in ['topics', 'category']:
        if cluster_name + '_w2v' not in app_cluster_map.columns:
            set_models_path(app_cluster_map, cluster_name)


def complete_app_to_cluster_map():
    global dictionary, tfidf, topic_model
    table_path = config.clusters['app_info']
    app_cluster_map = pd.read_csv(table_path)
    set_app_details(app_cluster_map, 'description')
    set_app_details(app_cluster_map, 'category')
    set_topics(app_cluster_map)
    app_cluster_map.to_csv(config.clusters['app_info'], index=False)
    set_category_base_models(app_cluster_map)
    app_cluster_map.to_csv(config.clusters['app_to_cluster'], index=False)


if __name__ == "__main__":
    complete_app_to_cluster_map()
