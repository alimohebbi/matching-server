from datetime import date, datetime
from typing import Optional

import pandas as pd

from config import Config
from descriptor_processes.load_data import reformat_df, concat_rows_horizontally, ApproachDescriptors

config = Config()
timestamp = None


class MigrationInfo():
    atm_craft_map = {'a6': 'ExpenseTracker', 'a7': 'NoteTaking', 'a8': 'ShoppingList'}

    def __init__(self, app_pair=None):
        if app_pair:
            split_app_names = app_pair.split('-')
            self.src = split_app_names[0]
            self.target = split_app_names[1]
            self.task = split_app_names[2] if len(split_app_names) == 4 else ''
            if self.src[:-1] in MigrationInfo.atm_craft_map:
                self.src = MigrationInfo.atm_craft_map[self.src[:-1]] + self.src[-1]
                self.target = MigrationInfo.atm_craft_map[self.target[:-1]] + self.target[-1]
                self.task = ''

    @staticmethod
    def set_info_from_log(log_row):
        mig_info = MigrationInfo()
        mig_info.src = log_row['src']
        mig_info.target = log_row['target']
        mig_info.task = log_row['task']
        if mig_info.src[:-1] in MigrationInfo.atm_craft_map:
            mig_info.src = MigrationInfo.atm_craft_map[mig_info.src[:-1]] + mig_info.src[-1]
            mig_info.target = MigrationInfo.atm_craft_map[mig_info.target[:-1]] + mig_info.target[-1]
            mig_info.task = ''
        return mig_info


def transform_json_to_df(events_list) -> pd.DataFrame:
    df_target = pd.DataFrame.from_dict(events_list['candidates']).T
    df_source = pd.DataFrame([events_list['sourceEvent']])
    df_target = reformat_df(df_target, 'target_')
    df_source = reformat_df(df_source, 'src_')
    source_target_df = concat_rows_horizontally(df_source, df_target)
    return source_target_df


def log_query_raw(events_list, semantic_config):
    today = date.today()
    file_name = config.query_log + str(today) + '-raw.csv'
    events_list_df = transform_json_to_df(events_list)
    add_migration_info(events_list_df, semantic_config)
    with open(file_name, 'a') as f:
        events_list_df.to_csv(f, mode='a', header=f.tell() == 0)


def add_non_existing_columns(df):
    required_columns = ApproachDescriptors.union + ['class']

    for i in required_columns:
        src_attr = 'src_' + i
        target_attr = 'target_' + i
        if src_attr not in df.columns:
            df[src_attr] = ''
            df[target_attr] = ''
    return df


def log_query_scored(scored_events: pd.DataFrame, semantic_config):
    today = date.today()
    file_name = config.query_log + str(today) + '.csv'
    add_migration_info(scored_events, semantic_config)
    add_non_existing_columns(scored_events)
    with open(file_name, 'a') as f:
        scored_events.to_csv(f, mode='a', header=f.tell() == 0)


def log_query(scored_events, events_list, semantic_config):
    global timestamp
    timestamp = datetime.now().now().timestamp()
    log_query_scored(scored_events, semantic_config)
    # log_query_raw(events_list, semantic_config)


def add_migration_info(events_list_df, semantic_config):
    events_list_df['index'] = timestamp
    mg_info = MigrationInfo(semantic_config['app_pair'])
    events_list_df['src'] = mg_info.src
    events_list_df['target'] = mg_info.target
    events_list_df['task'] = mg_info.task
    events_list_df['algorithm'] = semantic_config['algorithm']
    events_list_df['word_embedding'] = semantic_config['word_embedding']
    events_list_df['descriptors'] = semantic_config['descriptors']
    events_list_df['training_set'] = semantic_config['training_set']
