import enum
import glob
import os
import re

import pandas as pd
from pandas.io.parsers import read_csv

from config import Config
from descriptor_processes.text_pre_process import pre_process

config = Config()


class ApproachDescriptors:
    default_columns = ['src_app', 'target_app', 'src_event_index', 'target_label',
                       'src_class', 'target_class', 'src_type', 'target_type', 'target_event_index']
    atm = ['text', 'id', 'content_desc', 'hint', 'atm_neighbor', 'file_name']
    craftdroid = ['text', 'id', 'content_desc', 'hint', 'parent_text', 'sibling_text', 'activity']
    adaptdroid = ['text', 'id', 'content_desc', 'file_name', 'neighbors', 'fillable_neighbor']
    union = ['text', 'id', 'content_desc', 'hint', 'parent_text', 'sibling_text', 'activity', 'atm_neighbor',
             'file_name']
    intersection = ['text', 'id', 'content_desc', 'hint']
    descriptors_dict = {'default': default_columns, 'atm': atm, 'craftdroid': craftdroid, 'adaptdroid': adaptdroid,
                        'union': union, 'intersection': intersection}
    all = set(atm + craftdroid + adaptdroid)


def add_file_name(csv, path):
    file_name = os.path.basename(path).split('.')[0]
    csv['app'] = file_name
    return csv


def load_csv_dir(dir):
    desc_map_files = []
    for path in glob.glob(dir + "*.csv"):
        csv = read_csv(path, encoding='latin-1')
        csv = add_file_name(csv, path)
        desc_map_files.append(csv)
    return pd.concat(desc_map_files, axis=0, ignore_index=True)


def load_events():
    src_events = load_csv_dir(config.src_event_dir)
    src_events = src_events.loc[src_events['label'] == 'correct']
    target_events = load_csv_dir(config.target_event_dir)
    temp_col = []
    for i in src_events.columns:
        temp_col.append('src_' + i)
    src_events.columns = temp_col
    temp_col = []
    for i in target_events.columns:
        temp_col.append('target_' + i)
    target_events.columns = temp_col
    return src_events, target_events


def concat_rows_horizontally(src_row, target_rows):
    result = []
    for index, row in target_rows.iterrows():
        concated = pd.concat([src_row.reset_index(), row.to_frame().T.reset_index()], axis=1)
        concated.drop(['index'], axis=1, inplace=True)
        result.append(concated)
    return pd.concat(result, axis=0, ignore_index=True)


def clean(result):
    src_fields, target_fields = add_src_target_string(ApproachDescriptors.all)
    result.loc[:, src_fields + target_fields] = pre_process(result.loc[:, src_fields + target_fields], False)
    return result


def get_mapping():
    save_path = config.save_map_dir
    if os.path.exists(save_path):
        result = read_csv(save_path, encoding='latin-1')
    else:
        src_events, target_events = load_events()
        index_map = read_csv(config.index_path, encoding='latin-1')
        final_map = create_map_list(index_map, src_events, target_events)
        result = pd.concat(final_map, axis=0, ignore_index=True)
        result.to_csv(save_path, index=False)
    return result.fillna('')


def validate_correct_label(target_rows, index_row):
    t_row_copy = target_rows.copy(deep=True)
    correct_window_index = index_row['target_index']
    t_row_copy.loc[t_row_copy.target_event_index != correct_window_index, 'target_label'] = 'wrong'
    return t_row_copy


def get_duplicate_concerns(correct_row):
    columns = ApproachDescriptors.union.copy()
    if correct_row['src_type'].values[0] == 'fillable':
        columns.remove('text')
    _, concerns = add_src_target_string(columns)
    return concerns


def remove_duplicates(target_rows: pd.DataFrame, index_row):
    t_row_copy = target_rows.copy(deep=True)
    correct_window_index = index_row['target_index']
    window_condition = t_row_copy['target_event_index'] == correct_window_index
    label_condition = t_row_copy['target_label'] == 'correct'
    correct_row = t_row_copy.loc[window_condition & label_condition].copy(deep=True)
    duplicate_concern_columns = get_duplicate_concerns(correct_row)
    t_row_copy = t_row_copy.drop_duplicates(subset=duplicate_concern_columns)
    t_row_copy = pd.concat([correct_row, t_row_copy])
    return t_row_copy.drop_duplicates(subset=duplicate_concern_columns)


def create_map_list(index_map, src_events, target_events):
    final_map = []
    for index, index_row in index_map.iterrows():
        src_row = src_events.loc[(src_events['src_app'] == index_row['src_app']) &
                                 (src_events['src_event_index'] == index_row['src_index']), :]
        migration_name = index_row['src_app'] + '-' + index_row['target_app']
        target_rows = target_events.loc[(target_events['target_app'] == migration_name), :]
        target_rows = validate_correct_label(target_rows, index_row)
        new_rows = concat_rows_horizontally(src_row, target_rows)
        new_rows = clean(new_rows)
        new_rows = remove_duplicates(new_rows, index_row)
        final_map.append(new_rows)
    return final_map


def column_selector(descriptors):
    map = get_mapping()
    src_descriptors, target_descriptors = add_src_target_string(descriptors)
    return map[ApproachDescriptors.default_columns + src_descriptors + target_descriptors]


def add_src_target_string(descriptors):
    src_descriptors = ['src_' + i for i in descriptors]
    target_descriptors = ['target_' + i for i in descriptors]
    return src_descriptors, target_descriptors


def get_atm_minus_craft_map():
    atm_descriptors = ApproachDescriptors.atm
    craft_descriptors = ApproachDescriptors.craftdroid
    minus_descriptors = [i for i in atm_descriptors if i not in craft_descriptors]
    return column_selector(minus_descriptors)


def get_craft_minus_atm_map():
    atm_descriptors = ApproachDescriptors.atm
    craft_descriptors = ApproachDescriptors.craftdroid
    minus_descriptors = [i for i in craft_descriptors if i not in atm_descriptors]
    return column_selector(minus_descriptors)


class DescriptorTypes(enum.Enum):
    adaptdroid = 0
    atm = 1
    craftdroid = 2
    intersection = 3
    union = 4
    atm_craft = 5


def get_map(desc_type):
    return column_selector(ApproachDescriptors.descriptors_dict[desc_type])
