import enum

import pandas as pd

from config import Config
from descriptor_processes.text_pre_process import pre_process
from server.add_descriptors import add_neighbors, add_types

config = Config()


class ApproachDescriptors:
    default_columns = ['src_class', 'target_class', 'src_type', 'target_type']
    atm = ['text', 'id', 'content_desc', 'hint', 'neighbor', 'file_name']
    craftdroid = ['text', 'id', 'content_desc', 'hint', 'parent_text', 'sibling_text', 'activity']
    adaptdroid = ['text', 'id', 'content_desc', 'file_name', 'neighbors', 'fillable_neighbor']
    union = ['text', 'id', 'content_desc', 'hint', 'parent_text', 'sibling_text', 'activity', 'atm_neighbor',
             'file_name']
    intersection = ['text', 'id', 'content_desc', 'hint']
    descriptors_dict = {'default': default_columns, 'atm': atm, 'craftdroid': craftdroid, 'adaptdroid': adaptdroid,
                        'union': union, 'intersection': intersection}
    all = set(atm + craftdroid + adaptdroid)


def concat_rows_horizontally(src_row, target_rows):
    result = []
    for index, row in target_rows.iterrows():
        concated = pd.concat([src_row.reset_index(), row.to_frame().T.reset_index()], axis=1)
        concated.drop(['index'], axis=1, inplace=True)
        result.append(concated)
    return pd.concat(result, axis=0, ignore_index=True)


def clean(result):
    src_fields, target_fields = add_src_target_string(ApproachDescriptors.all)
    acceptable_columns = src_fields + target_fields
    cleaning_columns = [i for i in result.columns if i in acceptable_columns]
    result.loc[:, cleaning_columns] = pre_process(result.loc[:, cleaning_columns], False)
    return result


def get_mapping(events_list):
    df_target = pd.DataFrame(events_list['candidates']).T
    df_source_labels = pd.DataFrame(events_list['target_labels']).T
    df_target_labels = pd.DataFrame(events_list['source_labels']).T
    df_source = pd.DataFrame([events_list['source']])
    add_neighbors(df_target, df_target_labels)
    add_neighbors(df_source, df_source_labels)
    add_types(df_target)
    add_types(df_source)
    df_target = reformat_df(df_target, 'target_')
    df_source = reformat_df(df_source, 'src_')
    source_target_df = concat_rows_horizontally(df_source, df_target)
    final_map = clean(source_target_df)
    return final_map.fillna('')


def clean_id(reource_id):
    parts = reource_id.split(":id/")
    return parts[1] if len(parts) > 1 else ''


def reformat_df(df, columns_prefix):
    df = df.rename(columns={"content-desc": "content_desc", "resource-id": "id"})
    default = ['class', 'type']
    selected_columns = [i for i in df.columns if
                        i in ApproachDescriptors.all or i in default]
    df = df[selected_columns]
    temp_col = []
    for i in df.columns:
        temp_col.append(columns_prefix + i)
    df.columns = temp_col
    df[columns_prefix + 'id'] = df[columns_prefix + 'id'].apply(lambda s: clean_id(s))
    return df


def column_selector(events_list, descriptors_names):
    map = get_mapping(events_list)
    src_descriptors, target_descriptors = add_src_target_string(descriptors_names)
    return map[ApproachDescriptors.default_columns + src_descriptors + target_descriptors]


def add_src_target_string(descriptors):
    src_descriptors = ['src_' + i for i in descriptors]
    target_descriptors = ['target_' + i for i in descriptors]
    return src_descriptors, target_descriptors


class DescriptorTypes(enum.Enum):
    adaptdroid = 0
    atm = 1
    craftdroid = 2
    intersection = 3
    union = 4
    atm_craft = 5


def get_map(events_list, desc_type):
    return column_selector(events_list, ApproachDescriptors.descriptors_dict[desc_type])
