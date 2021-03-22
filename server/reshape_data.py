from pprint import pprint

import pandas as pd

from descriptor_processes.load_data import ApproachDescriptors, concat_rows_horizontally
from server.add_neighbors import add_neighbors


def clean_id(id):
    parts = id.split(":id/")
    return parts[1] if len(parts) > 1 else ''


def reformat_df(df, columns_prefix):
    df = df.rename(columns={"content-desc": "content_desc", "resource-id": "id"})
    selected_columns = [i for i in df.columns if i in ApproachDescriptors.all or i in ['class']]
    df = df[selected_columns]
    temp_col = []
    for i in df.columns:
        temp_col.append(columns_prefix + i)
    df.columns = temp_col
    df[columns_prefix + 'id'] = df[columns_prefix + 'id'].apply(lambda s: clean_id(s))
    return df


def create_source_target_map(events_list):
    df_target = pd.DataFrame(events_list['candidates']).T
    df_labels = pd.DataFrame(events_list['labels']).T
    df_source = pd.DataFrame([events_list['source']])
    add_neighbors(df_target, df_labels)
    df_target = reformat_df(df_target, 'target_')
    df_source = reformat_df(df_source, 'src_')
    source_target_df = concat_rows_horizontally(df_source, df_target)
    pprint(source_target_df)
