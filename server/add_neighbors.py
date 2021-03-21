import ast

import pandas as pd


def add_neighbors(events: pd.DataFrame, labels: pd.DataFrame):
    events['neighbor'] = ''
    for k, i in events.iterrows():
        text = ''
        min_distance = 100000
        for m, j in labels.iterrows():
            neighbor, distance = relative_position(i, j)
            if neighbor and min_distance > distance:
                text = get_text(j)
                min_distance = distance
            events['neighbor'][int(k)] = text


def relative_position(event: pd.Series, label: pd.Series):
    event_p = Position(event)
    label_p = Position(label)

    if event_p.get_x1() < label_p.get_x1():
        event_p.set_x1(event_p.get_x2())
    else:
        label_p.set_x1(label_p.get_x2())

    diff_y1 = abs(event_p.get_y1() - label_p.get_y1())
    diff_x1 = abs(event_p.get_x1() - label_p.get_x1())

    if diff_y1 < 150 and diff_x1 < 150:
        return True, diff_x1 + diff_y1
    return False, diff_x1 + diff_y1


class Position:
    def __init__(self, series):
        self.bounds = ast.literal_eval('[' + series['bounds'].replace('][', '],[') + ']')

    def get_x1(self):
        return self.bounds[0][0]

    def get_y1(self):
        return self.bounds[0][1]

    def get_x2(self):
        return self.bounds[1][0]

    def get_y2(self):
        return self.bounds[1][1]

    def set_x1(self, x1):
        self.bounds[0][0] = x1


def get_text(series):
    if series['text']:
        return series['text']
    else:
        return series['content-desc']
