from config import Config
import pandas as pd

config = Config()

thresholds = pd.read_csv(config.thresholds)


def get_threshold(embedding, train_set):
    condition = (thresholds['embedding'] == embedding) & (thresholds['train_set'] == train_set)
    index = thresholds.index[condition].tolist()[0]
    return thresholds.iloc[index]['threshold']


if __name__ == '__main__':
    print(get_threshold('edit', 'empty'))
