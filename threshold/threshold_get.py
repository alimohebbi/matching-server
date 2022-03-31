from config import Config
import pandas as pd

config = Config()

thresholds = pd.read_csv(config.thresholds)
pairs = set(thresholds['app_pairs'])


def revers_pair_name(pair: str):
    elements = pair.split('-')
    return '-'.join([elements[1], elements[0]])


def pair_find(raw_pair):
    pair = raw_pair.replace('b2', '')
    if pair in pairs:
        return pair
    else:
        return revers_pair_name(pair)


def get_threshold(evaluation_config):
    if 'craftdroid' in evaluation_config['app_pair'] or evaluation_config['algorithm'] == 'random' or\
            evaluation_config['algorithm'] == 'perfect':
        return 0

    app_pair = pair_find(evaluation_config['app_pair'])
    condition = (thresholds['embedding'] == evaluation_config['word_embedding']) & \
                (thresholds['train_set'] == evaluation_config['training_set']) & \
                (thresholds['algorithm'] == evaluation_config['algorithm']) & \
                (thresholds['app_pairs'] == app_pair)

    index = thresholds.index[condition].tolist()[0]
    return thresholds.iloc[index]['threshold']


if __name__ == '__main__':
    eval_conf = {'word_embedding': 'fast', 'training_set': 'blogs', 'algorithm': 'atm',
                 'app_pair': 'a13b2-a11b2'}
    print(get_threshold(eval_conf))
