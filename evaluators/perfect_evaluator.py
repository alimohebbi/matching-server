import pandas as pd

from config import Config
from evaluators.abstract_evaluator import AbstractEvaluator

important_filed = ['text', 'content-desc', 'resource-id', 'activity']
config = Config()
gt_table = pd.read_csv(config.ground_truth)


class PerfectEvaluator(AbstractEvaluator):

    def get_potential_matches(self, descriptors_data):
        return descriptors_data

    def make_descriptors_compatible(self, row):
        return row

    def assign_score(self, descriptors):
        app_pair = self.eval_config['app_pair'].replace('craftdroid-', '')
        src = app_pair.split('-')[0]
        target = app_pair.split('-')[1]
        task = app_pair.split('-')[2]
        condition_migration = (gt_table['src'] == src) & (gt_table['target'] == target) & (gt_table['task'] == task)
        condition_event = (gt_table['text'] == descriptors['target_text']) & (
                    gt_table['content-desc'] == descriptors['target_content_desc']) & \
                          (gt_table['resource-id'] == descriptors['target_id']) & (
                                      gt_table['activity'] == descriptors['target_activity'])

        migration = gt_table[condition_migration]
        migration = migration[condition_event]
        if migration.shape[0] > 0:
            return 1
        else:
            return 0

    def apply_threshold(self, potential_matches):
        return potential_matches
