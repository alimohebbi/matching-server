import pandas as pd

from config import Config
from evaluators.abstract_evaluator import AbstractEvaluator
from server.query_util import MigrationInfo

important_filed = ['text', 'content-desc', 'resource-id', 'activity']
config = Config()
gt_table = pd.read_csv(config.ground_truth).fillna('')


class PerfectEvaluator(AbstractEvaluator):

    def get_potential_matches(self, data):
        same_type_condition = (data['src_type'] == data['target_type'])
        return data[same_type_condition]

    def make_descriptors_compatible(self, row):
        return row

    def assign_score(self, descriptors):
        m_info = MigrationInfo(self.eval_config['app_pair'])
        return self.event_exist_in_gt(descriptors, m_info)

    @staticmethod
    def event_exist_in_gt(descriptors, m_info):
        src = m_info.src + m_info.task
        target = m_info.target + m_info.task
        condition_migration = (gt_table['src_app'] == src) & (gt_table['target_app'] == target)
        migration = gt_table[condition_migration]
        condition_target = (migration['target_text'] == descriptors['target_text']) & (
                migration['target_content_desc'] == descriptors['target_content_desc']) & \
                           (migration['target_id'] == descriptors[
                               'target_id'])  # & (migration['target_activity'] == descriptors['target_activity'])
        migration = migration[condition_target]
        condition_event = (migration['src_text'] == descriptors['src_text']) & (
                migration['src_content_desc'] == descriptors['src_content_desc']) & \
                          (migration['src_id'] == descriptors[
                              'src_id'])  # & (migration['target_activity'] == descriptors['target_activity'])
        migration = migration[condition_event]

        if migration.shape[0] > 0:
            return 1.0
        else:
            return 0.0

    def atm_request(self):
        if 'Expense' in self.eval_config['app_pair'] or 'Note' in self.eval_config['app_pair'] or 'Shop' in \
                self.eval_config['app_pair']:
            return True
        else:
            return False

    def apply_threshold(self, potential_matches):
        if self.atm_request():
            potential_matches = potential_matches[potential_matches['score'] > 0]
        return potential_matches
