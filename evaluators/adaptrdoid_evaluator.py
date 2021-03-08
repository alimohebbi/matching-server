import pandas as pd

from descriptor_processes.load_data import DescriptorTypes, add_src_target_string
from descriptor_processes.text_pre_process import space_cleaner
from evaluators.abstract_evaluator import AbstractEvaluator


class CompatibleDescriptors:
    src_semantic_rep = None
    target_semantic_rep = None

    def __init__(self, src_rep, target_rep):
        self.src_semantic_rep = src_rep
        self.target_semantic_rep = target_rep


class AdaptdroidEvaluator(AbstractEvaluator):
    def make_descriptors_compatible(self, row):
        src_rep = self.get_representation('src', row)
        target_rep = self.get_representation('target', row)
        return CompatibleDescriptors(src_rep, target_rep)

    def get_representation(self, event_side, row):
        representation = row[event_side + '_text']
        if event_side + '_neighbors' in row:
            representation += ' ' + row[event_side + '_neighbors']
        if not space_cleaner(representation) and row[event_side + '_id']:
            representation = self.get_file_name(event_side, row)
        if row[event_side + '_type'] == 'fillable' and self.get_fillable_neighbor(event_side, row):
            representation += ' ' + self.get_fillable_neighbor(event_side, row)
        if not space_cleaner(representation) or self.get_file_name(event_side, row):
            representation += ' ' + row[event_side + '_content_desc']
            representation += ' ' + row[event_side + '_id']
        if DescriptorTypes[self.descriptors_type] == DescriptorTypes.craftdroid or \
                DescriptorTypes[self.descriptors_type] == DescriptorTypes.union:
            representation += ' ' + self.add_craftdroid_extra_descriptors(event_side, row)
        return space_cleaner(representation)

    @staticmethod
    def add_craftdroid_extra_descriptors(event_side, row):
        fields = ['activity', 'parent_text', 'sibling_text']
        src_fields, target_fields = add_src_target_string(fields)
        if event_side == 'src':
            return ' '.join(row[src_fields].to_list())
        else:
            return ' '.join(row[target_fields].to_list())

    @staticmethod
    def get_file_name(event_side, row):
        column = event_side + '_file_name'
        if column in row:
            return row[column]
        return ''

    def get_fillable_neighbor(self, event_side, row):
        neighbor = ''
        if self.descriptors_type in [DescriptorTypes.atm, DescriptorTypes.union]:
            neighbor = row[event_side + '_atm_neighbor']
        elif self.descriptors_type == DescriptorTypes.adaptdroid:
            neighbor = row[event_side + '_fillable_neighbor']
        return neighbor

    def assign_score(self, descriptors: CompatibleDescriptors):
        return self.technique.calc_sim(descriptors.src_semantic_rep, descriptors.target_semantic_rep)

    def has_equal_word(self, row):
        src_rep = self.get_representation('src', row)
        target_rep = self.get_representation('target', row)
        a_tokens = src_rep.split()
        b_tokens = target_rep.split()
        words_to_avoid = ["edit", "plus", "be", "input", "view"]
        for i in a_tokens:
            if i in b_tokens and i not in words_to_avoid:
                return True
        return False

    def get_between_thresholds_matches(self, potential_matches):
        embedding = self.active_technique
        condition_a = (potential_matches[embedding] < self.config.adaptdroid_higher_threshold)
        condition_b = (potential_matches[embedding] >= self.config.adaptdroid_lower_threshold)
        between_threshold = potential_matches[condition_a & condition_b]
        matches = between_threshold[
            between_threshold.apply(lambda x: self.has_equal_word(x), axis=1)]
        return matches

    def evaluate(self):
        data = self.potential_matches.sample(frac=1, random_state=self.get_seed()).reset_index(drop=True)
        above_threshold = data[data[self.active_technique] >= self.config.adaptdroid_higher_threshold]
        between_threshold = self.get_between_thresholds_matches(self.potential_matches)
        matches = pd.concat([above_threshold, between_threshold], axis=0, ignore_index=True)
        return self.calc_metrics(matches, self.correct_number)

    def get_potential_matches(self, data):
        condition_same_type = (data['src_type'] == data['target_type'])
        condition_diff_type = ((data['src_type'] == 'fillable') & (data['target_type'] == 'clickable'))
        potential_matches = data[condition_same_type | condition_diff_type]
        return potential_matches

    def get_ranking_potential_matches(self, descriptors_data):
        return self.potential_matches[
            self.potential_matches[self.active_technique] >= self.config.adaptdroid_lower_threshold]

    @staticmethod
    def probability_find_correct(group):
        correct_exit = group[group['target_label'] == 'correct'].shape[0]
        if correct_exit == 0:
            return 0
        return 1 / group.shape[0]

    def calc_tp(self, matches):
        group_by = ['src_app', 'target_app', 'src_event_index']
        groups = matches.reset_index().groupby(group_by)
        prob_sum = 0
        for name, group in groups:
            prob_sum += self.probability_find_correct(group)
        tp = prob_sum
        return tp

    def set_threshold(self, threshold):
        adaptroid_threshold_diff = 0.15
        self.config.adaptdroid_lower_threshold = threshold
        self.config.adaptdroid_higher_threshold = threshold + adaptroid_threshold_diff
