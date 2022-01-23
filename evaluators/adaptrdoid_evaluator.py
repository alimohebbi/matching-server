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

    def get_potential_matches(self, data):
        condition_same_type = (data['src_type'] == data['target_type'])
        condition_diff_type = ((data['src_type'] == 'fillable') & (data['target_type'] == 'clickable'))
        potential_matches = data[condition_same_type | condition_diff_type]
        return potential_matches
