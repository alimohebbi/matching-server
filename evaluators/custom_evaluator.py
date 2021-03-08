from evaluators.abstract_evaluator import AbstractEvaluator
from descriptor_processes.load_data import DescriptorTypes, add_src_target_string, ApproachDescriptors


class CompatibleDescriptors:
    src_event = None
    target_event = None

    def __init__(self, src_event, target_event):
        self.src_event = src_event
        self.target_event = target_event


class CustomEvaluator(AbstractEvaluator):
    def make_descriptors_compatible(self, row):
        descriptors_fields = ApproachDescriptors.descriptors_dict[self.descriptors_type].copy()
        # if 'activity' in descriptors_fields:
        #     descriptors_fields.remove('activity')
        src_columns, target_columns = add_src_target_string(descriptors_fields)
        src_descriptors = set(row[src_columns])
        src_event = ' '.join(src_descriptors)
        target_descriptors = set(row[target_columns])
        target_event = ' '.join(target_descriptors)
        return CompatibleDescriptors(src_event, target_event)

    def assign_score(self, descriptors: CompatibleDescriptors):
        return self.technique.calc_sim(descriptors.src_event, descriptors.target_event)

    def get_potential_matches(self, descriptors_data):
        return descriptors_data[descriptors_data['src_type'] == descriptors_data['target_type']]
