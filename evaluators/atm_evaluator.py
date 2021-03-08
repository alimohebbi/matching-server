from descriptor_processes.load_data import DescriptorTypes, add_src_target_string, ApproachDescriptors
from evaluators.abstract_evaluator import AbstractEvaluator


class CompatibleDescriptors:
    src_label = None
    src_id = None
    src_text = None
    target_label = None
    target_id = None
    target_text = None
    src_class = None
    target_class = None


class ATMEvaluator(AbstractEvaluator):

    def make_descriptors_compatible(self, row):
        self.add_file_name_to_id(row)
        c_descriptors = CompatibleDescriptors()
        c_descriptors.src_id = row['src_id']
        c_descriptors.target_id = row['target_id']
        c_descriptors.src_text = self.get_text('src', row)
        c_descriptors.target_text = self.get_text('target', row)
        c_descriptors.src_label = self.get_label('src', row)
        c_descriptors.target_label = self.get_label('target', row)
        c_descriptors.src_class = row['src_class']
        c_descriptors.src_class = row['target_class']
        if DescriptorTypes[self.descriptors_type] == DescriptorTypes.craftdroid or \
                DescriptorTypes[self.descriptors_type] == DescriptorTypes.union:
            self.add_craftdroid_extra_descriptors(c_descriptors, row)
        return c_descriptors

    def assign_score(self, descriptors: CompatibleDescriptors):
        if 'EditText' in descriptors.src_class:
            return self.compute_editable(descriptors)
        else:
            return self.compute_non_editable(descriptors)

    def get_potential_matches(self, data):
        threshold_condition = (data[self.active_technique] >= self.config.atm_threshold)
        same_type_condition = (data['src_type'] == data['target_type'])
        return data[threshold_condition & same_type_condition]

    def set_threshold(self, threshold):
        self.config.atm_threshold = threshold

    @staticmethod
    def get_text(event_side, row):
        if row[event_side + '_text']:
            return row[event_side + '_text']
        if row[event_side + '_content_desc']:
            return row[event_side + '_content_desc']
        if event_side + '_hint' in row:
            return row[event_side + '_hint']
        return ''

    def get_label(self, event_side, row):
        if event_side + '_atm_neighbor' in row and row[event_side + '_atm_neighbor']:
            return row[event_side + '_atm_neighbor']
        elif DescriptorTypes[self.descriptors_type] == DescriptorTypes.adaptdroid:
            if row[event_side + '_neighbors'] or row[event_side + '_fillable_neighbor']:
                return row[event_side + '_neighbors'] + ' ' + row[event_side + '_fillable_neighbor']
        return row[event_side + '_id']

    @staticmethod
    def add_craftdroid_extra_descriptors(c_descriptors, row):
        fields = ['activity', 'parent_text', 'sibling_text']
        src_fields, target_fields = add_src_target_string(fields)
        c_descriptors.src_text += ' ' + ' '.join(row[src_fields].to_list())
        c_descriptors.target_text += ' ' + ' '.join(row[target_fields].to_list())

    def compute_editable(self, descriptors: CompatibleDescriptors):
        text_text_score = self.atm_token_sim(descriptors.src_text, descriptors.target_text)
        label_label_score = self.atm_token_sim(descriptors.src_label, descriptors.target_label)
        text_label_score = self.atm_token_sim(descriptors.src_text, descriptors.target_label)
        label_text_score = self.atm_token_sim(descriptors.src_label, descriptors.target_text)
        return max(text_text_score, label_label_score, text_label_score, label_text_score)

    def compute_non_editable(self, descriptors):
        text_text_score = self.atm_token_sim(descriptors.src_text, descriptors.target_text)
        id_id_score = self.atm_token_sim(descriptors.src_id, descriptors.target_id)
        id_text_score = self.atm_token_sim(descriptors.src_id, descriptors.target_text)
        text_id_score = self.atm_token_sim(descriptors.src_text, descriptors.target_id)
        return max(text_text_score, id_id_score * 0.9, id_text_score * 0.9, text_id_score * 0.9)

    def atm_token_sim(self, src, target):
        if self.technique.sentence_level:
            return self.technique.calc_sim(src, target)
        else:
            min_length = min(src.split().__len__(), target.split().__len__())
            return self.technique.calc_sim(src, target) * min_length

    def add_file_name_to_id(self, row):
        if 'file_name' not in ApproachDescriptors.descriptors_dict[self.descriptors_type]:
            return
        src_fields, target_fields = add_src_target_string(['file_name'])
        row['src_id'] += ' ' + ' '.join(row[src_fields].to_list())
        row['target_id'] += ' ' + ' '.join(row[target_fields].to_list())
