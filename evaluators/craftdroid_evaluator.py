import re

from descriptor_processes.load_data import ApproachDescriptors
from evaluators.abstract_evaluator import AbstractEvaluator


class CraftdroidEvaluator(AbstractEvaluator):

    @staticmethod
    def get_acceptable_targets(x):
        src_class = x['src_class']
        src_type = x['src_type']
        text = str(x['src_text'])
        tgt_classes = [src_class]
        if src_class in ['android.widget.ImageButton', 'android.widget.Button']:
            tgt_classes = ['android.widget.ImageButton', 'android.widget.Button', 'android.widget.TextView']
        elif src_class == 'android.widget.TextView':
            if src_type == 'clickable':
                tgt_classes += ['android.widget.ImageButton', 'android.widget.Button']
                if re.search(r'https://\w+\.\w+', text):  # e.g., a15-a1x-b12
                    tgt_classes.append('android.widget.EditText')
        elif src_class == 'android.widget.EditText':
            tgt_classes.append('android.widget.MultiAutoCompleteTextView')  # a43-a41-b42

        elif src_class == 'android.widget.MultiAutoCompleteTextView':  # a41-a43-b42
            tgt_classes.append('android.widget.EditText')

        return tgt_classes

    def get_potential_matches(self, data):
        return data[data.apply(lambda x: x['target_class'] in self.get_acceptable_targets(x), axis=1)]

    def assign_score(self, descriptors):
        fields = ApproachDescriptors.descriptors_dict[self.descriptors_type].copy()
        if 'hint' in fields:
            fields.remove('hint')
        w_scores = []
        for i, attr in enumerate(fields):
            src_field = 'src_' + attr
            target_field = 'target_' + attr
            if src_field in descriptors and descriptors[src_field] and descriptors[target_field]:
                sim_score = self.technique.calc_sim(descriptors[src_field], descriptors[target_field])
                if attr == 'sibling_text':
                    if all([descriptors['src_text'] + descriptors['src_parent_text'],
                            descriptors['src_text'] + descriptors['target_parent_text']]):
                        sim_score = sim_score * 0.5
                w_scores.append(sim_score)

        src_text_attrs = ['src_text', 'src_parent_text']
        target_text_attrs = ['target_text', 'target_parent_text']
        cross_score = -1
        for a1 in src_text_attrs:
            for a2 in target_text_attrs:
                if a1.replace("src_", '') != a2.replace("target_", '') and a1 in descriptors and descriptors[a1] and \
                        a2 in descriptors and descriptors[a2]:
                    sim = self.technique.calc_sim(descriptors[a1], descriptors[a2])
                    if sim and sim > cross_score:
                        cross_score = sim
        if cross_score > -1:
            w_scores.append(cross_score)
        return sum(w_scores) / len(w_scores) if len(w_scores) else 0

    def make_descriptors_compatible(self, row):
        self.add_hint(row)
        return row

    @staticmethod
    def add_hint(row):
        if 'src_hint' in row:
            row['src_text'] = row['src_text'] + ' ' + row['src_hint']
        if 'target_hint' in row:
            row['target_text'] = row['target_text'] + ' ' + row['target_hint']
