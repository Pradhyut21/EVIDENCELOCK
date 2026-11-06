import copy
import unittest

from gpu.decision_protocol import decision_protocol, select_decision


def fixture():
    return {'question': 'What is justified at X?', 'evidence_items': [],
            'as_of': '2026-11-01T10:00:00Z',
            'query_policy': {'required_fields': ['f'], 'in_scope': True},
            'provenance': {'site': 'X'}, 'case_id': 'secret-label-id',
            'oracle_status': 'CONFLICT', 'split': 'validation'}


class ProtocolTests(unittest.TestCase):
    def test_gold_and_annotation_changes_do_not_change_prompt_or_options(self):
        before = fixture()
        after = copy.deepcopy(before)
        after.update(oracle_status='SUPPORTED', case_id='other', split='train')
        after['query_policy']['in_scope'] = False
        self.assertEqual(decision_protocol(before), decision_protocol(after))

    def test_observable_evidence_is_preserved(self):
        data = fixture()
        data['evidence_items'] = [{'evidence_id': 'R77', 'claims': {'f': 'Z'}}]
        prompt = decision_protocol(data)['prompt']
        self.assertIn('R77', prompt)
        self.assertIn('"f":"Z"', prompt)
        self.assertNotIn('secret-label-id', prompt)
        self.assertNotIn('oracle_status', prompt)
        self.assertNotIn('in_scope', prompt)

    def test_choice_comes_from_model_scores(self):
        options = decision_protocol(fixture())['options']
        selected = select_decision({'A': -4., 'B': 3., 'C': 2., 'D': 0.}, options)
        self.assertEqual(selected['status'], options['B'])
        self.assertEqual(selected['logit_margin'], 1.)

    def test_nonfinite_or_missing_scores_fail(self):
        options = decision_protocol(fixture())['options']
        for scores in [{'A': 1.}, {'A': float('nan'), 'B': 1., 'C': 2., 'D': 3.}]:
            with self.assertRaises(ValueError):
                select_decision(scores, options)

    def test_rotations_balance_each_status_across_letters(self):
        options = [decision_protocol(fixture(), rotation=r)['options'] for r in range(4)]
        for letter in 'ABCD':
            self.assertEqual({o[letter] for o in options},
                             {'SUPPORTED', 'MISSING_EVIDENCE', 'CONFLICT', 'OUT_OF_SCOPE'})
        with self.assertRaises(ValueError):
            decision_protocol(fixture(), rotation=4)


if __name__ == '__main__':
    unittest.main()
