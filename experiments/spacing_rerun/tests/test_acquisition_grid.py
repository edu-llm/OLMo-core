import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from spacing_rerun.acquisition import GRID_TRAJECTORY_POLICY
from spacing_rerun.acquisition_grid import (RULE_POLICY, checkpoint_digest, common_grid_dose,
                                            choose_acquisition_grid_checkpoint, verify_acquisition_grid_selection)
from spacing_rerun.analysis import summarize_bundles
from spacing_rerun.common import digest, write_json
from spacing_rerun.data import assign_development_scale_roles
from spacing_rerun.prepare import validate_config
from spacing_rerun.training import run_arm


GRID = [2, 3, 4, 6, 8]
ROOT = Path(__file__).resolve().parents[1]


def rehash(payload):
    payload['sha256'] = digest({k: v for k, v in payload.items() if k != 'sha256'})


def grid_fixture():
    rule = {'schema': 'spacing-acquisition-grid-selection-rule-v1', 'policy': RULE_POLICY,
            'frozen_utc': '2000-01-01T00:00:00+00:00', 'candidate_exposures': GRID,
            'target_min': .4, 'target_max': .7, 'scaled_mean_target': .55,
            'normal_manifest_sha256': ['normal-' + str(i) for i in range(6)],
            'scaled_manifest_sha256': ['scaled-' + str(i) for i in range(3)]}
    rehash(rule)
    decisions = []
    for sha in rule['normal_manifest_sha256'] + rule['scaled_manifest_sha256']:
        decisions.append({'manifest_sha256': sha, 'mode': 'development',
                          'trajectory_policy': GRID_TRAJECTORY_POLICY, 'status': 'acquisition_grid_complete',
                          'selected_E': None, 'acquisition_usable': False,
                          'development_scale_probe': sha.startswith('scaled-'),
                          'grid_started_utc': '2000-01-02T00:00:00+00:00',
                          'grid_results': {str(e): {'exposures': e, 'old_exact_match_event_macro': .5,
                                                    'within_acquisition_gate': True,
                                                    'checkpoint': f'stage1-E{e:03}.pt',
                                                    'checkpoint_sha256': 'test-checkpoint-hash'} for e in GRID}})
    return rule, decisions


def score(decision, exposure, value):
    decision['grid_results'][str(exposure)].update(old_exact_match_event_macro=value,
                                                  within_acquisition_gate=.4 <= value <= .7)


class GridSelectionTests(unittest.TestCase):
    def test_all_nine_must_pass_and_scaled_mean_chooses_common_dose(self):
        rule, decisions = grid_fixture()
        score(decisions[5], 2, .39)
        for decision in decisions[6:]:
            score(decision, 3, .54)
            score(decision, 4, .55)
        result = common_grid_dose(rule, decisions)
        self.assertEqual(result['common_valid_exposures'], [3, 4, 6, 8])
        self.assertEqual(result['selected_E'], 4)
        score(decisions[8], 4, .71)
        self.assertEqual(common_grid_dose(rule, decisions)['selected_E'], 3)

    def test_exact_symmetric_mean_tie_selects_smaller_E(self):
        rule, decisions = grid_fixture()
        for d in decisions:
            for e in GRID:
                score(d, e, .8)
            score(d, 2, .45)
            score(d, 3, .65)
        self.assertEqual(common_grid_dose(rule, decisions)['selected_E'], 2)

    def test_no_common_dose_is_an_explicit_failed_selection(self):
        rule, decisions = grid_fixture()
        for e in GRID:
            score(decisions[e % 9], e, .2)
        result = common_grid_dose(rule, decisions)
        self.assertIsNone(result['selected_E'])
        self.assertEqual(result['common_valid_exposures'], [])

    def test_missing_duplicate_partial_late_or_reclassified_grid_is_rejected(self):
        for mutation in ('missing', 'duplicate', 'partial', 'late', 'naive', 'scale', 'invalid', 'gate'):
            rule, decisions = grid_fixture()
            if mutation == 'missing':
                decisions.pop()
            elif mutation == 'duplicate':
                decisions[-1] = copy.deepcopy(decisions[0])
            elif mutation == 'partial':
                decisions[0]['grid_results'].pop('8')
            elif mutation == 'late':
                rule['frozen_utc'] = '2000-01-03T00:00:00+00:00'
                rehash(rule)
            elif mutation == 'naive':
                decisions[0]['grid_started_utc'] = '2000-01-02T00:00:00'
            elif mutation == 'scale':
                decisions[0]['development_scale_probe'] = True
            elif mutation == 'invalid':
                decisions[0]['grid_results']['2']['old_exact_match_event_macro'] = float('nan')
            else:
                decisions[0]['grid_results']['2']['within_acquisition_gate'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                common_grid_dose(rule, decisions)

    def test_selection_binds_all_decisions_and_exact_checkpoint_even_after_packet_rehash(self):
        rule, decisions = grid_fixture()
        manifest = {'sha256': decisions[0]['manifest_sha256'], 'mode': 'development',
                    'config': {'acquisition_trajectory_policy': GRID_TRAJECTORY_POLICY}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = [root / d['manifest_sha256'] for d in decisions]
            for output, decision in zip(outputs, decisions):
                output.mkdir()
                checkpoint = output / 'stage1-E002.pt'
                checkpoint.write_bytes(b'synthetic complete-state binding test')
                decision['grid_results']['2']['checkpoint_sha256'] = checkpoint_digest(checkpoint)
                write_json(output / 'acquisition_decision.json', decision)
            write_json(root / 'rule.json', rule)
            selected = outputs[0] / 'stage1-E002.pt'
            with patch('spacing_rerun.acquisition_grid.load_prepared', return_value=(manifest, {}, {})):
                packet = choose_acquisition_grid_checkpoint(root / 'prepared', outputs, root / 'rule.json', root / 'selection.json')
                with self.assertRaisesRegex(ValueError, 'never overwrite'):
                    choose_acquisition_grid_checkpoint(root / 'prepared', outputs, root / 'rule.json', root / 'selection.json')
            self.assertEqual(verify_acquisition_grid_selection(root / 'selection.json', manifest, selected), packet)
            mutated = copy.deepcopy(decisions[-1])
            score(mutated, 8, .9)
            write_json(outputs[-1] / 'acquisition_decision.json', mutated)
            with self.assertRaisesRegex(ValueError, 'input grid was modified'):
                verify_acquisition_grid_selection(root / 'selection.json', manifest, selected)
            write_json(outputs[-1] / 'acquisition_decision.json', decisions[-1])
            selected.write_bytes(b'modified checkpoint')
            with self.assertRaisesRegex(ValueError, 'Shared checkpoint differs'):
                verify_acquisition_grid_selection(root / 'selection.json', manifest, selected)
            selected.write_bytes(b'synthetic complete-state binding test')
            other = root / 'different-dose.pt'
            other.write_bytes(selected.read_bytes())
            packet['selected_checkpoint'] = str(other.resolve())
            rehash(packet)
            write_json(root / 'selection.json', packet)
            with self.assertRaisesRegex(ValueError, "grid's frozen dose checkpoint"):
                verify_acquisition_grid_selection(root / 'selection.json', manifest, other)

    def test_modified_selection_rule_is_rejected(self):
        rule, decisions = grid_fixture()
        rule['scaled_mean_target'] = .6
        with self.assertRaisesRegex(ValueError, 'rule was modified'):
            common_grid_dose(rule, decisions)

    def test_arm_guards_run_before_model_initialization(self):
        manifest = {'mode': 'development', 'sha256': 'normal',
                    'config': {'acquisition_trajectory_policy': GRID_TRAJECTORY_POLICY}}
        with patch('spacing_rerun.training.load_prepared', return_value=(manifest, {}, {})), \
             patch('spacing_rerun.training.Session') as session:
            with self.assertRaisesRegex(ValueError, 'requires a frozen common-dose'):
                run_arm('prepared', 'output', 'checkpoint', 'UNI')
            manifest['config']['development_scale_probe'] = True
            with self.assertRaisesRegex(ValueError, 'Stage-1-only'):
                run_arm('prepared', 'output', 'checkpoint', 'UNI')
            session.assert_not_called()

    def test_scale_probe_cannot_enter_spacing_analysis(self):
        with patch('spacing_rerun.analysis.read_json', return_value={'config': {'development_scale_probe': True}}):
            with self.assertRaisesRegex(ValueError, 'cannot enter spacing-arm analysis'):
                summarize_bundles(['synthetic-scale-bundle'], 'output', 84)


class ScaleRoleTests(unittest.TestCase):
    def test_roles_depend_only_on_deduplicated_source_unit_metadata(self):
        events = [f'event_{i:03}' for i in range(20)]
        qa = events[-3:]
        facts = [{'id': f'{event}-{j}', 'event': event, 'statement': f'Source {j}.',
                  'question': 'Opaque question?', 'answer': 'Opaque answer'}
                 for event in events for j in range(1 if event in events[:2] else 3)]
        facts += [dict(facts[0], id='duplicate')]
        partition = {'development': events, 'sha256': 'source-only-partition'}
        counts = {'old': 15, 'new': 1, 'control': 1, 'qa': 3}
        result = assign_development_scale_roles(facts, partition, 'development', counts, qa)
        self.assertEqual(Counter(result['roles'].values()), counts)
        self.assertEqual(result['roles'][events[0]], 'new')
        self.assertEqual(result['roles'][events[1]], 'control')
        self.assertEqual(result['source_unit_counts_by_event'][events[0]], 1)
        changed = copy.deepcopy(facts)
        for fact in changed:
            fact.update(question='Different wording', answer='Different label')
        self.assertEqual(assign_development_scale_roles(changed, partition, 'development', counts, qa)['roles'], result['roles'])
        with self.assertRaisesRegex(ValueError, 'development-only'):
            assign_development_scale_roles(facts, partition, 'confirmation', counts, qa)

    def test_config_freezes_scale_roles_grid_gate_and_original_qa(self):
        config = json.loads((ROOT / 'configs/calibration-sourceqa.json').read_text())
        config.update(acquisition_trajectory_policy=GRID_TRAJECTORY_POLICY, acquisition_exposures=GRID,
                      development_scale_probe=True, development_role_counts={'old': 15, 'new': 1, 'control': 1, 'qa': 3},
                      development_fixed_qa_events=['event_010', 'event_017', 'event_070'])
        validate_config(config)
        for updates in ({'mode': 'confirmation'}, {'development_scale_probe': False},
                        {'development_fixed_qa_events': ['event_010', 'event_017', 'event_071']},
                        {'acquisition_exposures': [2, 4, 8]}, {'acquisition_target_max': .9}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_config(dict(config, **updates))
