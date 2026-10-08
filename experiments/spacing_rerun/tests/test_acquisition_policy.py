import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spacing_rerun.acquisition import GRID_TRAJECTORY_POLICY
from spacing_rerun.acquisition_grid import (common_grid_dose, checkpoint_digest, choose_acquisition_grid_checkpoint,
                                           verify_acquisition_grid_selection)
from spacing_rerun.acquisition_policy import (BINDING_SCHEMA, CORE_SCHEMA, RULE_POLICY, bind_policy_manifests,
                                               config_frame, manifest_policy_trial, resolve_manifest_binding,
                                               validate_policy_core)
from spacing_rerun.common import digest, write_json
from spacing_rerun.data import REVISION
from spacing_rerun.training import run_acquisition
from test_acquisition_grid import grid_fixture, rehash


ROOT = Path(__file__).resolve().parents[1]


def policy_fixture():
    base = json.loads((ROOT / 'configs/calibration-sourceqa.json').read_text())
    base.update(span=252, acquisition_exposures=[2, 3, 4, 6, 8],
                acquisition_trajectory_policy=GRID_TRAJECTORY_POLICY)
    normal_roles = {f'{role}-{i}': role for role, count in [('old', 5), ('new', 7), ('control', 5), ('qa', 3)]
                    for i in range(count)}
    scale_roles = {f'{role}-{i}': role for role, count in [('old', 15), ('new', 1), ('control', 1), ('qa', 3)]
                   for i in range(count)}
    core = {'schema': CORE_SCHEMA, 'policy': RULE_POLICY, 'frozen_utc': '2000-01-01T00:00:00+00:00',
            'candidate_exposures': [2, 3, 4, 6, 8], 'target_min': .4, 'target_max': .7,
            'scaled_mean_target': .55, 'source_catalog_sha256': 'catalog-hash', 'dataset_revision': REVISION,
            'trials': []}
    manifests = {}
    rule, decisions = grid_fixture()
    rows = []
    for i, decision in enumerate(decisions):
        scaled = i >= 6
        index = i - 6 if scaled else i
        config = dict(base, order_seed=(3701 if scaled else 2701) + index,
                      train_seed=(3801 if scaled else 2801) + index, eval_seed=(3901 if scaled else 2901) + index)
        if scaled:
            config.update(development_scale_probe=True,
                          development_role_counts={'old': 15, 'new': 1, 'control': 1, 'qa': 3},
                          development_fixed_qa_events=['event_010', 'event_017', 'event_070'])
        roles = scale_roles if scaled else normal_roles
        source = {'source_catalog_sha256': 'catalog-hash'} if scaled else {
            'grounding_bundle_sha256': 'grounding-hash', 'acquisition_qa_bundle_sha256': 'qa-hash',
            'acquisition_source_input_sha256': 'source-input-hash'}
        trial_id = decision['manifest_sha256']
        core['trials'].append({'trial_id': trial_id, 'development_scale_probe': scaled,
                               'config_frame': config_frame(config), 'roles_sha256': digest(roles),
                               'source_frame': source})
        manifest = {'sha256': trial_id, 'mode': 'development', 'dataset_revision': REVISION,
                    'config': config, 'split': {'roles': roles},
                    'acquisition_qa_audit': {'source_catalog_sha256': 'catalog-hash',
                                             'bundle_sha256': 'qa-hash', 'source_input_sha256': 'source-input-hash'},
                    'grounding_audit': {'bundle_sha256': 'grounding-hash'}}
        path = '/synthetic/' + trial_id
        manifests[path] = manifest
        rows.append({'trial_id': trial_id, 'manifest_sha256': trial_id, 'prepared_path': path})
        decision['policy_trial_id'] = trial_id
        decision['grid_started_utc'] = ('2000-01-04T00:00:00+00:00' if scaled else '2000-01-02T00:00:00+00:00')
    rehash(core)
    for decision in decisions:
        decision['policy_core_sha256'] = core['sha256']
    binding = {'schema': BINDING_SCHEMA, 'policy_core': core, 'policy_core_sha256': core['sha256'],
               'binding_utc': '2000-01-03T00:00:00+00:00', 'trials': rows}
    rehash(binding)
    for decision in decisions[6:]:
        decision['policy_manifest_binding_sha256'] = binding['sha256']
    return core, binding, decisions, manifests


def prepared_loader(manifests):
    return lambda path: (manifests[str(path)], {}, {})


class PolicyFreezeTests(unittest.TestCase):
    def test_normal_grids_can_start_before_later_manifest_binding(self):
        core, binding, decisions, manifests = policy_fixture()
        validate_policy_core(core)
        with patch('spacing_rerun.acquisition_policy.load_prepared', side_effect=prepared_loader(manifests)):
            resolved = resolve_manifest_binding(binding, decisions)
            self.assertEqual(resolved['frozen_utc'], core['frozen_utc'])
            self.assertEqual(len(resolved['normal_manifest_sha256']), 6)
            self.assertEqual(common_grid_dose(binding, decisions)['selected_E'], 2)

    def test_late_binding_cannot_cover_an_already_started_scale_grid(self):
        core, binding, decisions, manifests = policy_fixture()
        decisions[-1]['grid_started_utc'] = '2000-01-02T00:00:00+00:00'
        with patch('spacing_rerun.acquisition_policy.load_prepared', side_effect=prepared_loader(manifests)):
            with self.assertRaisesRegex(ValueError, 'Scale grid started before actual manifest binding'):
                resolve_manifest_binding(binding, decisions)

    def test_prelaunch_core_source_seed_role_or_settings_mismatch_rejects_manifest(self):
        for mutation in ('seed', 'lr', 'extra', 'role', 'source', 'catalog', 'dataset'):
            core, binding, decisions, manifests = policy_fixture()
            manifest = copy.deepcopy(manifests['/synthetic/' + ('scaled-0' if mutation == 'catalog' else 'normal-0')])
            if mutation == 'seed':
                manifest['config']['train_seed'] = 999
            elif mutation == 'lr':
                manifest['config']['learning_rate'] = .001
            elif mutation == 'extra':
                manifest['config']['unfrozen_change'] = True
            elif mutation == 'role':
                manifest['split']['roles']['old-0'] = 'new'
            elif mutation == 'source':
                manifest['acquisition_qa_audit']['bundle_sha256'] = 'different-reviewed-qa'
            elif mutation == 'catalog':
                manifest['acquisition_qa_audit']['source_catalog_sha256'] = 'different-source-catalog'
            else:
                manifest['dataset_revision'] = 'different-dataset'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                manifest_policy_trial(manifest, core)

    def test_modified_core_binding_or_missing_prelaunch_identity_is_rejected(self):
        for mutation in ('core', 'core_hash', 'binding_hash', 'duplicate', 'missing_core', 'wrong_trial', 'early_start'):
            core, binding, decisions, manifests = policy_fixture()
            if mutation == 'core':
                binding['policy_core']['scaled_mean_target'] = .6
                rehash(binding)
            elif mutation == 'core_hash':
                binding['policy_core_sha256'] = 'different-core'
                rehash(binding)
            elif mutation == 'binding_hash':
                binding['binding_utc'] = '2000-01-02T12:00:00+00:00'
            elif mutation == 'duplicate':
                binding['trials'][-1] = copy.deepcopy(binding['trials'][0])
                rehash(binding)
            elif mutation == 'missing_core':
                decisions[0].pop('policy_core_sha256')
            elif mutation == 'wrong_trial':
                decisions[0]['policy_trial_id'] = 'normal-1'
            else:
                decisions[0]['grid_started_utc'] = '1999-12-31T23:59:59+00:00'
            with patch('spacing_rerun.acquisition_policy.load_prepared', side_effect=prepared_loader(manifests)):
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    resolve_manifest_binding(binding, decisions)

    def test_binding_builder_uses_actual_time_and_cannot_overwrite_or_omit_trials(self):
        core, binding, decisions, manifests = policy_fixture()
        with tempfile.TemporaryDirectory() as directory, \
             patch('spacing_rerun.acquisition_policy.load_prepared', side_effect=prepared_loader(manifests)):
            output = Path(directory) / 'binding.json'
            created = bind_policy_manifests(core, list(manifests), output)
            self.assertGreater(created['binding_utc'], '2000-01-03')
            self.assertEqual(created['policy_core_sha256'], core['sha256'])
            self.assertEqual(len(created['trials']), 9)
            with self.assertRaisesRegex(ValueError, 'never overwrite'):
                bind_policy_manifests(core, list(manifests), output)
            with self.assertRaisesRegex(ValueError, 'every predeclared'):
                bind_policy_manifests(core, list(manifests)[:-1], Path(directory) / 'partial.json')

    def test_prelaunch_policy_validation_precedes_model_initialization(self):
        core, binding, decisions, manifests = policy_fixture()
        manifest = copy.deepcopy(manifests['/synthetic/normal-0'])
        manifest['config']['train_seed'] = 100
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'core.json'
            write_json(path, core)
            with patch('spacing_rerun.training.load_prepared', return_value=(manifest, {}, {})), \
                 patch('spacing_rerun.training.Session') as session:
                with self.assertRaisesRegex(ValueError, 'settings/seeds differ'):
                    run_acquisition('prepared', 'output', acquisition_policy_core=path)
                session.assert_not_called()

    def test_scale_requires_actual_binding_before_model_initialization(self):
        core, binding, decisions, manifests = policy_fixture()
        manifest = manifests['/synthetic/scaled-0']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'core.json'
            write_json(path, core)
            with patch('spacing_rerun.training.load_prepared', return_value=(manifest, {}, {})), \
                 patch('spacing_rerun.training.Session') as session:
                with self.assertRaisesRegex(ValueError, 'require the actual nine-manifest binding'):
                    run_acquisition('prepared', 'output', acquisition_policy_core=path)
                session.assert_not_called()

    def test_selection_and_continuation_verification_revalidate_the_two_stage_binding(self):
        core, binding, decisions, manifests = policy_fixture()
        manifest = manifests['/synthetic/normal-0']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = []
            for decision in decisions:
                output = root / decision['manifest_sha256']
                output.mkdir()
                checkpoint = output / 'stage1-E002.pt'
                checkpoint.write_bytes(b'synthetic checkpoint provenance fixture')
                decision['grid_results']['2']['checkpoint_sha256'] = checkpoint_digest(checkpoint)
                write_json(output / 'acquisition_decision.json', decision)
                outputs.append(output)
            write_json(root / 'binding.json', binding)
            selected = outputs[0] / 'stage1-E002.pt'
            with patch('spacing_rerun.acquisition_grid.load_prepared', return_value=(manifest, {}, {})), \
                 patch('spacing_rerun.acquisition_policy.load_prepared', side_effect=prepared_loader(manifests)):
                packet = choose_acquisition_grid_checkpoint('prepared', outputs, root / 'binding.json', root / 'selection.json')
                self.assertEqual(packet['selection_rule_sha256'], binding['sha256'])
                self.assertEqual(packet['selected_E'], 2)
                self.assertEqual(verify_acquisition_grid_selection(root / 'selection.json', manifest, selected), packet)
                manifests['/synthetic/scaled-2']['config']['learning_rate'] = .0002
                with self.assertRaisesRegex(ValueError, 'settings/seeds differ'):
                    verify_acquisition_grid_selection(root / 'selection.json', manifest, selected)
