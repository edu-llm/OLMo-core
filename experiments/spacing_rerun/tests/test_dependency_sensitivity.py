import copy
import datetime
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from spacing_rerun.acquisition import GRID_TRAJECTORY_POLICY
from spacing_rerun.common import digest, read_json, write_json
from spacing_rerun.dependency_sensitivity import (
    BINDING_SCHEMA, GRID, POLICY_SCHEMA, file_sha256, PRIMARY, SECONDARY, bind_pregrid,
    cluster_metric, grid_sensitivity, seal, validate_membership, validate_policy,
)
from spacing_rerun.evaluation import aggregate
from spacing_rerun.units import GROUNDED_METRICS, GROUNDED_POLICY


def fixture():
    groups, sources, registry, raw_registry, facts, starts = [], [], [], [], [], {}
    for index in range(5):
        event = 'event_a' if index < 4 else 'event_b'
        gid, sid, pid = f'group_{index}', f'unit_{index}', f'probe_{index}'
        text = f'Raw assertion {index}'
        status = 'preserve_source_ambiguity' if index in (0, 2) else 'grounded'
        group = {'id': gid, 'event': event, 'source_unit_ids': [sid], 'statement': text,
                 'review_status': 'unreviewed'}
        source = {'unit_id': sid, 'event': event, 'source_assertion': text,
                  'source_unit_sha256': digest(['source', index]),
                  'canonical_answer_labels': ['gold'], 'status': status}
        groups.append(group)
        sources.append(source)
        registry.append(dict(group, role='old', member_probe_ids=[pid]))
        raw_registry.append({'id': sid, 'event': event, 'role': 'old',
                             'source_statement': text, 'member_probe_ids': [pid]})
        facts.append({'id': pid, 'event': event, 'role': 'old', 'unit_id': gid,
                      'source_unit_id': sid, 'aliases': ['gold'], 'probe': {'answer_ids': [42, 43]}})
        starts[gid] = index
    selected = []
    for group, source in zip(groups[:3], sources[:3]):
        projected = {k: group[k] for k in ('id', 'event', 'source_unit_ids', 'statement')}
        projected['content_sha256'] = digest(projected)
        projected['authored_group_content_sha256'] = digest(group)
        projected['sources'] = [seal(source)]
        # Source projection uses content_sha256, not a top-level artifact seal.
        projected['sources'][0]['content_sha256'] = projected['sources'][0].pop('sha256')
        selected.append(projected)
    authored = seal({'source_catalog_sha256': 'catalog', 'groups': groups, 'source_units': sources})
    policy = seal({'schema': POLICY_SCHEMA, 'frozen_utc': '2000-01-01T00:00:00+00:00',
                   'candidate_exposures': GRID, 'source_catalog_sha256': 'catalog',
                   'authored_chunk_sha256': authored['sha256'], 'dataset_revision': 'dataset',
                   'policy_core_sha256': 'core', 'primary_aggregation': PRIMARY, 'secondary_aggregation': SECONDARY,
                   'diagnostic_only': True, 'dose_selection_permitted': False, 'arm_eligibility_permitted': False,
                   'target_min': .4, 'target_max': .7, 'scale_outcomes_seen_at_freeze': False,
                   'normal_grid_outcomes_seen': True,
                   'dependency_sets': [{'id': 'SET8', 'event': 'event_a', 'role': 'old', 'groups': selected}]})
    grounding_sources = []
    for source, group, raw in zip(sources, groups, raw_registry):
        grounding_sources.append({'unit_id': source['unit_id'], 'event_id': source['event'], 'role': 'old',
                                  'source_statement': source['source_assertion'],
                                  'source_unit_sha256': source['source_unit_sha256'], 'status': source['status'],
                                  'group_id': group['id'], 'probe_ids': raw['member_probe_ids'],
                                  'canonical_answers': source['canonical_answer_labels']})
    grounding = seal({'groups': [dict(group, role='old') for group in groups],
                      'source_units': grounding_sources, 'source_catalog_sha256': 'catalog'})
    schedule = seal({'starts': starts})
    manifest = seal({'schema': 'spacing-rerun-v1', 'created_utc': '2001-01-01T00:00:00+00:00',
                     'mode': 'development', 'dataset_revision': 'dataset',
                     'rehearsal_unit_policy': GROUNDED_POLICY,
                     'config': {'development_scale_probe': True, 'acquisition_trajectory_policy': GRID_TRAJECTORY_POLICY},
                     'schedule_sha256': schedule['sha256'],
                     'grounding_audit': {'bundle_sha256': grounding['sha256']},
                     'acquisition_qa_audit': {'source_catalog_sha256': 'catalog'},
                     'split': {'roles': {'event_a': 'old', 'event_b': 'old'}, 'facts': facts},
                     'unit_registry': {'units': registry}, 'source_unit_registry': {'units': raw_registry},
                     'acquisition_qa_pool': {'records': [{'id': f'target_{i}'} for i in range(5)]}})
    return policy, manifest, schedule, grounding, authored


@contextmanager
def grid_fixture():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        policy, manifest, schedule, grounding, authored = fixture()
        prepared, stage1 = root / 'prepared', root / 'stage1'
        prepared.mkdir()
        stage1.mkdir()
        paths = {'policy': root / 'policy.json', 'prepared': prepared,
                 'source': root / 'source.json', 'receipt': root / 'receipt.json',
                 'binding': root / 'nine-binding.json', 'stage1': stage1, 'output': root / 'report.json'}
        for path, value in [(paths['policy'], policy), (paths['source'], authored),
                            (prepared / 'manifest.json', manifest), (prepared / 'schedule.json', schedule),
                            (prepared / 'grounding-declarations.json', grounding)]:
            write_json(path, value)
        with patch('spacing_rerun.dependency_sensitivity.load_prepared', return_value=(manifest, schedule, {})):
            receipt = bind_pregrid(paths['policy'], prepared, paths['source'], paths['receipt'])
        core = {'sha256': 'core'}
        trial = {'trial_id': 'scaled-01', 'development_scale_probe': True}
        bound = [({'manifest_sha256': manifest['sha256']}, trial, manifest)]
        binding = seal({'schema': 'spacing-acquisition-grid-manifest-binding-v1',
                        'binding_utc': receipt['binding_utc'], 'policy_core_sha256': 'core'})
        write_json(paths['binding'], binding)
        decision = {'schema': manifest['schema'], 'manifest_sha256': manifest['sha256'], 'mode': 'development',
                    'trajectory_policy': GRID_TRAJECTORY_POLICY, 'development_scale_probe': True,
                    'status': 'acquisition_grid_complete', 'selected_E': None, 'acquisition_usable': False,
                    'final_exposures': 8,
                    'grid_started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'policy_core_sha256': 'core', 'policy_manifest_binding_sha256': binding['sha256'],
                    'policy_trial_id': trial['trial_id'], 'grid_results': {}}
        for exposure in GRID:
            rows = [dict(id=fact['id'], unit_id=fact['unit_id'], event=fact['event'], role='old',
                         exact_match=int(index < 3), loss=float(index + 1), prediction='gold' if index < 3 else 'wrong',
                         answer_tokens=2, answer_nll_sum=float(2 * (index + 1)))
                    for index, fact in enumerate(manifest['split']['facts'])]
            old = aggregate(rows)['old']
            evaluation = {'schema': manifest['schema'], 'manifest_sha256': manifest['sha256'],
                          'metric_schema': GROUNDED_METRICS, 'tag': f'acquisition-E{exposure:03}',
                          'epoch': exposure, 'stage2_step': None, 'global_step': exposure * 10,
                          'acquisition_qa_tokens_total': exposure * 90,
                          'variants': {'canonical': {'facts': rows, 'aggregate': {'old': old}}}}
            write_json(stage1 / 'evaluations' / f'acquisition-E{exposure:03}.json', evaluation)
            import torch
            checkpoint = stage1 / f'stage1-E{exposure:03}.pt'
            progress = {'epoch': exposure, 'epoch_cursor': 0, 'status': 'stage1_complete',
                        'global_step': exposure * 10, 'acquisition_qa_tokens_total': exposure * 90,
                        'acquisition_usable': .4 <= old['exact_match_event_macro'] <= .7,
                        'acquisition_qa_target_doses': {f'target_{i}': exposure for i in range(5)},
                        'acquisition_qa_variant_doses': {f'target_{i}': [(exposure + 1) // 2, exposure // 2] for i in range(5)}}
            torch.save({'schema': manifest['schema'], 'manifest_sha256': manifest['sha256'], 'model': {},
                        'optimizer': {}, 'rng': {}, 'progress': progress}, checkpoint)
            decision['grid_results'][str(exposure)] = {
                'exposures': exposure, 'old_exact_match_event_macro': old['exact_match_event_macro'],
                'checkpoint': checkpoint.name, 'checkpoint_sha256': file_sha256(checkpoint),
                'within_acquisition_gate': .4 <= old['exact_match_event_macro'] <= .7}
        write_json(stage1 / 'acquisition_decision.json', decision)
        with patch('spacing_rerun.dependency_sensitivity.load_prepared', return_value=(manifest, schedule, {})), \
             patch('spacing_rerun.dependency_sensitivity.validate_manifest_binding', return_value=(core, bound)):
            yield paths, (policy, manifest, schedule, grounding, authored), receipt


def report(paths):
    return grid_sensitivity(paths['policy'], paths['prepared'], paths['receipt'], paths['binding'],
                            paths['stage1'], paths['output'])


class DependencySensitivityTests(unittest.TestCase):
    def test_three_dependent_groups_count_as_one_cluster_without_filtering_probes(self):
        policy, *_ = fixture()
        rows = [{'unit_id': f'group_{index}', 'event': 'event_a', 'exact_match': int(index < 3)} for index in range(4)]
        self.assertEqual(cluster_metric(rows, policy['dependency_sets'], 'exact_match')['event_macro'], .5)
        self.assertEqual(aggregate([dict(row, role='old') for row in rows])['old']['exact_match_event_macro'], .75)
        rows += [dict(rows[0], exact_match=0), dict(rows[0], exact_match=0)]
        result = cluster_metric(rows, policy['dependency_sets'], 'exact_match')
        self.assertAlmostEqual(result['event_macro'], 7 / 18)
        self.assertEqual(result['cluster_counts_by_event'], {'event_a': 2})

    def test_all_five_doses_have_separate_primary_and_secondary_loss_and_em(self):
        with grid_fixture() as (paths, _, receipt):
            result = report(paths)
            self.assertEqual(len(result['results']), 10)
            self.assertTrue(result['diagnostic_only'])
            self.assertFalse(result['dose_selection_permitted'])
            em, loss = result['results'][:2]
            self.assertEqual((em['primary_event_macro'], em['equal_dependency_cluster_event_macro']), (.375, .25))
            self.assertEqual((loss['primary_event_macro'], loss['equal_dependency_cluster_event_macro']), (3.75, 4.0))
            self.assertEqual(result['placements'][0]['placement'], 'mixed_schedule')
            self.assertFalse(result['placements'][0]['review_arms_executed'])
            self.assertEqual(result['pregrid_binding_sha256'], receipt['sha256'])
            self.assertEqual(len(result['inputs']), 14)
            self.assertGreater(result['verification_measurements']['checkpoint_bytes_stream_hashed'], 0)
            self.assertGreater(result['verification_measurements']['host_peak_rss_bytes'], 0)
            with self.assertRaisesRegex(ValueError, 'never overwrite'):
                report(paths)

    def test_policy_rejects_missing_cross_event_merged_or_duplicated_sources(self):
        for change in ('cross-event', 'missing-unit', 'merged-preserved', 'duplicate', 'late-schema', 'gate'):
            policy, *_ = fixture()
            group = policy['dependency_sets'][0]['groups'][0]
            if change == 'cross-event':
                group['event'] = 'event_b'
            elif change == 'missing-unit':
                group['sources'] = []
            elif change == 'merged-preserved':
                group['source_unit_ids'].append('other')
            elif change == 'duplicate':
                policy['dependency_sets'].append(copy.deepcopy(policy['dependency_sets'][0]))
            elif change == 'late-schema':
                policy['candidate_exposures'] = [4]
            else:
                policy['target_max'] = .8
            policy = seal({k: v for k, v in policy.items() if k != 'sha256'})
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_policy(policy)

    def test_prepared_membership_cannot_move_or_drop_dependency_raw_units(self):
        for change in ('wrong-raw', 'missing-raw', 'role', 'event', 'probe', 'missing-group', 'gold', 'sourcehash'):
            policy, manifest, schedule, grounding, _ = fixture()
            if change == 'wrong-raw':
                manifest['source_unit_registry']['units'][0]['source_statement'] = 'Rewritten assertion'
            elif change == 'missing-raw':
                manifest['source_unit_registry']['units'].pop(0)
            elif change == 'role':
                manifest['split']['roles']['event_a'] = 'control'
            elif change == 'event':
                manifest['unit_registry']['units'][0]['event'] = 'event_b'
            elif change == 'probe':
                manifest['unit_registry']['units'][0]['member_probe_ids'] = []
            elif change == 'missing-group':
                manifest['unit_registry']['units'].pop(0)
            elif change == 'gold':
                grounding['source_units'][0]['canonical_answers'] = ['changed']
            else:
                grounding['source_units'][0]['source_unit_sha256'] = 'changed'
            grounding = seal({k: v for k, v in grounding.items() if k != 'sha256'})
            manifest['grounding_audit']['bundle_sha256'] = grounding['sha256']
            manifest = seal({k: v for k, v in manifest.items() if k != 'sha256'})
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_membership(policy, manifest, schedule, grounding)

    def test_late_binding_or_policy_mutation_cannot_be_attached_after_training(self):
        for change in ('late', 'future', 'policy', 'decision-manifest', 'binding', 'trial', 'missing-dose'):
            with self.subTest(change=change), grid_fixture() as (paths, _, _):
                decision_path = paths['stage1'] / 'acquisition_decision.json'
                decision = read_json(decision_path)
                if change == 'late':
                    decision['grid_started_utc'] = '2002-01-01T00:00:00+00:00'
                elif change == 'future':
                    decision['grid_started_utc'] = '2099-01-01T00:00:00+00:00'
                elif change == 'policy':
                    policy = read_json(paths['policy'])
                    policy['frozen_utc'] = '2099-01-01T00:00:00+00:00'
                    write_json(paths['policy'], seal({k: v for k, v in policy.items() if k != 'sha256'}))
                elif change == 'decision-manifest':
                    decision['manifest_sha256'] = 'another'
                elif change == 'binding':
                    decision['policy_manifest_binding_sha256'] = 'another'
                elif change == 'trial':
                    decision['policy_trial_id'] = 'normal-01'
                else:
                    decision['grid_results'].pop('8')
                write_json(decision_path, decision)
                with self.assertRaises(ValueError):
                    report(paths)
                self.assertFalse(paths['output'].exists())

    def test_evaluation_membership_behavior_and_primary_scores_are_verified(self):
        for change in ('missing', 'duplicate', 'unit', 'event', 'manifest', 'primary', 'gate', 'em', 'nll', 'nan', 'tokens'):
            with self.subTest(change=change), grid_fixture() as (paths, _, _):
                path = paths['stage1'] / 'evaluations' / 'acquisition-E002.json'
                evaluation = read_json(path)
                canonical = evaluation['variants']['canonical']
                if change == 'missing':
                    canonical['facts'].pop()
                elif change == 'duplicate':
                    canonical['facts'].append(copy.deepcopy(canonical['facts'][0]))
                elif change == 'unit':
                    canonical['facts'][0]['unit_id'] = 'other'
                elif change == 'event':
                    canonical['facts'][0]['event'] = 'other'
                elif change == 'manifest':
                    evaluation['manifest_sha256'] = 'other'
                elif change == 'primary':
                    canonical['aggregate']['old']['loss_event_macro'] += .1
                elif change == 'gate':
                    decision_path = paths['stage1'] / 'acquisition_decision.json'
                    decision = read_json(decision_path)
                    decision['grid_results']['2']['within_acquisition_gate'] = True
                    write_json(decision_path, decision)
                elif change == 'em':
                    canonical['facts'][0]['exact_match'] = 0
                elif change == 'nll':
                    canonical['facts'][0]['answer_nll_sum'] += 1
                elif change == 'nan':
                    canonical['facts'][0]['loss'] = None
                else:
                    canonical['facts'][0]['answer_tokens'] = 1
                write_json(path, evaluation)
                with self.assertRaises(ValueError):
                    report(paths)
                self.assertFalse(paths['output'].exists())

    def test_resealed_receipt_cannot_omit_or_substitute_bound_inputs(self):
        for change in ('reduced', 'source', 'manifest', 'schedule', 'grounding'):
            with self.subTest(change=change), grid_fixture() as (paths, _, receipt):
                if change == 'reduced':
                    receipt['inputs'] = receipt['inputs'][:1]
                else:
                    index = {'source': 4, 'manifest': 1, 'schedule': 2, 'grounding': 3}[change]
                    other = paths['output'].with_name('substituted.json')
                    write_json(other, seal({'source_catalog_sha256': 'catalog', 'groups': [], 'source_units': []}))
                    receipt['inputs'][index] = {'path': str(other.resolve()), 'file_sha256': file_sha256(other)}
                write_json(paths['receipt'], seal({k: v for k, v in receipt.items() if k != 'sha256'}))
                with self.assertRaises(ValueError):
                    report(paths)

    def test_checkpoint_hash_fullstate_and_exposure_clocks_are_verified(self):
        import torch
        for change in ('file', 'manifest', 'fullstate', 'step', 'tokens', 'epoch', 'cursor', 'target-dose', 'variant-dose'):
            with self.subTest(change=change), grid_fixture() as (paths, _, _):
                path = paths['stage1'] / 'stage1-E002.pt'
                if change == 'file':
                    path.write_bytes(b'changed checkpoint')
                else:
                    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
                    if change == 'manifest':
                        state['manifest_sha256'] = 'another'
                    elif change == 'fullstate':
                        state.pop('rng')
                    elif change == 'step':
                        state['progress']['global_step'] += 1
                    elif change == 'tokens':
                        state['progress']['acquisition_qa_tokens_total'] += 1
                    elif change == 'epoch':
                        state['progress']['epoch'] = 3
                    elif change == 'cursor':
                        state['progress']['epoch_cursor'] = 1
                    elif change == 'target-dose':
                        state['progress']['acquisition_qa_target_doses']['target_0'] = 1
                    else:
                        state['progress']['acquisition_qa_variant_doses']['target_0'] = [2, 0]
                    torch.save(state, path)
                    decision_path = paths['stage1'] / 'acquisition_decision.json'
                    decision = read_json(decision_path)
                    decision['grid_results']['2']['checkpoint_sha256'] = file_sha256(path)
                    write_json(decision_path, decision)
                with self.assertRaises(ValueError):
                    report(paths)

    def test_binding_is_actual_time_hash_bound_and_never_overwrites(self):
        with grid_fixture() as (paths, data, receipt):
            policy, manifest, schedule, *_ = data
            self.assertEqual(receipt['schema'], BINDING_SCHEMA)
            self.assertGreater(datetime.datetime.fromisoformat(receipt['binding_utc']),
                               datetime.datetime.fromisoformat(policy['frozen_utc']))
            with self.assertRaisesRegex(ValueError, 'never overwrite'):
                bind_pregrid(paths['policy'], paths['prepared'], paths['source'], paths['receipt'])
            altered = read_json(paths['source'])
            altered['groups'][0]['statement'] = 'Changed source'
            write_json(paths['source'], altered)
            with self.assertRaisesRegex(ValueError, 'input file was modified'):
                report(paths)


if __name__ == '__main__':
    unittest.main()
