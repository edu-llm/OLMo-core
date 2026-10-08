"""Labeled structural fixtures. Synthetic reports do not approve production runs."""
import copy
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spacing_rerun.common import digest, write_json
from spacing_rerun.confirmation import (PROTOCOL, NUMERICAL_SCOPE, AA_CONFIG_KEYS, build_aa_compatibility,
                                       current_numerical_hashes, verify_confirmation_preregistration)
from spacing_rerun.analysis import protected_cluster_loss
from spacing_rerun.units import GROUNDED_POLICY, GROUNDED_METRICS
from spacing_rerun.acquisition import SOURCE_QA_ACQUISITION_POLICY
from spacing_rerun.teaching import SOURCE_QA_POLICY
from test_confirmation import confirmation_fixture, seal


def prereg_fixture(directory, manifest=None):
    directory = Path(directory)
    facts, audit = confirmation_fixture()
    model = 'explicit-synthetic-test-model'
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/calibration-sourceqa.json').read_text())
    config.update(split_seed=2026100801, order_seed=4701, train_seed=4801, eval_seed=4901,
                  model_revision='test-model-revision', max_attempts=3)
    manifest = manifest or seal({'confirmation_protocol_schema': PROTOCOL, 'audit_complete': True,
        'rehearsal_unit_policy': GROUNDED_POLICY, 'confirmation_audit': audit,
        'split': {'facts': facts, 'roles': {}}, 'model': model,
        'model_revision': config['model_revision'], 'tokenizer_sha256': 'test-tokenizer', 'config': config,
        'test_fixture': True})
    reference = seal({'model': manifest['model'], 'model_revision': manifest['model_revision'],
                      'tokenizer_sha256': manifest['tokenizer_sha256'], 'config': copy.deepcopy(manifest['config']), 'test_fixture': True})
    reference_path = directory / 'aa-prepared' / 'manifest.json'
    write_json(reference_path, reference)
    report = {'schema': 'spacing-aa-replay-v1', 'passed': True, 'status': 'passed', 'device': 'cuda',
              'source_artifacts_unchanged': True, 'gpu': 'SYNTHETIC FIXTURE GPU', 'manifest_sha256': reference['sha256'],
              'code_commit': 'a' * 40, 'prepared': str(reference_path.parent),
              'bound_prepared_artifacts': {'manifest.json': {'sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest()}},
              'software': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'ai2-olmo', 'numpy')},
              'comparisons': {name: {'bitwise_equal': True, 'mismatch_count': 0} for name in
                ('model', 'optimizer', 'rng', 'progress_and_clocks', 'training_metrics', 'probe_observations', 'realized_exposures')},
              'test_fixture': True}
    write_json(directory / 'aa-report.json', report)
    compatibility = seal({'schema': 'spacing-aa-code-compatibility-v1', 'aa_report_sha256': digest(report),
        'aa_code_commit': report['code_commit'], 'launch_code_commit': 'b' * 40,
        'method': 'actual_git_function_source_identity',
        'numerical_scope': {key: list(value) for key, value in NUMERICAL_SCOPE.items()},
        'function_sha256': current_numerical_hashes(), 'reference_manifest_path': str(reference_path),
        'reference_manifest_sha256': reference['sha256'],
        'reference_manifest_file_sha256': report['bound_prepared_artifacts']['manifest.json']['sha256'],
        'reference_numerical_config':{key:reference['config'][key] for key in AA_CONFIG_KEYS},
        'backend': {'model': reference['model'], 'model_revision': reference['model_revision'],
                    'tokenizer_sha256': reference['tokenizer_sha256'], 'software': report['software'],
                    'device': report['device'], 'gpu': report['gpu'],
                    'load_backend': 'hf_olmo_transformers_fp32_parameters_bf16_cuda_autocast_deterministic_no_tf32'},
        'test_fixture': True})
    precision = {'chosen_n': 2, 'claim': 'estimation', 'power_target_met': False, 'test_fixture': True}
    final_span = seal({'assay_usable': True, 'selected_E': 4, 'cost': {'accounting_complete': True},
                      'precision_plan': precision, 'planning_policy_sha256': 'synthetic-policy', 'test_fixture': True})
    cfg = manifest['config']
    designs = [{'manifest_sha256': manifest['sha256'], **{k: cfg[k] for k in ('split_seed','order_seed','train_seed','eval_seed')},
                'roles_sha256': digest(manifest['split']['roles'])},
               {'manifest_sha256': 'other-test-replicate', 'split_seed': 2026100802,
                'order_seed':4702,'train_seed':4802,'eval_seed':4902,'roles_sha256':'test-other-roles'}]
    prereg = seal({'schema': 'spacing-confirmation-preregistration-v1', 'confirmation_protocol_schema': PROTOCOL,
        'rehearsal_unit_policy': GROUNDED_POLICY, 'metric_schema': GROUNDED_METRICS,
        'acquisition_policy': SOURCE_QA_ACQUISITION_POLICY, 'qa_teaching_policy': SOURCE_QA_POLICY,
        'confirmation_audit_sha256': manifest['confirmation_audit']['sha256'],
        'source_catalog_sha256': manifest['confirmation_audit']['source_catalog']['sha256'], 'human_review_complete': False,
        'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'final_span_report_path': 'explicitly-mocked-synthetic-final-span.json', 'final_span_report_sha256': final_span['sha256'],
        'E':4,'n':2,'inference_claim':'estimation','planning_policy_sha256': final_span['planning_policy_sha256'],
        'power_scenarios':precision,'loss_margin':.02,'primary_delay':84,'common_span':252,'buffer_steps':168,
        'cohort_size':2,'review_cap':8,'secondary_holm_family':['H1','H3','HG'],
        'missing_pair_rule':'report incomplete fixed design; no pair deletion','maximum_attempts':3,
        'event_cluster_sensitivity_policy':'same_role_components_as_single_cluster_then_equal_clusters_v1',
        'same_role_components':[['event_000','event_071']], 'model_revision':manifest['model_revision'],
        'tokenizer_sha256':manifest['tokenizer_sha256'], 'measured_cost':{'final_span_report_sha256':final_span['sha256'],
            'confirmation_projection':{'test_fixture':True},'content_scale_limitations':'Explicit synthetic fixture'},
        'replicate_manifest_sha256':[row['manifest_sha256'] for row in designs],'replicate_designs':designs,
        'aa_report_path':str(directory/'aa-report.json'),'aa_report_sha256':digest(report),
        'hardware':{'gpu':report['gpu']},'aa_code_compatibility':compatibility,'code_commit':'b'*40,'test_fixture':True})
    return manifest, prereg, final_span


class ConfirmationPreregTests(unittest.TestCase):
    def test_complete_labeled_fixture_and_missing_actual_report(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, prereg, final_span = prereg_fixture(directory)
            with patch('spacing_rerun.final_span.verify_final_span_report', return_value=final_span):
                self.assertEqual(verify_confirmation_preregistration(prereg, manifest, check_code=False), prereg)
            with self.assertRaises(FileNotFoundError):
                verify_confirmation_preregistration(prereg, manifest, check_code=False)

    def test_mutations_do_not_bypass_protocol_provenance_or_fixed_design(self):
        for mutation in ('human', 'audit', 'n', 'E', 'claim', 'cluster', 'seeds', 'train-seeds', 'order-seeds', 'eval-seeds',
                         'microbatch', 'eval-batch', 'model', 'software', 'numerical', 'allocation', 'assay', 'future'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                manifest, prereg, final_span = prereg_fixture(directory)
                if mutation == 'human': prereg['human_review_complete'] = True
                elif mutation == 'audit': prereg['confirmation_audit_sha256'] = 'changed'
                elif mutation == 'n': prereg['n'] = 3
                elif mutation == 'E': prereg['E'] = 6
                elif mutation == 'claim': prereg['inference_claim'] = 'equivalence_and_useful_direction_planning'
                elif mutation == 'cluster': prereg['same_role_components'] = []
                elif mutation == 'seeds': prereg['replicate_designs'][1]['split_seed'] = 2026100801
                elif mutation in ('train-seeds', 'order-seeds', 'eval-seeds'):
                    key=mutation.replace('-seeds','_seed');prereg['replicate_designs'][1][key]=prereg['replicate_designs'][0][key]
                elif mutation == 'microbatch': manifest['config']['microbatch_size'] = 4
                elif mutation == 'eval-batch': manifest['config']['eval_batch_size'] = 8
                elif mutation == 'model': manifest['model_revision'] = 'different-actual-model'
                elif mutation == 'software': prereg['aa_code_compatibility']['backend']['software']['torch'] = '0.0-invalid'
                elif mutation == 'numerical': prereg['aa_code_compatibility']['function_sha256']['training.py']['train_update'] = 'changed'
                elif mutation == 'allocation': final_span['cost']['accounting_complete'] = False
                elif mutation == 'assay': final_span['assay_usable'] = False
                else: prereg['frozen_utc'] = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)).isoformat()
                seal(prereg['aa_code_compatibility']); seal(prereg)
                with patch('spacing_rerun.final_span.verify_final_span_report', return_value=final_span), self.assertRaises(ValueError):
                    verify_confirmation_preregistration(prereg, manifest, check_code=False)

    def test_actual_git_builder_binds_reference_manifest_and_detects_numerical_change(self):
        package = Path(__file__).resolve().parents[1] / 'spacing_rerun'
        with tempfile.TemporaryDirectory() as directory:
            manifest, prereg, _ = prereg_fixture(directory)
            report = json.loads((Path(directory) / 'aa-report.json').read_text())
            def actual_source(args, **kwargs):
                return (package / args[-1].split('/')[-1]).read_text()
            with patch('spacing_rerun.confirmation.subprocess.check_output', side_effect=actual_source):
                packet = build_aa_compatibility(report, 'b'*40, directory)
                self.assertEqual(packet['backend']['model_revision'], manifest['model_revision'])
                self.assertEqual(packet['function_sha256'], current_numerical_hashes())
            def changed_source(args, **kwargs):
                text = actual_source(args)
                if args[-1].startswith('b'*40) and args[-1].endswith('training.py'):
                    text = text.replace('eps=1e-8', 'eps=1e-7')
                return text
            with patch('spacing_rerun.confirmation.subprocess.check_output', side_effect=changed_source), self.assertRaisesRegex(ValueError, 'implementations changed'):
                build_aa_compatibility(report, 'b'*40, directory)

    def test_event_cluster_average_and_partial_component_rejection(self):
        rows = [{'event':'A','role':'old','unit_id':'a','loss':0},
                {'event':'A','role':'old','unit_id':'a','loss':2},
                {'event':'A','role':'old','unit_id':'b','loss':5},
                {'event':'B','role':'old','unit_id':'c','loss':9},
                {'event':'C','role':'old','unit_id':'d','loss':0}]
        result = protected_cluster_loss(rows, [['A','B']])
        self.assertEqual(result['loss_equal_cluster_macro'],3)
        self.assertEqual(result['clusters'],2)
        with self.assertRaisesRegex(ValueError,'crosses roles'):
            protected_cluster_loss(rows[:-2]+rows[-1:], [['A','B']])


def prepared_confirmation_fixture(directory, split_seed=2026100801):
    """Real guarded preparation from explicitly synthetic 80-event reviewed data."""
    from spacing_rerun.confirmation import assign_confirmation_roles, validate_confirmation_audit
    from spacing_rerun.grounding import FACTSHEETS_SHA256, build_grounded_registry
    from spacing_rerun.units import build_units, schedule_records
    from spacing_rerun.teaching import build_teaching_pool
    from spacing_rerun.acquisition import acquisition_similarity
    from spacing_rerun.encoding import build_examples
    from spacing_rerun.schedule import compile_schedule
    from spacing_rerun.data import REVISION
    import re

    class RuntimeTokenizer:
        eos_token_id = 0
        def encode(self, text, add_special_tokens=False):
            return [1 + int(digest(word)[:8], 16) % 126 for word in re.findall(r'\n|[\w]+|[^\w\s]', text)]
        def decode(self, ids, skip_special_tokens=False):
            return 'unmatched synthetic-test generated answer'

    facts, audit = confirmation_fixture()
    context = validate_confirmation_audit(audit, source_facts=facts)
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/calibration-sourceqa.json').read_text())
    config.update(mode='confirmation', confirmation_protocol_schema=PROTOCOL, confirmation_audit='fixture-audit.json',
        split_seed=split_seed, order_seed=4701 + split_seed - 2026100801,
        train_seed=4801 + split_seed - 2026100801, eval_seed=4901 + split_seed - 2026100801, span=252,
        acquisition_exposures=[4], max_answer_tokens=1, microbatch_size=2, model_revision='test-model-revision')
    split = assign_confirmation_roles(facts, audit['source_catalog']['partition'], config['split_seed'], context['same_role_components'])
    facts = split['facts']; roles = split['roles']
    by_eval = {row['id']: row for row in context['evaluation_records']}
    for fact in facts:
        fact['paraphrase'] = by_eval[fact['id']]['paraphrase']
        fact['mcq'] = {'choices':[fact['answer'], 'incorrect distractor'], 'correct':0}
    source_registry = build_units(facts)
    by_source = {row['unit_id']: row for row in context['sources']}
    selected = [unit for unit in source_registry['units'] if unit['role'] in ('old','new')]
    groups, sources = [], []
    original_groups = {row['id']: row for row in context['groups']}
    for index, unit in enumerate(selected):
        authored = by_source[unit['id']]
        group = dict(original_groups[authored['group_id']], role=unit['role'], source_indices=[index], review_status='approved_confirmation')
        groups.append(group)
        row = dict(index=index, unit_id=unit['id'],event_id=unit['event'],role=unit['role'],
            source_statement=unit['statement'],canonical_answers=[by_eval[pid]['canonical_answer'] for pid in unit['member_probe_ids']],
            probe_ids=unit['member_probe_ids'],group_id=group['id'],trained_declaration=group['statement'],
            status=authored['status'],conflict_rationale=authored['conflict_rationale'],
            nonliteral_answer_labels=authored['nonliteral_answer_labels'],evidence=authored['evidence'],
            factsheet_sha256=audit['source_catalog']['factsheets'][unit['event']]['sha256'],review_status='approved_confirmation',
            source_statement_sha256=digest(unit['statement']))
        row['source_fields_sha256'] = digest({key:row[key] for key in ('unit_id','event_id','role','source_statement','canonical_answers','probe_ids')})
        sources.append(row)
    provenance = dict(confirmation_audit_sha256=audit['sha256'],source_catalog_sha256=audit['source_catalog']['sha256'],human_review_complete=False)
    grounding = seal(dict(schema='spacing-grounded-declarations-v1',policy=GROUNDED_POLICY,
        source_dataset_revision=REVISION,source_factsheets_sha256=FACTSHEETS_SHA256,
        partition_sha256=audit['source_catalog']['partition']['sha256'],roles_sha256=digest(roles),
        independent_agent_review_complete=True,independent_agent_reviewer='Explicit synthetic fixture receipt',
        authoring_inputs=['source_statement','canonical_answers','event_factsheet'],evaluation_question_text_included=False,
        model_outcomes_used=False,original_evaluation_probe_count=len(facts),source_unit_count=len(sources),
        factsheets={unit['event']:audit['source_catalog']['factsheets'][unit['event']]['text'] for unit in selected},
        groups=groups,source_units=sources,**provenance))
    registry = build_grounded_registry(facts,source_registry,grounding)
    rows = [dict(event_id=fact['event'],question_id=fact['id'],question=fact['question'],natural_answer=fact['answer'],fiction_id='fixture') for fact in facts]
    teaching = build_teaching_pool(rows,facts,roles,'confirmation',confirmation_audit=audit)
    source_input_units = []
    ground_sources = {row['unit_id']:row for row in sources}
    for unit in registry['units']:
        if unit['role'] == 'old':
            source_input_units.append(dict(unit_id=unit['id'],event=unit['event'],statement=unit['statement'],
                answer_targets=sorted({answer for sid in unit['source_unit_ids'] for answer in ground_sources[sid]['canonical_answers']}),
                source_assertions=[dict(statement=ground_sources[sid]['source_statement'],answer_targets=sorted(set(ground_sources[sid]['canonical_answers'])),
                    status=ground_sources[sid]['status'],evidence=ground_sources[sid]['evidence']) for sid in unit['source_unit_ids']]))
    source_input = seal(dict(schema='source-only-acquisition-authoring-input-v1',dataset_revision=REVISION,
        grounding_bundle_sha256=grounding['sha256'],units=source_input_units,
        factsheets={unit['event']:grounding['factsheets'][unit['event']] for unit in source_input_units},**provenance))
    records = [dict(row,review_status='approved_confirmation') for row in context['acquisition_records'] if roles[row['event']] == 'old']
    acquisition = seal(dict(schema='spacing-acquisition-qa-v1',mode='confirmation',grounding_bundle_sha256=grounding['sha256'],
        source_input_sha256=source_input['sha256'],authoring_inputs=['source_assertions','source_answer_labels','event_factsheets'],
        evaluation_question_text_included=False,model_outcomes_used=False,independent_agent_review_complete=True,
        independent_agent_reviewer='Explicit synthetic fixture receipt',records=records,**provenance))
    examples = build_examples(facts,RuntimeTokenizer(),list(range(1,128))*100,config,qa_teaching_records=teaching['records'],acquisition_records=records)
    split['facts']=facts; seal(split)
    schedule = compile_schedule(schedule_records(facts,registry,teaching['records']),config)
    manifest = seal(dict(mode='confirmation',config=config,split=split,partition=audit['source_catalog']['partition'],
        rehearsal_unit_policy=GROUNDED_POLICY,unit_registry=registry,source_unit_registry=source_registry,
        schedule_sha256=schedule['sha256'],examples_sha256=digest(examples),generic_eval=[],
        grounding_audit={'bundle_sha256':grounding['sha256'],'source_factsheets_sha256':FACTSHEETS_SHA256},
        qa_teaching_pool=teaching,acquisition_policy=SOURCE_QA_ACQUISITION_POLICY,acquisition_qa_pool=acquisition,
        acquisition_qa_audit={'source_input_sha256':source_input['sha256'],'bundle_sha256':acquisition['sha256'],
            'answer_targets':len(records),'stage2_acquisition_qa_dose':0},
        acquisition_qa_similarity=acquisition_similarity(facts,records),confirmation_protocol_schema=PROTOCOL,
        confirmation_audit=audit,audit_sha256=audit['sha256'],audit_complete=True,
        confirmation_audit_summary={key:value for key,value in context.items() if key not in ('sources','groups','acquisition_records','evaluation_records')},
        model='explicit-synthetic-test-model',model_revision=config['model_revision'],tokenizer_sha256='test-tokenizer',test_fixture=True))
    prepared = Path(directory)/'prepared'
    for name,payload in [('manifest.json',manifest),('schedule.json',schedule),('examples.json',examples),
                         ('grounding-declarations.json',grounding),('acquisition-qa.json',acquisition),
                         ('acquisition-source-input.json',source_input),('acquisition-similarity.json',manifest['acquisition_qa_similarity']),
                         ('confirmation-audit.json',audit)]:
        write_json(prepared/name,payload)
    return prepared,manifest,schedule,RuntimeTokenizer


class ConfirmationRuntimeTests(unittest.TestCase):
    def test_guarded_analysis_shape_keeps_estimation_fallback_and_complete_design(self):
        """Analysis shape only. Dedicated completion tests validate actual terminal files."""
        from spacing_rerun.analysis import summarize_bundles
        from spacing_rerun.evaluation import aggregate
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/'rep01',Path(directory)/'rep02']
            manifests=[]
            for index,path in enumerate(paths):
                _,manifest,schedule,_=prepared_confirmation_fixture(path,2026100801+index)
                manifests.append(manifest)
                for offset,arm in enumerate(('NONE','UNI','EXP','MASS','GEN')):
                    rows=[{'id':f['id'],'unit_id':f['unit_id'],'event':f['event'],'role':f['role'],
                           'loss':1+offset*.01+index*.005} for f in manifest['split']['facts'] if f['role'] in ('old','new','control')]
                    for step in (0,schedule['stage2_steps'],schedule['stage2_steps']+84):
                        write_json(path/arm/'evaluations'/f'stage2-{step:06}.json',{
                            'manifest_sha256':manifest['sha256'],'metric_schema':GROUNDED_METRICS,'stage2_step':step,
                            'variants':{'canonical':{'aggregate':aggregate(rows),'facts':rows}},'test_fixture':True})
            _,prereg,final_span=prereg_fixture(directory,manifests[0])
            prereg['replicate_designs']=[{'manifest_sha256':manifest['sha256'],**{key:manifest['config'][key] for key in
                ('split_seed','order_seed','train_seed','eval_seed')},'roles_sha256':digest(manifest['split']['roles'])} for manifest in manifests]
            prereg['replicate_manifest_sha256']=[manifest['sha256'] for manifest in manifests];seal(prereg)
            prereg_path=Path(directory)/'synthetic-prereg.json';write_json(prereg_path,prereg)
            for path,manifest in zip(paths,manifests):
                write_json(path/'stage1'/'acquisition_decision.json',{'manifest_sha256':manifest['sha256'],'selected_E':4,
                    'fixed_dose_continuation_authorized':True,'outcome_filtering_permitted':False,
                    'confirmation_preregistration_sha256':prereg['sha256'],'test_fixture':True})
            output=Path(directory)/'analysis.json'
            with patch('spacing_rerun.final_span.verify_final_span_report',return_value=final_span), \
                 patch('spacing_rerun.analysis.validate_confirmation_completion',return_value={
                     'test_fixture':True,'scope':'analysis_shape_only_no_production_completion_claim'}), \
                 patch.dict('os.environ',{'SPACING_CODE_COMMIT':prereg['code_commit']}):
                summarize_bundles(paths,output,84,preregistration=prereg_path)
                result=json.loads(output.read_text())
                self.assertEqual(result['inference_claim'],'estimation')
                self.assertIsNone(result['contrasts']['H2']['loss_equivalent'])
                self.assertIsNone(result['contrasts']['H2']['protected_event_cluster_sensitivity']['useful_direction'])
                self.assertEqual(result['contrasts']['H2']['n'],2)
                with self.assertRaisesRegex(ValueError,'Incomplete preregistered replicate set'):
                    summarize_bundles(paths[:1],output,84,preregistration=prereg_path)

    def test_actual_tiny_hf_olmo_guarded_confirmation_retains_failed_gate_and_continues(self):
        import torch
        from olmo.config import ModelConfig
        from olmo.model import OLMo
        from hf_olmo.configuration_olmo import OLMoConfig
        from hf_olmo.modeling_olmo import OLMoForCausalLM
        from spacing_rerun.prepare import load_prepared
        from spacing_rerun.training import run_acquisition,run_arm
        old_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                prepared,manifest,schedule,tokenizer_class=prepared_confirmation_fixture(directory)
                load_prepared(prepared)
                _,prereg,final_span=prereg_fixture(directory,manifest)
                prereg_path=Path(directory)/'synthetic-prereg.json';write_json(prereg_path,prereg)
                def factory(*args):
                    cfg=ModelConfig(d_model=16,n_heads=2,n_layers=1,mlp_ratio=2,max_sequence_length=256,
                                    vocab_size=128,embedding_size=128,rope=True,alibi=False,attention_dropout=0.,
                                    residual_dropout=0.,embedding_dropout=0.,init_device='cpu')
                    return OLMoForCausalLM(OLMoConfig(**cfg.asdict()),OLMo(cfg)),tokenizer_class()
                # The final-span/planning artifact is explicitly synthetic.
                # Real hf_olmo training, evaluations, prepared-data guards, dose,
                # checkpoint and continuation validation all execute.
                with patch('spacing_rerun.final_span.verify_final_span_report',return_value=final_span),patch.dict('os.environ',{'SPACING_CODE_COMMIT':prereg['code_commit']}):
                    stage1=Path(directory)/'stage1'
                    run_acquisition(prepared,stage1,device='cpu',fixed_exposures=4,preregistration=prereg_path,model_factory=factory)
                    decision=json.loads((stage1/'acquisition_decision.json').read_text())
                    self.assertFalse(decision['acquisition_usable'])
                    self.assertEqual(decision['observed_acquisition_old_exact_match_event_macro'],0)
                    self.assertTrue(decision['fixed_dose_continuation_authorized'])
                    self.assertFalse(decision['outcome_filtering_permitted'])
                    self.assertEqual(decision['selected_E'],4)
                    output=Path(directory)/'UNI'
                    run_arm(prepared,output,stage1/'stage1.pt','UNI',device='cpu',preregistration=prereg_path,model_factory=factory)
                    state=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
                    self.assertEqual(state['progress']['status'],'complete')
                    self.assertEqual(state['progress']['continuation_cursor'],len(schedule['arms']['UNI']))
                    self.assertFalse(state['progress']['acquisition_usable'])
                    self.assertEqual(state['progress']['confirmation_preregistration_sha256'],prereg['sha256'])
                    self.assertTrue(all('acq/' not in line for line in (output/'exposures.jsonl').read_text().splitlines()))
        finally:
            torch.set_num_threads(old_threads)
