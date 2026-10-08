"""Synthetic protocol coverage evidence only, never production approval claims."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from test_source_patch import native_trace_fixture
from spacing_rerun.common import digest
from spacing_rerun.sourceqa_protocol import (build_protocol_coverage, completed_artifact,
    development_census_snapshot, development_source_runtime_snapshot, protocol_source_snapshot,
    seeded_audit_plan, verify_protocol_coverage as _verify_protocol_coverage, _census_prior_approvals, legacy_claim_unverified_envelope)
from experiments.spacing_rerun.tests.test_confirmation_dependency import final_fixture, actual_evidence_fixture, seal


def raw(value, path):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return {'path': path, 'utf8': text, 'file_sha256': hashlib.sha256(text.encode()).hexdigest(), 'canonical_sha256': value['sha256']}


def package(artifact, name, **extra):
    result={'artifact': artifact, 'evidence': actual_evidence_fixture(artifact, name), **extra}
    refresh_package(result,name)
    return result


def refresh_package(value, name=None):
    seal(value['artifact'])
    value['evidence'] = actual_evidence_fixture(value['artifact'], name or value['evidence']['client_request_id'])
    value['evidence']['task_status']['latestTerminalRunId']=value['evidence']['task_status']['childRunId']
    reviewer=value['artifact'].get('reviewer',value['artifact'].get('reviewer_identity',{}))
    if reviewer.get('model'):value['evidence']['task_status']['model']=reviewer['model']
    snapshot={'task_statuses':[{'clientRequestId':value['evidence']['client_request_id'],
                               'result':value['evidence']['task_status']}]}
    value['evidence']['task_status_file']=raw(seal(snapshot),'/tmp/synthetic-original-status.json')
    if 'first_pass_receipt' in value:
        first=package(value['first_pass_receipt'],value['evidence']['client_request_id'])
        first['evidence']['receipt_file']=copy.deepcopy(value['first_pass_file'])
        value['first_pass_evidence']=first['evidence']


def trusted_evaluator_fixture(chunks,catalog):
    snapshot=protocol_source_snapshot(chunks,catalog)
    def binding(name):return {'path':'/tmp/synthetic-trusted-'+name+'.json',
                              'file_sha256':digest(['synthetic-external-bytes',name]),
                              'content_sha256':digest(['synthetic-external-content',name])}
    return seal({'schema':'spacing-sourceqa-current-evaluator-input-metadata-v1',
        'scope':{'mode':'confirmation','source_catalog_sha256':catalog['sha256'],'source_snapshot_sha256':digest(snapshot)},
        'configuration_sha256':digest('synthetic externally pinned evaluator configuration'),
        'catalog_binding':binding('catalog'),'snapshot_binding':binding('snapshot'),'provenance_binding':binding('provenance'),
        'required_surface_correlate_ids_by_record':{r['id']:[] for r in snapshot if r['field']=='acquisition_records'}})


def verify_protocol_coverage(proof,protocol,chunks,catalog,**kwargs):
    # This deterministic factory is external to each candidate proof. Production has no default pins.
    kwargs.setdefault('expected_evaluator_metadata',trusted_evaluator_fixture(chunks,catalog))
    return _verify_protocol_coverage(proof,protocol,chunks,catalog,**kwargs)


def census_sources_fixture():
    """Synthetic full populations, with a three-version normal QA overlay."""
    def original(value,name):
        seal(value);return {'artifact':value,'artifact_file':raw(value,'/tmp/synthetic-'+name+'.json')}
    def rows(prefix,group_count,qa_count,unit_count):
        units=[{'unit_id':f'{prefix}-unit-{i}','event_id':'event-0','source_statement':f'Synthetic assertion {i}',
                'canonical_answers':[f'answer-{i}'],'evidence':[], 'rationale':'Forbidden author content',
                'probe_ids':['Forbidden evaluator identity']} for i in range(unit_count)]
        groups=[{'id':f'{prefix}-group-{i}','event':'event-0','source_unit_ids':[units[i]['unit_id']],
                 'statement':f'Synthetic declaration {i}','evidence':[]} for i in range(group_count)]
        qa=[{'id':f'{prefix}-qa-{i}','event':'event-0','unit_id':groups[i%group_count]['id'],
             'answer':f'answer-{i%group_count}','questions':['Which synthetic answer?','What synthetic answer?'],
             'evidence':[]} for i in range(qa_count)]
        return units,groups,qa
    units,groups,qa=rows('normal',157,78,188)
    grounded=original({'schema':'spacing-grounded-declarations-v1','mode':'development',
        'groups':groups,'source_units':units,'factsheets':{'event-0':'Synthetic factsheet'}},'normal-grounded')
    versions=[original({'records':qa},'normal-qa')]
    for i in range(1,3):
        row=copy.deepcopy(qa[i]);row['questions'][0]+=str(i)
        versions.append(original({'records':[row],'parent_review_packet_sha256':versions[-1]['artifact']['sha256']},f'normal-qa-{i}'))
    scale=[]
    for i,count in ((1,128),(2,129)):
        units,groups,qa=rows(f'scale-{i}',count,143,count)
        units=[{'unit_id':u['unit_id'],'event':u['event_id'],'source_assertion':u['source_statement'],
                'canonical_answer_labels':u['canonical_answers'],'canonical_probes':[{'forbidden':'probe counts'}]}
               for u in units]
        scale.append(original({'chunk_index':i,'author_rationales_included':False,'groups':groups,
            'acquisition_records':qa,'source_units':units,'factsheets':{'event-0':'Synthetic factsheet'}},f'scale-{i}'))
    sources={'normal_grounded':grounded,'normal_qa_versions':versions,'scale_packets':scale}
    effective={q['id']:q for v in versions for q in v['artifact']['records']}
    packets=[('normal',seal({'records':list(effective.values())}))]+[('scale',p['artifact']) for p in scale]
    prior=[]
    for i,(scope,packet) in enumerate(packets):
        field='records' if 'records' in packet else 'acquisition_records'
        review=seal({'review_packet_sha256':packet['sha256'],field:[{'id':q['id'],'status':'approved',
            'reviewed_content_sha256':digest(q)} for q in packet[field]]})
        prior.append({'scope':scope,'review_packet':packet,'review_packet_file':raw(packet,f'/tmp/synthetic-development-prior-packet-{i}.json'),
                      'reviews':[package(review,f'synthetic-development-prior-{i}')]})
    return sources,prior


def attach_census_contract(census_package,protocol,snapshot,chunks):
    sources,prior=census_sources_fixture();development=development_census_snapshot(sources)
    for c in chunks:
        c['artifact_files']={'reviews':[raw(r,f'/tmp/synthetic-original-prior-{c["review_packet"]["chunk_index"]}-{i}.json')
                                        for i,r in enumerate(c['reviews'])]}
        c['original_review_packages']=[package(r,r['reviewer']['agent_id']) for r in c['reviews']]
        for p,info in zip(c['original_review_packages'],c['artifact_files']['reviews']):p['evidence']['receipt_file']=info
    properties={k:{'operational_definition':'Synthetic source-only definition '+k} for k in
                ('LE1a','LE1b','LE1c','C6_masked','ordinary_retained_defect')}
    freeze=seal({'schema':'spacing-sourceqa-census-rule-freeze-v1','protocol_sha256':protocol['sha256'],
        'rule_text':protocol['exception_rules'],'census_constants':protocol['census_constants'],
        'prior_count_view_disclosure':{'prior_counts_seen':False,'report_sha256s':[],
                                     'known_row_motivation':'Synthetic test-only rule, no production motivation claim'},
        'normalization_reference':{'path':'experiments/spacing_rerun/spacing_rerun/data.py',
            'file_sha256':protocol['frozen_mechanical_files']['experiments/spacing_rerun/spacing_rerun/data.py'],
            'function_text':'return " ".join(str(text).strip().casefold().split()).rstrip(".!?,;:")'},
        'property_definitions':properties,'eligibility_sub_kinds':{'LE1a':'conflict','LE1b':'alias','LE1c':'original_label_nonuniqueness'}})
    census_package.update(development_population_sources=sources,development_prior_approval_sources=prior,
                          rule_freeze=package(freeze,'synthetic-rule-freeze'),prior_census_versions=[],prior_rule_versions=[])
    confirmation=[{'domain':'confirmation','scope':str(r['chunk_index']),**{k:v for k,v in r.items() if k!='chunk_index'}} for r in snapshot]
    population=sorted(confirmation+development,key=lambda r:tuple(r[k] for k in ('domain','scope','field','id')))
    refs=_census_prior_approvals(chunks,prior)
    trace=json.dumps({'test_fixture':True,'events':['Actual synthetic freeze','Actual synthetic counts view']})
    trace_info={'path':'/tmp/synthetic-census-original-timeline.json','utf8':trace,'file_sha256':hashlib.sha256(trace.encode()).hexdigest()}
    timeline={'rule_freeze_sha256':freeze['sha256'],'frozen_utc':'2026-10-08T00:00:00Z','census_started_utc':'2026-10-08T00:01:00Z',
        'first_count_view_utc':'2026-10-08T00:02:00Z','original_trace_file_sha256s':[trace_info['file_sha256']],
        'retained_prior_census_versions':[],'retained_prior_rule_versions':[],'prior_counts_viewed_before_current_freeze':False}
    contract={'population_snapshot':population,'property_rows':[{**{k:r[k] for k in ('domain','scope','field','id','content_sha256','context_sha256')},
        'properties':{k:False for k in properties},'source_only_evidence':'Synthetic test of every property'} for r in population],
        'lane_sets':[],'newly_lane_eligible_sets':[],'prior_approval_results':[{'original_approval_reference':r,'masked_C6_passed':True,
             'source_only_evidence':'Synthetic original masked test'} for r in refs],
        'previously_approved_failing_C6_rows':[],'per_sub_kind_counts':{kind:0 for kind in freeze['eligibility_sub_kinds'].values()},
        'rule_freeze_sha256':freeze['sha256'],'prior_approval_inventory_sha256':digest(refs),'retained_prior_census_versions':[],
        'retained_prior_rule_versions':[],
        'chronology':timeline}
    census_package['artifact']['contract']=contract;refresh_package(census_package)
    attestation=seal({'rule_freeze_sha256':freeze['sha256'],'census_artifact_sha256':census_package['artifact']['sha256'],
        'all_versions_retained':True,'chronology_verified_from_original_traces':True,'chronology':timeline})
    census_package['contract_pointers']={k:'/contract/'+k for k in contract}
    census_package['freeze_order_attestation']=package(attestation,'synthetic-independent-chronology-custodian',
        attestation_pointers={k:'/'+k for k in ('rule_freeze_sha256','census_artifact_sha256','all_versions_retained','chronology_verified_from_original_traces')},
        chronology_pointer='/chronology',chronology_trace_files=[trace_info])


def separate_development_fixture(protocol):
    """Synthetic independent negative source report and successful hashes-only runtime tool output."""
    rows = []
    for scope, count, events in (('normal78', 78, 5), ('scale286', 286, 15)):
        rows.extend({'set': scope, 'id': f'{scope}-{i:03}', 'event': f'event-{i % events:03}',
                     'content_sha256': digest([scope, i]), 'reclassification': 'unresolved_negative_retained'}
                    for i in range(count))
    artifact = seal({'test_fixture': True, 'schema': 'synthetic-independent-negative-development-review',
        'criteria_binding': {'exact_adopted_protocol_sha256_declared': protocol['sha256'], 'protocol_verified': False},
        'truthfulness': {'source_content_edited': False}, 'rows': rows, 'blockers': ['Runtime evidence not available to source reviewer']})
    value = package(artifact, 'synthetic-independent-negative-development-review',
        attestation_pointers={'protocol_sha256': '/criteria_binding/exact_adopted_protocol_sha256_declared',
                             'development_content_changed': '/truthfulness/source_content_edited'},
        development_source_record_rows_pointer='/rows')
    info = value['evidence']['receipt_file'];scopes = {};trials = []
    for scope, count, runtime_count, prefix, trial_count in (
            ('normal78', 78, 78, 'normal', 6), ('scale286', 286, 218, 'scaled', 3)):
        source = sorted(({k:r[k] for k in ('id','event','content_sha256')} for r in rows if r['set']==scope), key=lambda r:r['id'])
        deployed = source[:runtime_count]
        scopes[scope] = {'reviewed_record_count':count, 'runtime_record_count':runtime_count,
            'excluded_reviewed_record_ids':[r['id'] for r in source[runtime_count:]],
            'reviewed_source_projection_sha256':digest(source), 'runtime_source_projection_sha256':digest(deployed),
            'all_runtime_records_exact_reviewed_source_content':True,
            'membership_kind':'exact_equality' if prefix=='normal' else 'strict_subset'}
        trials.extend({'trial_id':f'{prefix}-{i:02}', 'source_qa_record_count':runtime_count,
                       'source_event_count':len({r['event'] for r in deployed}),
                       'source_review_content_projection_sha256':digest(deployed),
                       'all_runtime_records_exact_reviewed_source_content':True} for i in range(1,trial_count+1))
    metadata = {'domain':'development_source_runtime_v1', 'actual_source_review_binding':{
        'sha256':artifact['sha256'],'file_sha256':info['file_sha256'],'bytes':len(info['utf8'].encode())},
        'actual_runtime_custody_binding':{'sha256':digest('synthetic actual runtime custody')},
        'actual_observation_binding':{'sha256':digest('synthetic actual remote observation')},
        'scopes':scopes,'trials':trials,'semantic_decisions_created':False,'runtime_or_source_edited':False,
        'models_run':False,'evaluation_wording_read':False,'attestations_not_established':[
            'development_retrained','E_reselected','gate_changed','prospective_snapshot_unchanged']}
    text = json.dumps(metadata)
    def original(value, path):
        text = json.dumps(value)
        return {'path':path,'utf8':text,'file_sha256':hashlib.sha256(text.encode()).hexdigest()}
    value['runtime_observation_interface'] = {
        'metadata_stdout_file':original(metadata,'/tmp/synthetic-original-runtime-metadata.json'),
        'actual_execution_file':original({'chunk_id':'synthetic-tool-call','exit_code':0,'output':text},
                                         '/tmp/synthetic-original-runtime-execution.json'),
        'validator_script_file_sha256':'cf81ef91d8599e19e5f66f916cbff5523f0bfea433f85b0ef68eb12a8d610374'}
    preservation = seal({'test_fixture':True,'schema':'synthetic-separate-runtime-preservation',
        'source_review_sha256':artifact['sha256'],
        'runtime_custody_sha256':metadata['actual_runtime_custody_binding']['sha256'],
        'runtime_source_observation_sha256':metadata['actual_observation_binding']['sha256'],
        'development_retrained':False,'E_reselected':False,'gate_changed':False})
    value['runtime_preservation_attestation'] = package(preservation, 'synthetic-separate-runtime-preservation',
        attestation_pointers={k:'/'+k for k in preservation if k not in ('sha256','schema','test_fixture')})
    return value


def coverage_fixture():
    _, chunks, catalog, _ = final_fixture()
    repo = Path(__file__).resolve().parents[3]
    protocol = json.loads((repo / 'plans/spacing_sourceqa_review_protocol_amendment_20261008.json').read_text())
    snapshot = protocol_source_snapshot(chunks, catalog);snapshot_sha = digest(snapshot)
    fresh, groups, conditional = [], [], []
    for c in chunks:
        packet = c['review_packet'];original = copy.deepcopy(c['reviews'][0]);index = packet['chunk_index']
        groups.append(package(original, original['reviewer']['agent_id'], review_packet=packet,
                              review_packet_file=raw(packet, f'/tmp/synthetic-protocol-packet-{index}.json')))
        first = copy.deepcopy(original)
        first.update(stage='first_pass_blind', author_rationales_seen=False, model_outcomes_seen=False,
                     evaluator_wording_seen=False, other_reviewer_judgments_seen=False, prior_review_history_seen=False)
        first['groups'] = []
        for d in first['acquisition_records']:
            d['retained_conditions'] = {f'R{i}': {'status': 'pass', 'notes': 'Explicit synthetic retained rule test'} for i in range(1, 10)}
            if d.get('conditions'):conditional.append(d['id'])
        seal(first);first_info = raw(first, f'/tmp/synthetic-B1-{index}.json')
        history = seal({'schema': 'synthetic-original-review-history', 'current_review_packet_sha256': packet['sha256'],
            'review_decisions_created': False, 'source_only': True, 'author_rationales_included': False,
            'model_outcomes_included': False, 'actual_prior_review_materials': []})
        history_info = raw(history, f'/tmp/synthetic-history-{index}.json')
        final = copy.deepcopy(first);final.update(stage='stage2_final', prior_review_history_seen=True,
            protocol_sha256=protocol['sha256'], first_pass_binding={'path': first_info['path'],
                'sha256': first['sha256'], 'file_sha256': first_info['file_sha256']},
            prior_review_supplement_binding={'path': history_info['path'], 'sha256': history['sha256'],
                'file_sha256': history_info['file_sha256']})
        seal(final)
        fresh.append(package(final, final['reviewer']['agent_id'], chunk_index=index,
                             first_pass_receipt=first, first_pass_file=first_info,
                             prior_review_supplement=history, prior_review_supplement_file=history_info))
    context = {key: digest(key) for key in ('membership_sha256', 'exception_lanes_sha256',
        'required_source_conditions_sha256', 'lane_reporting_policy_sha256', 'algorithm_sha256')}
    base = {'test_fixture': True, 'protocol_sha256': protocol['sha256'], 'source_snapshot_sha256': snapshot_sha}
    census = seal({**base, 'schema': 'synthetic-protocol-census', 'model_outcomes_seen': False,
        'evaluation_wording_seen': False, 'author_rationales_seen': False, 'no_semantic_decisions_created': True,
        'rows': snapshot})
    census_package = package(census, 'synthetic-independent-census',
        attestation_pointers={k: '/' + k for k in ('protocol_sha256', 'source_snapshot_sha256', 'model_outcomes_seen',
            'evaluation_wording_seen', 'author_rationales_seen', 'no_semantic_decisions_created')},
        record_binding_pointers=[f'/rows/{i}' for i in range(len(snapshot))])
    attach_census_contract(census_package,protocol,snapshot,chunks)
    development = seal({'test_fixture': True, 'protocol_sha256': protocol['sha256'],
        'schema': 'synthetic-development-reclassification', 'development_content_changed': False,
        'development_retrained': False, 'E_reselected': False, 'gate_changed': False, 'symmetry_complete': True,
        'normal': {'source': {'records': 78, 'source_sha256': digest('normal synthetic source')},
                   'runtime': {'E': 6, 'manifest_sha256': digest('normal synthetic runtime')}},
        'scale': {'source': {'records': 286, 'source_sha256': digest('scale synthetic source')},
                  'runtime': {'E': 6, 'manifest_sha256': digest('scale synthetic runtime')}}})
    development_package = package(development, 'synthetic-independent-development-provenance',
        attestation_pointers={k: '/' + k for k in ('protocol_sha256', 'development_content_changed',
            'development_retrained', 'E_reselected', 'gate_changed', 'symmetry_complete')},
        development_source_runtime_snapshot_pointers={scope:{kind:f'/{scope}/{kind}' for kind in ('source','runtime')}
                                                      for scope in ('normal','scale')})
    flags = [{'id': r['id'], 'content_sha256': r['content_sha256'], 'canonical_or_paraphrase_overlap_flag': False,
              'surface_correlate_flags': {}} for r in snapshot if r['field'] == 'acquisition_records']
    expected_evaluator_metadata=trusted_evaluator_fixture(chunks,catalog)
    custody = seal({**base, 'schema': 'synthetic-evaluation-custody', 'boolean_only_output': True,
        'evaluation_text_exposed': False, 'filtering_permitted': False, 'missing_flags_imputed_false': False,
        'evaluator_input_metadata':expected_evaluator_metadata,
        'required_surface_correlate_ids_by_record':copy.deepcopy(expected_evaluator_metadata['required_surface_correlate_ids_by_record']),
        'flags': flags, 'canonical_root_counts': {u['unit_id']: len(u['canonical_probes']) for u in catalog['units']}})
    custody_package = package(custody, 'synthetic-separate-evaluation-custodian',
        attestation_pointers={k: '/' + k for k in ('protocol_sha256', 'source_snapshot_sha256', 'boolean_only_output',
            'evaluation_text_exposed', 'filtering_permitted', 'missing_flags_imputed_false')},
        flag_rows_pointer='/flags', canonical_root_counts_pointer='/canonical_root_counts',
        evaluator_input_metadata_pointer='/evaluator_input_metadata',
        required_surface_correlate_ids_pointer='/required_surface_correlate_ids_by_record',
        required_surface_correlate_ids_by_record={r['id']: [] for r in flags}, dependency_context=context)
    proof = build_protocol_coverage(protocol, chunks, catalog, fresh_qa_reviews=fresh, group_reviews=groups,
        census=census_package, development_reclassification=development_package, evaluation_custody=custody_package,
        dependency_context=context, conditional_acquisition_record_ids=conditional,
        expected_evaluator_metadata=expected_evaluator_metadata)
    return protocol, chunks, catalog, proof


class ProtocolCoverageTests(unittest.TestCase):
    @staticmethod
    def sync_first(value):
        seal(value['first_pass_receipt'])
        value['first_pass_file']=raw(value['first_pass_receipt'],value['first_pass_file']['path'])
        value['artifact']['first_pass_binding'].update(sha256=value['first_pass_receipt']['sha256'],
                                                       file_sha256=value['first_pass_file']['file_sha256'])

    def test_every_supplied_B1_protocol_and_finality_declaration_is_checked(self):
        protocol,chunks,catalog,base=coverage_fixture()
        for kind in ('wrong-protocol','contradictory-protocol','pointer-contradiction','true-finality','integer-finality',
                     'true-first-pass-finality','integer-first-pass-finality','absent-protocol','correct-protocol'):
            with self.subTest(kind=kind):
                proof=copy.deepcopy(base);value=proof['fresh_qa_reviews'][0];first=value['first_pass_receipt']
                if kind in ('wrong-protocol','contradictory-protocol','pointer-contradiction'):
                    first['protocol_sha256']='0'*64
                    if kind=='contradictory-protocol':first['adopted_protocol_sha256']=protocol['sha256']
                    if kind=='pointer-contradiction':
                        first['actual_protocol']=protocol['sha256'];value['original_review_value_pointers']={'first_pass':{'protocol_sha256':'/actual_protocol'}}
                elif 'finality' in kind:
                    field='first_pass_counts_as_final_approval' if 'first-pass' in kind else 'counts_as_final_approval'
                    first[field]=0 if 'integer' in kind else True
                elif kind=='absent-protocol':
                    for field in ('protocol_sha256','adopted_protocol_sha256','review_protocol_sha256'):first.pop(field,None)
                else:first['protocol_sha256']=protocol['sha256']
                self.sync_first(value);refresh_package(value);seal(proof)
                if kind in ('absent-protocol','correct-protocol'):verify_protocol_coverage(proof,protocol,chunks,catalog)
                else:
                    with self.assertRaises(ValueError):verify_protocol_coverage(proof,protocol,chunks,catalog)

    def test_completed_package_aliases_and_typed_execution_channels_are_exact(self):
        _,_,_,proof=coverage_fixture();base=proof['fresh_qa_reviews'][0]
        for kind in ('artifact-receipt','task-id','client-id','child-run','run-id','provider-instance',
                     'unnamed-author-role','unnamed-runtime','receipt-bool-int','status-bool-int'):
            with self.subTest(kind=kind):
                value=copy.deepcopy(base);person=value['artifact']['reviewer'];status=value['evidence']['task_status']
                if kind=='artifact-receipt':
                    value['receipt']=copy.deepcopy(value['artifact']);value['receipt']['model_outcomes_seen']=True;seal(value['receipt'])
                elif kind in ('task-id','client-id','child-run','run-id'):
                    field={'task-id':'task_id','client-id':'client_request_id','child-run':'child_run_id','run-id':'run_id'}[kind]
                    person[field]=status['childThreadId']
                elif kind=='provider-instance':person['provider_instance_id']='different-provider-instance'
                elif kind in ('unnamed-author-role','unnamed-runtime'):
                    value['artifact'].pop('reviewer_actor_identity',None)
                    value['artifact']['reviewer']={'model':person['model'],
                        'role':'source_qa_author' if kind=='unnamed-author-role' else 'independent reviewer'}
                    if kind=='unnamed-runtime':value['artifact']['reviewer']['runtime_thread_id']='another-thread'
                refresh_package(value)
                if kind=='provider-instance':
                    value['evidence']['task_status']['providerInstanceId']='actual-original-provider-instance'
                    snapshot={'task_statuses':[{'clientRequestId':value['evidence']['client_request_id'],
                        'result':value['evidence']['task_status']}]}
                    value['evidence']['task_status_file']=raw(seal(snapshot),'/tmp/synthetic-provider-status.json')
                elif kind in ('receipt-bool-int','status-bool-int'):
                    info=value['evidence']['receipt_file' if kind=='receipt-bool-int' else 'task_status_file']
                    original=json.loads(info['utf8'])
                    if kind=='receipt-bool-int':original['model_outcomes_seen']=0
                    else:original['task_statuses'][0]['result']['hasPendingChildRuns']=0
                    text=json.dumps(original);info.update(utf8=text,file_sha256=hashlib.sha256(text.encode()).hexdigest())
                with self.assertRaises(ValueError):completed_artifact(value)

    def test_group_packet_aliases_and_explicit_provisional_stages_cannot_qualify(self):
        protocol,chunks,catalog,base=coverage_fixture()
        for kind in ('nested-group-packet','group-packet-content','provisional-group','provisional-fresh'):
            with self.subTest(kind=kind):
                proof=copy.deepcopy(base);value=proof['fresh_qa_reviews' if kind=='provisional-fresh' else 'group_reviews'][0]
                if kind=='nested-group-packet':value['artifact']['bindings']={'primary_packet':{'sha256':'0'*64}}
                elif kind=='group-packet-content':value['artifact']['packet_content_sha256']='0'*64
                else:value['artifact']['stage']='provisional'
                refresh_package(value);seal(proof)
                with self.assertRaises(ValueError):verify_protocol_coverage(proof,protocol,chunks,catalog)

    def test_historical_completed_nonfinal_reviews_are_retained_without_final_qualification(self):
        _,chunks,_,proof=coverage_fixture();sources=proof['census']['development_prior_approval_sources']
        baseline=_census_prior_approvals(chunks,sources)
        historical=copy.deepcopy(chunks[0]);historical.pop('history',None)
        for package_value,info in zip(historical['original_review_packages'],historical['artifact_files']['reviews']):
            package_value['artifact']['stage']='provisional';package_value['artifact']['counts_as_final_approval']=False
            refresh_package(package_value);info.clear();info.update(copy.deepcopy(package_value['evidence']['receipt_file']))
        historical['reviews']=[p['artifact'] for p in historical['original_review_packages']]
        retained=copy.deepcopy(chunks);retained[0]['history']=[historical]
        self.assertEqual(_census_prior_approvals(retained,sources),baseline)
        for kind in ('missing-raw','missing-completion','final-promotion','current-nonfinal','development-nonfinal',
                     'extra-raw','source-author-original','mixed-incomplete-final'):
            with self.subTest(kind=kind):
                own=copy.deepcopy(retained);development=copy.deepcopy(sources);old=own[0]['history'][0]
                if kind=='missing-raw':old['artifact_files']['reviews'].pop()
                elif kind=='missing-completion':old['original_review_packages'].pop()
                elif kind=='final-promotion':
                    p=old['original_review_packages'][0];p['artifact']['counts_as_final_approval']=True;refresh_package(p)
                    old['artifact_files']['reviews'][0]=copy.deepcopy(p['evidence']['receipt_file']);old['reviews'][0]=p['artifact']
                elif kind=='current-nonfinal':own[0]=old
                elif kind=='development-nonfinal':
                    p=development[0]['reviews'][0];p['artifact']['stage']='provisional';refresh_package(p)
                elif kind=='extra-raw':old['artifact_files']['reviews'].append(copy.deepcopy(old['artifact_files']['reviews'][0]))
                elif kind=='source-author-original':
                    p=own[0]['original_review_packages'][0];p['artifact']['reviewer']['agent_id']=own[0]['authored']['author']['agent_id']
                    refresh_package(p,p['artifact']['reviewer']['agent_id']);own[0]['reviews'][0]=p['artifact']
                    own[0]['artifact_files']['reviews'][0]=copy.deepcopy(p['evidence']['receipt_file'])
                else:
                    p=copy.deepcopy(old['original_review_packages'][0]);p['artifact']['stage']='final'
                    p['artifact']['counts_as_final_approval']=True;p['artifact']['acquisition_records']=[];refresh_package(p)
                    old['original_review_packages'].append(p);old['reviews'].append(p['artifact'])
                    old['artifact_files']['reviews'].append(copy.deepcopy(p['evidence']['receipt_file']))
                with self.assertRaises(ValueError):_census_prior_approvals(own,development)

    def test_nested_frozen_json_boolean_integer_values_are_distinct(self):
        protocol,chunks,catalog,base=coverage_fixture()
        def coerce(value):
            if isinstance(value,dict):
                for k,v in value.items():
                    if type(v) is int and v in (0,1):value[k]=bool(v);return True
                    if type(v) is bool:value[k]=int(v);return True
                    if coerce(v):return True
            elif isinstance(value,list):
                for i,v in enumerate(value):
                    if type(v) is int and v in (0,1):value[i]=bool(v);return True
                    if type(v) is bool:value[i]=int(v);return True
                    if coerce(v):return True
            return False
        proof=copy.deepcopy(base);value=proof['census']['rule_freeze'];freeze=value['artifact']
        self.assertTrue(coerce(freeze['census_constants']) or coerce(freeze['rule_text']))
        refresh_package(value)
        # Preserve the candidate's internal seal/binding while the external
        # adopted nested rule values remain exact and unchanged.
        census=proof['census'];census['artifact']['contract']['rule_freeze_sha256']=freeze['sha256']
        refresh_package(census);seal(proof)
        with self.assertRaisesRegex(ValueError,'exact adopted rules'):verify_protocol_coverage(proof,protocol,chunks,catalog)

    def test_all_original_aliases_and_pointers_reject_contradictory_declarations(self):
        protocol,chunks,catalog,base=coverage_fixture()
        kinds=('nested-packet','packet-content','native-negative','stage-phase','visibility-alias','pathless-b1',
               'partial-flat-b1','malformed-binding-container','receipt-evidence-seal')
        for kind in kinds:
            with self.subTest(kind=kind):
                proof=copy.deepcopy(base);value=proof['fresh_qa_reviews'][0];receipt=value['artifact']
                if kind in ('nested-packet','packet-content'):
                    receipt['pointer_packet']=receipt['review_packet_sha256']
                    value['original_review_value_pointers']={'final':{'packet_sha256':'/pointer_packet'}}
                    if kind=='nested-packet':receipt['bindings']={'primary_packet':{'sha256':'0'*64}}
                    else:receipt['packet_content_sha256']='0'*64
                elif kind=='native-negative':
                    receipt['pointer_rows']=copy.deepcopy(receipt['acquisition_records'])
                    value['original_review_value_pointers']={'final':{'acquisition_records':'/pointer_rows'}}
                    receipt['acquisition_decisions']=copy.deepcopy(receipt['acquisition_records'])
                    receipt['acquisition_decisions'][0]['status']='needs_revision'
                elif kind=='stage-phase':receipt['phase']='B1_first_pass'
                elif kind=='visibility-alias':receipt['evaluation_wording_seen']=True
                elif kind=='pathless-b1':receipt['bindings']={'b1_receipt':{'sha256':'0'*64}}
                elif kind=='partial-flat-b1':receipt['b1_first_pass_path']='/another/original.json'
                elif kind=='malformed-binding-container':receipt['bindings']=[]
                refresh_package(value)
                if kind=='receipt-evidence-seal':value['evidence']['receipt_sha256']='0'*64
                seal(proof)
                with self.assertRaises(ValueError):verify_protocol_coverage(proof,protocol,chunks,catalog)

    def test_fresh_and_group_receipts_cannot_promote_explicit_nonfinal_decisions(self):
        protocol,chunks,catalog,base=coverage_fixture()
        for kind in ('fresh-false','fresh-integer','group-first-pass','group-false'):
            with self.subTest(kind=kind):
                proof=copy.deepcopy(base);value=proof['fresh_qa_reviews' if kind.startswith('fresh') else 'group_reviews'][0]
                if kind=='group-first-pass':value['artifact']['stage']='B1_first_pass'
                else:value['artifact']['counts_as_final_approval']=0 if kind=='fresh-integer' else False
                refresh_package(value);seal(proof)
                with self.assertRaisesRegex(ValueError,'nonfinal'):verify_protocol_coverage(proof,protocol,chunks,catalog)

    def test_b1_b2_reviewer_actor_model_and_all_author_aliases_are_bound(self):
        protocol,chunks,catalog,base=coverage_fixture()
        for kind in ('different-b1','author-b1','author-final-alias','different-task-model','missing-b1-completion'):
            with self.subTest(kind=kind):
                proof=copy.deepcopy(base);value=proof['fresh_qa_reviews'][0]
                if kind in ('different-b1','author-b1'):
                    value['first_pass_receipt']['reviewer']['agent_id']=(chunks[0]['authored']['author']['agent_id']
                        if kind=='author-b1' else 'another-synthetic-actor')
                    self.sync_first(value)
                elif kind=='author-final-alias':value['artifact']['reviewer']['name']=chunks[0]['authored']['author']['agent_id']
                refresh_package(value)
                if kind=='different-task-model':
                    evidence=value['evidence'];evidence['task_status']['model']='another-actual-model'
                    evidence['task_status_file']=raw(seal({'task_statuses':[{'clientRequestId':evidence['client_request_id'],
                        'result':evidence['task_status']}]}),'/tmp/synthetic-changed-task-model.json')
                elif kind=='missing-b1-completion':value.pop('first_pass_evidence')
                seal(proof)
                with self.assertRaises(ValueError):verify_protocol_coverage(proof,protocol,chunks,catalog)

    def test_separate_actual_stage_tasks_bind_stable_actor_and_unnamed_originals_need_same_task(self):
        protocol,chunks,catalog,proof=coverage_fixture();value=proof['fresh_qa_reviews'][0]
        first=value['first_pass_receipt'];final=value['artifact'];actor='synthetic-independent-stable-reviewer'
        for receipt,label in ((first,'synthetic-B1-task'),(final,'synthetic-B2-task')):
            receipt['reviewer_actor_identity']=actor;receipt['reviewer']['agent_id']=label
        self.sync_first(value);refresh_package(value,'synthetic-B2-task')
        first_package=package(first,'synthetic-B1-task');first_package['evidence']['receipt_file']=value['first_pass_file']
        value['first_pass_evidence']=first_package['evidence'];seal(proof)
        verify_protocol_coverage(proof,protocol,chunks,catalog)
        unnamed=copy.deepcopy(proof);v=unnamed['fresh_qa_reviews'][0]
        for receipt in (v['first_pass_receipt'],v['artifact']):
            receipt.pop('reviewer_actor_identity');receipt['reviewer']={'model':receipt['reviewer']['model'],'role':'Original unnamed actual independent reviewer'}
        self.sync_first(v);refresh_package(v,'synthetic-same-original-task');seal(unnamed)
        verify_protocol_coverage(unnamed,protocol,chunks,catalog)
        other=package(v['first_pass_receipt'],'synthetic-different-original-task')
        other['evidence']['receipt_file']=v['first_pass_file'];v['first_pass_evidence']=other['evidence'];seal(unnamed)
        with self.assertRaisesRegex(ValueError,'same declared'):verify_protocol_coverage(unnamed,protocol,chunks,catalog)

    def test_original_final_decisions_include_negatives_and_need_genuine_completion(self):
        _,chunks,_,proof=coverage_fixture();sources=proof['census']['development_prior_approval_sources']
        negative=copy.deepcopy(sources);p=negative[0]['reviews'][0]
        p['artifact']['records'][0]['status']='needs_revision';refresh_package(p)
        refs=_census_prior_approvals(chunks,negative)
        self.assertFalse(any(r['domain']=='development' and r['id']==p['artifact']['records'][0]['id'] for r in refs))
        for kind in ('empty-development-decisions','duplicate-decision','negative-stale-content','missing-confirmation-completion',
                     'missing-original-row-hash'):
            with self.subTest(kind=kind):
                own_chunks=copy.deepcopy(chunks);own_sources=copy.deepcopy(sources);value=own_sources[0]['reviews'][0]
                if kind=='empty-development-decisions':value['artifact']['records']=[]
                elif kind=='duplicate-decision':value['artifact']['records'].append(copy.deepcopy(value['artifact']['records'][0]))
                elif kind=='negative-stale-content':value['artifact']['records'][0].update(status='needs_revision',reviewed_content_sha256='0'*64)
                elif kind=='missing-confirmation-completion':own_chunks[0].pop('original_review_packages')
                else:value['artifact']['records'][0].pop('reviewed_content_sha256')
                refresh_package(value)
                with self.assertRaises(ValueError):_census_prior_approvals(own_chunks,own_sources)

    def test_external_current_evaluator_pins_and_original_correlate_inventory_are_required(self):
        protocol,chunks,catalog,base=coverage_fixture()
        with self.assertRaisesRegex(ValueError,'Trusted external'):
            _verify_protocol_coverage(base,protocol,chunks,catalog)
        for kind in ('catalog-alias','snapshot-alias','changed-current-pin','custodian-inventory','self-pinned-inventory'):
            with self.subTest(kind=kind):
                proof=copy.deepcopy(base);value=proof['evaluation_custody'];artifact=value['artifact']
                if kind=='catalog-alias':artifact['evaluation_catalog_sha256']='0'*64
                elif kind=='snapshot-alias':artifact['evaluation_snapshot_sha256']='0'*64
                elif kind=='changed-current-pin':
                    artifact['evaluator_input_metadata']['snapshot_binding']['file_sha256']='0'*64
                    seal(artifact['evaluator_input_metadata'])
                else:
                    rid=artifact['flags'][0]['id'];artifact['required_surface_correlate_ids_by_record'][rid]=['invented-correlate']
                    if kind=='self-pinned-inventory':
                        value['required_surface_correlate_ids_by_record'][rid]=['invented-correlate']
                        artifact['flags'][0]['surface_correlate_flags']={'invented-correlate':False}
                refresh_package(value);seal(proof)
                with self.assertRaises(ValueError):verify_protocol_coverage(proof,protocol,chunks,catalog)

    def test_boolean_obligations_and_derived_counts_never_coerce_integer_types(self):
        protocol,chunks,catalog,base=coverage_fixture()
        for kind in ('custody-bool','census-bool','development-bool','derived-count'):
            with self.subTest(kind=kind):
                proof=copy.deepcopy(base)
                if kind=='custody-bool':value=proof['evaluation_custody'];value['artifact']['filtering_permitted']=0
                elif kind=='census-bool':value=proof['census'];value['artifact']['no_semantic_decisions_created']=1
                elif kind=='development-bool':value=proof['development_reclassification'];value['artifact']['development_retrained']=0
                else:value=proof['census'];value['artifact']['contract']['per_sub_kind_counts']['conflict']=False
                refresh_package(value)
                if value is proof['census']:
                    attestation=value['freeze_order_attestation'];attestation['artifact']['census_artifact_sha256']=value['artifact']['sha256']
                    refresh_package(attestation)
                seal(proof)
                with self.assertRaises(ValueError):verify_protocol_coverage(proof,protocol,chunks,catalog)

    def test_original_nested_review_pointers_preserve_native_approve_decisions(self):
        protocol,chunks,catalog,proof=coverage_fixture();value=proof['fresh_qa_reviews'][0]
        original=copy.deepcopy(value['artifact'])
        pointers={}
        for stage,receipt in (('first_pass',value['first_pass_receipt']),('final',value['artifact'])):
            nested={key:receipt.pop(key) for key in ('stage','author_rationales_seen','model_outcomes_seen',
                'evaluator_wording_seen','prior_review_history_seen')}
            if stage=='first_pass':nested['other_reviewer_judgments_seen']=receipt.pop('other_reviewer_judgments_seen')
            receipt['original_isolation']=nested;receipt['native_decisions']=receipt.pop('acquisition_records')
            for row in receipt['native_decisions']:
                row['content_sha256']=row.pop('reviewed_content_sha256');row['status']='approve'
            seal(receipt)
            pointers[stage]={key:'/original_isolation/'+key for key in nested}
            pointers[stage]['acquisition_records']='/native_decisions'
        value['original_review_value_pointers']=pointers
        value['first_pass_file']=raw(value['first_pass_receipt'],'/tmp/synthetic-nested-original-B1.json')
        value['artifact']['first_pass_binding']={'path':value['first_pass_file']['path'],
            'sha256':value['first_pass_receipt']['sha256'],'file_sha256':value['first_pass_file']['file_sha256']}
        refresh_package(value);seal(proof);verify_protocol_coverage(proof,protocol,chunks,catalog)
        self.assertEqual(original['acquisition_records'][0]['reviewed_content_sha256'],value['artifact']['native_decisions'][0]['content_sha256'])
        native=copy.deepcopy(proof);current=native['fresh_qa_reviews'][0]
        for stage,receipt in (('first_pass',current['first_pass_receipt']),('final',current['artifact'])):
            receipt['acquisition_decisions']=receipt.pop('native_decisions');seal(receipt)
            current['original_review_value_pointers'][stage].pop('acquisition_records')
        current['first_pass_file']=raw(current['first_pass_receipt'],'/tmp/synthetic-original-acquisition-decisions-B1.json')
        current['artifact']['first_pass_binding']={'path':current['first_pass_file']['path'],
            'sha256':current['first_pass_receipt']['sha256'],'file_sha256':current['first_pass_file']['file_sha256']}
        refresh_package(current);seal(native);verify_protocol_coverage(native,protocol,chunks,catalog)
        for kind in ('contradiction','invented-field','duplicate-row','missing-stage'):
            with self.subTest(kind=kind):
                changed=copy.deepcopy(proof);current=changed['fresh_qa_reviews'][0]
                if kind=='contradiction':current['artifact']['author_rationales_seen']=True
                elif kind=='invented-field':current['original_review_value_pointers']['final']['invented_approval']='/native_decisions'
                elif kind=='duplicate-row':current['artifact']['native_decisions'].append(current['artifact']['native_decisions'][0])
                else:current['artifact']['original_isolation']['stage']='B1_first_pass'
                refresh_package(current);seal(changed)
                with self.assertRaises(ValueError):verify_protocol_coverage(changed,protocol,chunks,catalog)

    def test_full_symmetric_census_and_freeze_results_fail_closed_on_missing_properties_or_history(self):
        protocol,chunks,catalog,proof=coverage_fixture()
        for kind in ('omit-development-group','deploy-subset','missing-property','tag-without-set','bad-count',
                     'prior-inventory','missing-masked-test','drop-failing-approval','normalization','late-freeze',
                     'missing-trace','unretained-previous-counts','missing-development-history'):
            with self.subTest(kind=kind):
                changed=copy.deepcopy(proof);value=changed['census'];contract=value['artifact']['contract']
                if kind=='omit-development-group':contract['population_snapshot'].pop()
                elif kind=='deploy-subset':value['development_population_sources']['scale_packets'].pop()
                elif kind=='missing-property':contract['property_rows'][0]['properties'].pop('LE1a')
                elif kind=='tag-without-set':contract['property_rows'][0]['properties']['LE1a']=True
                elif kind=='bad-count':contract['per_sub_kind_counts']['alias']=1
                elif kind=='prior-inventory':contract['prior_approval_inventory_sha256']='0'*64
                elif kind=='missing-masked-test':contract['prior_approval_results'].pop()
                elif kind=='drop-failing-approval':contract['prior_approval_results'][0]['masked_C6_passed']=False
                elif kind=='normalization':
                    value['rule_freeze']['artifact']['normalization_reference']['function_text']='Strip articles'
                    refresh_package(value['rule_freeze'])
                elif kind=='late-freeze':contract['chronology']['frozen_utc']='2026-10-08T00:03:00Z'
                elif kind=='missing-trace':value['freeze_order_attestation']['chronology_trace_files']=[]
                elif kind=='unretained-previous-counts':contract['chronology']['prior_counts_viewed_before_current_freeze']=True
                else:value['development_prior_approval_sources'].pop()
                refresh_package(value)
                custody=value['freeze_order_attestation'];custody['artifact']['census_artifact_sha256']=value['artifact']['sha256']
                custody['artifact']['chronology']=contract['chronology'];refresh_package(custody);seal(changed)
                with self.assertRaises((ValueError,KeyError)):verify_protocol_coverage(changed,protocol,chunks,catalog)

    def test_multitagged_census_rows_require_a_matching_set_for_each_eligibility_property(self):
        protocol,chunks,catalog,proof=coverage_fixture();value=proof['census'];contract=value['artifact']['contract']
        old_rule=seal({'test_fixture':True,'schema':'synthetic-prior-rule','rule':'Synthetic earlier property definitions'})
        rule_file=raw(old_rule,'/tmp/synthetic-original-multitag-prior-rule.json')
        rule_ref={'sha256':old_rule['sha256'],'file_sha256':rule_file['file_sha256']}
        value['prior_rule_versions']=[{'artifact':old_rule,'artifact_file':rule_file}]
        contract['retained_prior_rule_versions']=[rule_ref]
        contract['chronology']['retained_prior_rule_versions']=[rule_ref]
        candidates=[r for r in contract['property_rows'] if r['domain']=='development' and r['scope']=='normal' and
                    r['field']=='acquisition_records'][:2]
        for row in candidates:
            row['properties'].update(LE1a=True,LE1b=True)
        # Retained-condition findings do not create lane eligibility or require lane sets.
        contract['property_rows'][-1]['properties'].update(C6_masked=True,ordinary_retained_defect=True)
        members=[{k:r[k] for k in ('domain','scope','field','id')} for r in candidates]
        contract['lane_sets']=[{'set_id':'synthetic-'+key,'property_key':key,'sub_kind':kind,
            'previously_lane_eligible':False,'members':copy.deepcopy(members),
            'source_only_evidence':'Synthetic source assessment for '+key,
            'prior_eligibility_assessment':{'rule_version_sha256':old_rule['sha256'],'eligible':False,
                'source_only_evidence':'Synthetic earlier-rule assessment for '+key}}
            for key,kind in (('LE1a','conflict'),('LE1b','alias'))]
        contract['newly_lane_eligible_sets']=['synthetic-LE1a','synthetic-LE1b']
        contract['per_sub_kind_counts'].update(conflict=1,alias=1)

        def reseal(current):
            census=current['census'];refresh_package(census)
            custody=census['freeze_order_attestation']
            custody['artifact'].update(census_artifact_sha256=census['artifact']['sha256'],
                                       chronology=census['artifact']['contract']['chronology'])
            refresh_package(custody);seal(current)

        reseal(proof)
        result=verify_protocol_coverage(proof,protocol,chunks,catalog)['census_contract']
        self.assertEqual(result['per_sub_kind_counts'],{'conflict':1,'alias':1,'original_label_nonuniqueness':0})
        # The collective union still covers every tagged row after either set is dropped.
        for missing,kind in (('LE1b','alias'),('LE1a','conflict')):
            with self.subTest(missing=missing):
                changed=copy.deepcopy(proof);current=changed['census']['artifact']['contract']
                current['lane_sets']=[s for s in current['lane_sets'] if s['property_key']!=missing]
                current['newly_lane_eligible_sets'].remove('synthetic-'+missing)
                current['per_sub_kind_counts'][kind]=0
                reseal(changed)
                with self.assertRaisesRegex(ValueError,'K6 '+missing+' property tags disagree with matching lane-set membership'):
                    verify_protocol_coverage(changed,protocol,chunks,catalog)

    def test_development_census_uses_full_groups_and_exact_qa_overlay_chain(self):
        sources,_=census_sources_fixture();rows=development_census_snapshot(sources)
        self.assertEqual(len(rows),778)
        self.assertEqual(sum(r['scope']=='normal' and r['field']=='groups' for r in rows),157)
        self.assertEqual(sum(r['scope']=='scale' and r['field']=='acquisition_records' for r in rows),286)
        for kind in ('normal-group','overlay-parent','scale-subset','raw-bytes'):
            with self.subTest(kind=kind):
                changed=copy.deepcopy(sources)
                if kind=='normal-group':current=changed['normal_grounded'];current['artifact']['groups'].pop()
                elif kind=='overlay-parent':current=changed['normal_qa_versions'][1];current['artifact']['parent_review_packet_sha256']='0'*64
                elif kind=='scale-subset':current=changed['scale_packets'][0];current['artifact']['acquisition_records'].pop()
                else:current=changed['normal_grounded'];current['artifact_file']['utf8']+=' '
                if kind!='raw-bytes':seal(current['artifact']);current['artifact_file']=raw(current['artifact'],current['artifact_file']['path'])
                with self.assertRaises(ValueError):development_census_snapshot(changed)

    def test_legacy_development_approval_claims_retain_truthful_missing_transport(self):
        protocol,chunks,catalog,proof=coverage_fixture();value=proof['census'];source=value['development_prior_approval_sources'][0]
        review=source['reviews'][0];original=copy.deepcopy(review['artifact'])
        legacy={'review_packet':source['review_packet'],'review_packet_file':source['review_packet_file'],
            'original_receipt':original,'original_receipt_file':raw(original,'/tmp/synthetic-original-legacy-claim.json'),
            'original_task_completion_verified':False,'completion_evidence_unresolved':'missing_original_task_transport'}
        envelope=legacy_claim_unverified_envelope(legacy)
        self.assertTrue(envelope['proposal_only']);self.assertFalse(envelope['accepted_by_protocol_coverage'])
        self.assertTrue(all(r['approval_kind']=='claimed_approval_unverified' for r in envelope['claims']))
        with self.assertRaises(KeyError):completed_artifact({'artifact':envelope})
        review.pop('evidence');seal(proof)
        with self.assertRaises(KeyError):verify_protocol_coverage(proof,protocol,chunks,catalog)
        for kind in ('claimed-completed','invented-metadata','missing-disclosure','raw-corruption'):
            with self.subTest(kind=kind):
                prior=copy.deepcopy(legacy)
                if kind=='claimed-completed':prior['completion_evidence_unresolved']='completed'
                elif kind=='invented-metadata':prior['original_task_completion_verified']=True
                elif kind=='missing-disclosure':prior.pop('completion_evidence_unresolved')
                else:prior['original_receipt_file']['utf8']='{}'
                with self.assertRaises(ValueError):legacy_claim_unverified_envelope(prior)

    def test_census_derives_nonempty_new_sets_prior_failures_and_retained_rerun(self):
        protocol,chunks,catalog,proof=coverage_fixture();value=proof['census'];contract=value['artifact']['contract']
        old_rule=seal({'test_fixture':True,'schema':'synthetic-prior-rule','rule':'Synthetic earlier stricter property'})
        rule_file=raw(old_rule,'/tmp/synthetic-original-earlier-rule.json')
        rule_ref={'sha256':old_rule['sha256'],'file_sha256':rule_file['file_sha256']}
        value['prior_rule_versions']=[{'artifact':old_rule,'artifact_file':rule_file}]
        old_census=seal({'test_fixture':True,'schema':'synthetic-prior-census','count':1})
        census_file=raw(old_census,'/tmp/synthetic-original-earlier-census.json')
        census_ref={'sha256':old_census['sha256'],'file_sha256':census_file['file_sha256'],
                    'superseded_reason':'Synthetic category change required a complete rerun'}
        value['prior_census_versions']=[{'artifact':old_census,'artifact_file':census_file}]
        contract['retained_prior_rule_versions']=[rule_ref];contract['retained_prior_census_versions']=[census_ref]
        freeze=value['rule_freeze'];freeze['artifact']['prior_count_view_disclosure'].update(
            prior_counts_seen=True,report_sha256s=[old_census['sha256']],known_row_motivation='Synthetic known-row correction disclosed prospectively')
        refresh_package(freeze);contract['rule_freeze_sha256']=freeze['artifact']['sha256']
        candidates=[r for r in contract['property_rows'] if r['domain']=='development' and r['scope']=='normal' and r['field']=='acquisition_records'][:2]
        for row in candidates:row['properties']['LE1c']=True
        members=[{k:r[k] for k in ('domain','scope','field','id')} for r in candidates]
        contract['lane_sets']=[{'set_id':'synthetic-set','property_key':'LE1c','sub_kind':'original_label_nonuniqueness',
            'previously_lane_eligible':False,'members':members,'source_only_evidence':'Synthetic masked assertion assessment',
            'prior_eligibility_assessment':{'rule_version_sha256':old_rule['sha256'],'eligible':False,'source_only_evidence':'Synthetic old-rule recheck'}}]
        contract['newly_lane_eligible_sets']=['synthetic-set'];contract['per_sub_kind_counts']['original_label_nonuniqueness']=1
        first=contract['prior_approval_results'][0];first['masked_C6_passed']=False
        contract['previously_approved_failing_C6_rows']=[first['original_approval_reference']]
        timeline=contract['chronology'];timeline.update(rule_freeze_sha256=freeze['artifact']['sha256'],
            retained_prior_rule_versions=[rule_ref],retained_prior_census_versions=[census_ref],
            prior_counts_viewed_before_current_freeze=True,R2_rerun_completed=True)
        refresh_package(value);custody=value['freeze_order_attestation']
        custody['artifact'].update(rule_freeze_sha256=freeze['artifact']['sha256'],census_artifact_sha256=value['artifact']['sha256'],chronology=timeline)
        refresh_package(custody);seal(proof)
        result=verify_protocol_coverage(proof,protocol,chunks,catalog)['census_contract']
        self.assertEqual(result['per_sub_kind_counts']['original_label_nonuniqueness'],1)
        self.assertEqual(result['previously_approved_failing_C6_count'],1)
        for kind in ('unretained-prior-rule','duplicate-set','wrong-prior-rule','not-rerun'):
            with self.subTest(kind=kind):
                changed=copy.deepcopy(proof);current=changed['census'];c=current['artifact']['contract']
                if kind=='unretained-prior-rule':current['prior_rule_versions']=[]
                elif kind=='duplicate-set':c['lane_sets'].append(copy.deepcopy(c['lane_sets'][0]))
                elif kind=='wrong-prior-rule':c['lane_sets'][0]['prior_eligibility_assessment']['rule_version_sha256']='0'*64
                else:c['chronology']['R2_rerun_completed']=False
                refresh_package(current);att=current['freeze_order_attestation'];att['artifact']['census_artifact_sha256']=current['artifact']['sha256']
                att['artifact']['chronology']=c['chronology'];refresh_package(att);seal(changed)
                with self.assertRaises(ValueError):verify_protocol_coverage(changed,protocol,chunks,catalog)

    def test_original_binding_and_checklist_aliases_preserve_all_declared_seals(self):
        protocol,chunks,catalog,proof=coverage_fixture();value=proof['fresh_qa_reviews'][0];receipt=value['artifact']
        first=receipt.pop('first_pass_binding');first['payload_seal_sha256']=first.pop('sha256')
        first['self_seal']=first['payload_seal_sha256']
        history=receipt.pop('prior_review_supplement_binding');history['content_sha256']=history.pop('sha256')
        receipt['bindings']={'b1_receipt':first,'prior_review_history_binding':history}
        for row in receipt['acquisition_records']:
            for rule in row['retained_conditions'].values():rule.update(result=rule.pop('status'),reason=rule.pop('notes'))
        refresh_package(value);seal(proof);verify_protocol_coverage(proof,protocol,chunks,catalog)
        for kind in ('secondary-seal','raw-file','status-alias','empty-reason'):
            with self.subTest(kind=kind):
                changed=copy.deepcopy(proof);current=changed['fresh_qa_reviews'][0]
                if kind=='secondary-seal':current['artifact']['bindings']['b1_receipt']['self_seal']='0'*64
                elif kind=='raw-file':current['artifact']['bindings']['prior_review_history_binding']['file_sha256']='0'*64
                elif kind=='status-alias':current['artifact']['acquisition_records'][0]['retained_conditions']['R2']['status']='fail'
                else:current['artifact']['acquisition_records'][0]['retained_conditions']['R3']['reason']=''
                refresh_package(current);seal(changed)
                with self.assertRaises(ValueError):verify_protocol_coverage(changed,protocol,chunks,catalog)

    def test_separate_development_runtime_subset_preserves_negative_source_report(self):
        protocol,chunks,catalog,proof=coverage_fixture();value=separate_development_fixture(protocol)
        original=copy.deepcopy(value['artifact']);snapshot=development_source_runtime_snapshot(value)
        self.assertEqual(value['artifact'],original)
        self.assertEqual((len(snapshot['normal']['source']),len(snapshot['normal']['runtime'])),(78,78))
        self.assertEqual((len(snapshot['scale']['source']),len(snapshot['scale']['runtime'])),(286,218))
        self.assertEqual(len(snapshot['scale']['original_observed_scope']['excluded_reviewed_record_ids']),68)
        proof['development_reclassification']=value;proof['development_source_runtime_snapshot']=snapshot
        proof['development_source_runtime_snapshot_sha256']=digest(snapshot);seal(proof)
        verify_protocol_coverage(proof,protocol,chunks,catalog)
        for kind in ('missing-preservation','retraining','wrong-review','wrong-custody','wrong-observation'):
            with self.subTest(kind=kind):
                changed=copy.deepcopy(proof);development=changed['development_reclassification']
                if kind=='missing-preservation':development.pop('runtime_preservation_attestation')
                else:
                    preservation=development['runtime_preservation_attestation'];record=preservation['artifact']
                    field={'retraining':'development_retrained','wrong-review':'source_review_sha256',
                           'wrong-custody':'runtime_custody_sha256','wrong-observation':'runtime_source_observation_sha256'}[kind]
                    record[field]=True if kind=='retraining' else '0'*64;refresh_package(preservation)
                seal(changed)
                with self.assertRaises((ValueError,KeyError)):verify_protocol_coverage(changed,protocol,chunks,catalog)

    def test_development_runtime_metadata_cannot_forge_membership_or_original_execution(self):
        protocol,_,_,_=coverage_fixture();value=separate_development_fixture(protocol)
        for kind in ('raw-bytes','exit','stdout','script','partial-review','excluded-id','wrong-subset-hash','duplicate-trial','trial-pool','loss-of-limitations'):
            with self.subTest(kind=kind):
                changed=copy.deepcopy(value);interface=changed['runtime_observation_interface']
                metadata=json.loads(interface['metadata_stdout_file']['utf8']);execution=json.loads(interface['actual_execution_file']['utf8'])
                if kind=='raw-bytes':interface['metadata_stdout_file']['utf8']+=' '
                elif kind=='script':interface['validator_script_file_sha256']='0'*64
                elif kind=='partial-review':
                    changed['artifact']['rows'].pop();refresh_package(changed)
                else:
                    if kind=='exit':execution['exit_code']=1
                    elif kind=='stdout':execution['output']='another output'
                    elif kind=='excluded-id':metadata['scopes']['scale286']['excluded_reviewed_record_ids'][0]='absent'
                    elif kind=='wrong-subset-hash':metadata['scopes']['scale286']['runtime_source_projection_sha256']='0'*64
                    elif kind=='duplicate-trial':metadata['trials'][-1]=copy.deepcopy(metadata['trials'][0])
                    elif kind=='trial-pool':metadata['trials'][0]['source_review_content_projection_sha256']='0'*64
                    else:metadata['attestations_not_established']=[]
                    text=json.dumps(metadata);interface['metadata_stdout_file'].update(utf8=text,file_sha256=hashlib.sha256(text.encode()).hexdigest())
                    if kind not in ('exit','stdout'):execution['output']=text
                    text=json.dumps(execution);interface['actual_execution_file'].update(utf8=text,file_sha256=hashlib.sha256(text.encode()).hexdigest())
                with self.assertRaises(ValueError):development_source_runtime_snapshot(changed)

    def test_original_child_write_trace_binds_abbreviated_completion_without_status_rewrite(self):
        _,_,_,proof=coverage_fixture();value=proof['fresh_qa_reviews'][0];evidence=value['evidence']
        evidence['receipt_file']['path']='/tmp/synthetic-original-completed-receipt.json'
        status=evidence['task_status'];status['latestTerminalSummary']='Original output '+evidence['receipt_file']['path']+' seal abbreviated'
        snapshot={'task_statuses':[{'clientRequestId':evidence['client_request_id'],'result':status}]}
        text=json.dumps(snapshot);evidence['task_status_file'].update(utf8=text,file_sha256=hashlib.sha256(text.encode()).hexdigest())
        trace=native_trace_fixture(status,value['artifact'],evidence['receipt_file'])
        def envelope(current):
            return json.dumps({'structuredContent':current,'isError':False,'content':[{'type':'text','text':json.dumps(current)}]})
        trace_text=envelope(trace)
        evidence['completion_trace_file']={'utf8':trace_text,'file_sha256':hashlib.sha256(trace_text.encode()).hexdigest(),
                                           'trace_json_pointer':'/structuredContent'}
        original_status=copy.deepcopy(status);completed_artifact(value);self.assertEqual(status,original_status)
        for kind in ('bytes','thread','run','pending','seal','file','duplicate','command-only','nonlocal','truncated','model','seal-not-verified','not-readonly','packing-missing-context'):
            with self.subTest(kind=kind):
                changed=copy.deepcopy(value);current=json.loads(trace_text)['structuredContent']
                if kind=='bytes':changed['evidence']['completion_trace_file']['utf8']+=' '
                else:
                    if kind=='thread':current['thread']['threadId']='different-thread'
                    elif kind=='run':current['items'][0]['runId']='different-run'
                    elif kind=='pending':current['thread']['pendingRequestCount']=1
                    elif kind=='seal':current['items'][0]['text']=current['items'][0]['text'].replace(value['artifact']['sha256'],'0'*64)
                    elif kind=='file':current['items'][0]['text']=current['items'][0]['text'].replace(evidence['receipt_file']['file_sha256'],'0'*64)
                    elif kind=='duplicate':current['items'].append(copy.deepcopy(current['items'][0]))
                    elif kind=='nonlocal':current['items'][0]['visibility']='inherited'
                    elif kind=='truncated':current['items'][0]['textTruncated']=True
                    elif kind=='model':current['recentRuns'][0]['model']='different-model'
                    elif kind=='seal-not-verified':current['items'][0]['text']=current['items'][0]['text'].replace('\"canonical_verified\": true','\"canonical_verified\": false')
                    elif kind=='not-readonly':current['items'][0]['text']=current['items'][0]['text'].replace('0o444','0o644')
                    elif kind=='packing-missing-context':
                        descriptor={'path':evidence['receipt_file']['path'],'sha256':value['artifact']['sha256'],
                                    'file_sha256':evidence['receipt_file']['file_sha256']}
                        report={'packing_output':{},'patch':descriptor,
                                'response':{'path':'/tmp/unbound-response.json','sha256':'1'*64,'file_sha256':'2'*64},
                                'canonical_verified':True,'packing_import_profile_event_count':0,'packing_stderr_non_import_lines':[]}
                        extra=copy.deepcopy(current['items'][1]);extra.pop('text');extra['position']=20
                        extra.update(command="python -I -S -B - <<'PY'\nimport json\nprint(json.dumps("+repr(report)+",indent=2))\nPY",
                                     stdout=json.dumps(report,indent=2));current['items'].append(extra)
                    else:current['items'][0]['text']='$ echo '+value['artifact']['sha256']+' '+evidence['receipt_file']['file_sha256']+' '+evidence['receipt_file']['path']
                    current_text=envelope(current)
                    changed['evidence']['completion_trace_file'].update(utf8=current_text,file_sha256=hashlib.sha256(current_text.encode()).hexdigest())
                if kind=='packing-missing-context':
                    with self.assertRaisesRegex(ValueError,'lacks its original response context'):completed_artifact(changed)
                else:
                    with self.assertRaises(ValueError):completed_artifact(changed)

    def test_development_binds_own_source_runtime_inputs_without_confirmation_corpus_visibility(self):
        protocol,chunks,catalog,proof=coverage_fixture()
        self.assertNotIn('source_snapshot_sha256',proof['development_reclassification']['artifact'])
        self.assertEqual(proof['development_source_runtime_snapshot']['domain'],'development_source_runtime_v1')
        self.assertEqual(proof['development_source_runtime_snapshot']['normal']['source']['records'],78)
        self.assertEqual(proof['development_source_runtime_snapshot']['scale']['source']['records'],286)
        verify_protocol_coverage(proof,protocol,chunks,catalog)
        for kind in ('source','runtime','domain','missing-domain'):
            with self.subTest(kind=kind):
                changed=copy.deepcopy(proof)
                if kind in ('source','runtime'):
                    changed['development_source_runtime_snapshot']['normal'][kind]['different_parent_value']=True
                    changed['development_source_runtime_snapshot_sha256']=digest(changed['development_source_runtime_snapshot'])
                elif kind=='domain':
                    changed['development_source_runtime_snapshot']['domain']='confirmation_source_snapshot'
                    changed['development_source_runtime_snapshot_sha256']=digest(changed['development_source_runtime_snapshot'])
                else:changed['development_reclassification']['development_source_runtime_snapshot_pointers'].pop('scale')
                seal(changed)
                with self.assertRaises(ValueError):verify_protocol_coverage(changed,protocol,chunks,catalog)

    def test_exact_raw_calltoolresult_status_pointer_preserves_original_bytes(self):
        _, _, _, proof = coverage_fixture();value=proof['fresh_qa_reviews'][0]
        status=value['evidence']['task_status'];name=value['evidence']['client_request_id']
        status['taskId']='synthetic:delegate-task:'+name
        snapshot={'content':[{'type':'text','text':json.dumps(status)}], 'structuredContent':status, 'isError':False}
        text=json.dumps(snapshot,indent=2)
        value['evidence']['task_status_file'].update(utf8=text,file_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                                   status_json_pointer='/structuredContent')
        artifact,_=completed_artifact(value)
        self.assertEqual(artifact['sha256'],value['artifact']['sha256'])
        self.assertEqual(value['evidence']['task_status_file']['utf8'],text)
        value['evidence']['task_status_file']['status_json_pointer']='/content/0'
        with self.assertRaises(ValueError):completed_artifact(value)

    def test_failed_or_omitted_rule_cannot_be_replaced_by_parent_coverage_claim(self):
        for kind in ('failed','omitted','unexplained','conflicting-alias'):
            with self.subTest(kind=kind):
                protocol,chunks,catalog,proof=coverage_fixture();value=proof['fresh_qa_reviews'][0]
                row=value['artifact']['acquisition_records'][0]
                if kind=='failed':row['retained_conditions']['R2']['status']='fail'
                elif kind=='omitted':row['retained_conditions'].pop('R5')
                elif kind=='unexplained':row['retained_conditions']['R6']['notes']=''
                else:
                    row['checklist']=copy.deepcopy(row['retained_conditions']);row['checklist']['R8']['status']='fail'
                proof['all_rules_passed']=True;refresh_package(value);seal(proof)
                with self.assertRaises(ValueError):verify_protocol_coverage(proof,protocol,chunks,catalog)

    def test_blind_stage_custody_and_actual_history_binding_fail_closed(self):
        for kind in ('b1-history-visible','b1-evaluator-visible','history-bytes','history-binding','first-binding','stale-history'):
            with self.subTest(kind=kind):
                protocol,chunks,catalog,proof=coverage_fixture();value=proof['fresh_qa_reviews'][0]
                if kind.startswith('b1-'):
                    value['first_pass_receipt']['prior_review_history_seen' if kind=='b1-history-visible' else 'evaluator_wording_seen']=True
                    seal(value['first_pass_receipt']);value['first_pass_file']=raw(value['first_pass_receipt'],value['first_pass_file']['path'])
                    value['artifact']['first_pass_binding'].update(sha256=value['first_pass_receipt']['sha256'],file_sha256=value['first_pass_file']['file_sha256'])
                elif kind=='history-bytes':value['prior_review_supplement_file']['utf8']+=' '
                elif kind=='history-binding':value['artifact']['prior_review_supplement_binding']['file_sha256']='0'*64
                elif kind=='first-binding':value['artifact']['first_pass_binding']['sha256']='0'*64
                else:
                    value['prior_review_supplement']['current_review_packet_sha256']='0'*64;seal(value['prior_review_supplement'])
                    value['prior_review_supplement_file']=raw(value['prior_review_supplement'],value['prior_review_supplement_file']['path'])
                    value['artifact']['prior_review_supplement_binding'].update(sha256=value['prior_review_supplement']['sha256'],file_sha256=value['prior_review_supplement_file']['file_sha256'])
                refresh_package(value);seal(proof)
                with self.assertRaises(ValueError):verify_protocol_coverage(proof,protocol,chunks,catalog)

    def test_b1_fail_to_b2_approval_requires_real_written_error_finding(self):
        protocol,chunks,catalog,proof=coverage_fixture();value=proof['fresh_qa_reviews'][0]
        value['first_pass_receipt']['acquisition_records'][0]['status']='needs_revision'
        seal(value['first_pass_receipt']);value['first_pass_file']=raw(value['first_pass_receipt'],value['first_pass_file']['path'])
        value['artifact']['first_pass_binding'].update(sha256=value['first_pass_receipt']['sha256'],file_sha256=value['first_pass_file']['file_sha256'])
        refresh_package(value);seal(proof)
        with self.assertRaisesRegex(ValueError,'written error finding'):verify_protocol_coverage(proof,protocol,chunks,catalog)
        value['artifact']['acquisition_records'][0]['b1_error_finding']='Explicit synthetic error: wrong answer-slot comparison in B1.'
        refresh_package(value);seal(proof)
        verify_protocol_coverage(proof,protocol,chunks,catalog)

    def test_group_approval_cannot_carry_over_changed_original_source_context(self):
        protocol,chunks,catalog,proof=coverage_fixture();value=proof['group_reviews'][0]
        event=value['review_packet']['groups'][0]['event'];value['review_packet']['factsheets'][event]['text']+=' Different source context.'
        seal(value['review_packet']);value['review_packet_file']=raw(value['review_packet'],value['review_packet_file']['path'])
        value['artifact']['review_packet_sha256']=value['review_packet']['sha256'];refresh_package(value);seal(proof)
        with self.assertRaisesRegex(ValueError,'group approval/context'):verify_protocol_coverage(proof,protocol,chunks,catalog)

    def test_census_and_evaluation_custodian_cannot_share_a_source_review_task(self):
        for field in ('census','evaluation_custody'):
            with self.subTest(field=field):
                protocol,chunks,catalog,proof=coverage_fixture()
                reviewer=proof['fresh_qa_reviews'][0]['evidence']['client_request_id']
                refresh_package(proof[field],reviewer);seal(proof)
                with self.assertRaises(ValueError):verify_protocol_coverage(proof,protocol,chunks,catalog)

    def test_custody_requires_every_explicit_boolean_correlate_and_original_root_count(self):
        for kind in ('omit-flag','nonbool-flag','omit-count','change-count','omit-correlate-population','extra-correlate','different-dependency'):
            with self.subTest(kind=kind):
                protocol,chunks,catalog,proof=coverage_fixture();value=proof['evaluation_custody']
                row=value['artifact']['flags'][0]
                if kind=='omit-flag':row.pop('canonical_or_paraphrase_overlap_flag')
                elif kind=='nonbool-flag':row['canonical_or_paraphrase_overlap_flag']=0
                elif kind=='omit-count':value['artifact']['canonical_root_counts'].pop(next(iter(value['artifact']['canonical_root_counts'])))
                elif kind=='change-count':value['artifact']['canonical_root_counts'][next(iter(value['artifact']['canonical_root_counts']))]+=1
                elif kind=='omit-correlate-population':value['required_surface_correlate_ids_by_record'].pop(row['id'])
                elif kind=='extra-correlate':row['surface_correlate_flags']['uncertified']=False
                else:value['dependency_context']={**value['dependency_context'],'exception_lanes_sha256':'0'*64}
                refresh_package(value);seal(proof)
                with self.assertRaises(ValueError):verify_protocol_coverage(proof,protocol,chunks,catalog)

    def test_census_stale_or_partial_actual_record_contexts_fail_closed(self):
        for kind in ('stale','partial','undeclared-pointer'):
            with self.subTest(kind=kind):
                protocol,chunks,catalog,proof=coverage_fixture();value=proof['census']
                if kind=='stale':value['artifact']['rows'][0]['context_sha256']='0'*64
                elif kind=='partial':value['record_binding_pointers'].pop()
                else:value['record_binding_pointers'][0]={'field':'groups','pointers':{'id':'/rows/0/id'}}
                refresh_package(value);seal(proof)
                with self.assertRaises(ValueError):verify_protocol_coverage(proof,protocol,chunks,catalog)

    def test_preregistration_guard_validates_actual_frozen_policy_and_freeze_order(self):
        from spacing_rerun.confirmation import verify_confirmation_dependency_preregistration
        prereg={'confirmation_dependency_policy_path':'/synthetic/policy.json','confirmation_dependency_policy_sha256':'a'*64,
                'sourceqa_protocol_sha256':'b'*64,'frozen_utc':'2026-10-08T09:00:00+00:00'}
        manifest={'sha256':'c'*64,'confirmation_audit':{'source_chunks':['synthetic-chunks'],'source_catalog':{'synthetic':True}},
                  'config':{'sourceqa_protocol_sha256':'b'*64}}
        with patch('spacing_rerun.confirmation_dependency.verify_frozen_policy',return_value={'frozen_utc':'2026-10-08T08:00:00+00:00'}) as frozen:
            with patch('spacing_rerun.confirmation_dependency.verify_preregistered_dependencies') as verifier:
                verify_confirmation_dependency_preregistration(prereg,manifest)
                frozen.assert_called_once_with('/synthetic/policy.json','a'*64,['synthetic-chunks'],{'synthetic':True},manifest_sha256='c'*64)
                verifier.assert_called_once()
                frozen.return_value={'frozen_utc':'2026-10-08T10:00:00+00:00'}
                with self.assertRaisesRegex(ValueError,'frozen after'):verify_confirmation_dependency_preregistration(prereg,manifest)
        missing={'sourceqa_protocol_sha256':'b'*64}
        with self.assertRaisesRegex(ValueError,'actual frozen dependency'):verify_confirmation_dependency_preregistration(missing,manifest)

    def test_complete_fresh_coverage_proves_true_empty_fixed_seed_audit_population(self):
        protocol, chunks, catalog, proof = coverage_fixture()
        result = verify_protocol_coverage(proof, protocol, chunks, catalog, dependency_context=proof['dependency_context'])
        self.assertEqual(result['fresh_qa_count'], 80)
        self.assertEqual(result['carryover_qa_count'], 0)
        self.assertTrue(proof['carryover_audit']['empty_population'])
        self.assertFalse(result['semantic_approvals_created'])

    def test_missing_current_review_cannot_be_hidden_by_empty_audit_or_claimed_parent_pass(self):
        protocol, chunks, catalog, proof = coverage_fixture()
        proof['fresh_qa_reviews'].pop();proof['coverage_complete'] = True;seal(proof)
        with self.assertRaisesRegex(ValueError, 'Full current fresh QA'):
            verify_protocol_coverage(proof, protocol, chunks, catalog)

    def test_actual_conditional_row_cannot_be_routed_as_ordinary_fresh(self):
        protocol, chunks, catalog, proof = coverage_fixture()
        proof['conditional_acquisition_record_ids'] = [];seal(proof)
        with self.assertRaisesRegex(ValueError, 'conditional review'):
            verify_protocol_coverage(proof, protocol, chunks, catalog)

    def test_context_and_original_source_mutations_invalidate_coverage(self):
        for kind in ('factsheet', 'coevent-question', 'group-disposition'):
            with self.subTest(kind=kind):
                protocol, chunks, catalog, proof = coverage_fixture();c = chunks[0]
                if kind == 'factsheet':
                    event = c['review_packet']['events'][0];c['review_packet']['factsheets'][event]['text'] += ' changed'
                elif kind == 'coevent-question':c['review_packet']['acquisition_records'][0]['questions'][0] += ' changed'
                else:c['review_packet']['source_preservation_protocol']['source_dispositions'].append(
                    {'unit_id': c['review_packet']['source_units'][0]['unit_id'], 'status': 'preserve_source_conflict'})
                seal(c['review_packet'])
                with self.assertRaises(ValueError):verify_protocol_coverage(proof, protocol, chunks, catalog)

    def test_task_receipt_pending_and_raw_bytes_fail_closed(self):
        for kind in ('pending', 'latest-failed', 'different-summary', 'raw'):
            with self.subTest(kind=kind):
                _, _, _, proof = coverage_fixture();p = proof['fresh_qa_reviews'][0]
                if kind == 'raw':p['evidence']['receipt_file']['utf8'] += ' '
                else:
                    status = p['evidence']['task_status']
                    if kind == 'pending':status['hasPendingChildRuns'] = True
                    elif kind == 'latest-failed':status['latestTerminalStatus'] = 'failed'
                    else:status['latestTerminalSummary'] = 'Different result'
                    text = json.dumps({'task_statuses': [{'clientRequestId': p['evidence']['client_request_id'], 'result': status}]})
                    p['evidence']['task_status_file'].update(utf8=text, file_sha256=hashlib.sha256(text.encode()).hexdigest())
                with self.assertRaises(ValueError):completed_artifact(p)

    def test_evaluation_custody_cannot_impute_missing_rows_or_flags_or_root_counts(self):
        protocol, chunks, catalog, proof = coverage_fixture()
        p = proof['evaluation_custody'];p['artifact']['flags'].pop();seal(p['artifact'])
        refresh_package(p,'synthetic-separate-evaluation-custodian');seal(proof)
        with self.assertRaisesRegex(ValueError, 'flag coverage'):
            verify_protocol_coverage(proof, protocol, chunks, catalog)

    def test_fixed_seed_sampling_and_chunk_or_global_escalation(self):
        populations = {i: [f'c{i}-{n:03}' for n in range(100)] for i in range(1, 9)}
        plan = seeded_audit_plan(populations)
        self.assertTrue(all(len(ids) == 30 for ids in plan['sample_by_chunk'].values()))
        self.assertEqual(plan, seeded_audit_plan(populations))
        one = seeded_audit_plan(populations, [plan['sample_by_chunk']['1'][0]])
        self.assertEqual(one['escalated_record_ids'], populations[1])
        two = seeded_audit_plan(populations, [plan['sample_by_chunk']['1'][0], plan['sample_by_chunk']['2'][0]])
        self.assertEqual(len(two['escalated_record_ids']), 800)
        with self.assertRaises(ValueError):seeded_audit_plan(populations, ['outside-frozen-sample'])
