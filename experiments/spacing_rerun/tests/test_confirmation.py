"""Explicit synthetic audit fixtures; no record here certifies actual data."""
import copy
import unittest

from spacing_rerun.acquisition import acquisition_identity
from spacing_rerun.common import digest
from spacing_rerun.confirmation import (AUDIT_SCHEMA, PROTOCOL, QA_SHA256, assign_confirmation_roles,
                                       source_review_content, evaluation_review_content, validate_catalog, validate_confirmation_audit)
from spacing_rerun.data import REVISION
from spacing_rerun.grounding import FACTSHEETS_SHA256, group_identity
from spacing_rerun.units import unit_identity


def seal(payload):
    payload['sha256'] = digest({k: v for k, v in payload.items() if k != 'sha256'})
    return payload


def person(name):
    return {'agent_id': name, 'provider': 'synthetic-test-provider', 'model': 'fixture-model'}


def confirmation_fixture():
    partition = seal({'seed': 20261007, 'confirmation': [f'event_{i:03}' for i in range(80)],
                      'development': [f'event_{i:03}' for i in range(80, 100)]})
    facts, units, maps, factsheets, groups, sources, qa, evaluation = [], [], {}, {}, [], [], [], []
    for i, event in enumerate(partition['confirmation']):
        answer, pid = f'Value{i}', f'canonical-{i}'
        statement, question = f'Event {i} stored {answer}.', f'Which value was stored in event {i}?'
        opaque = 'probe-' + digest([REVISION, pid])
        uid = unit_identity(event, statement)[0]
        gid = group_identity([uid])
        excerpt = {'line': 0, 'quote': statement}
        factsheets[event] = {'text': statement, 'sha256': digest(statement)}
        fact = {'id': pid, 'event': event, 'style': 'blog', 'statement': statement, 'question': question,
                'answer': answer, 'members': [pid], 'aliases': [answer]}
        facts.append(fact)
        probe = {'opaque_probe_id': opaque, 'opaque_duplicate_cluster_members': [opaque],
                 'source_answer_label': answer, 'source_style': 'blog'}
        unit = seal({'unit_id': uid, 'event': event, 'source_assertion': statement,
                     'canonical_answer_labels': [answer], 'canonical_probes': [probe]})
        units.append(unit)
        maps[opaque] = {'original_probe_id': pid, 'event': event, 'source_unit_id': uid}
        groups.append({'id': gid, 'event': event, 'source_unit_ids': [uid], 'statement': statement,
                       'grouping_rationale': 'Synthetic singleton', 'evidence': [excerpt], 'review_status': 'unreviewed'})
        sources.append({'unit_id': uid, 'source_unit_sha256': unit['sha256'], 'event': event,
                        'source_assertion': statement, 'opaque_probe_ids': [opaque], 'group_id': gid,
                        'status': 'grounded', 'conflict_rationale': '', 'nonliteral_answer_labels': [],
                        'evidence': [excerpt], 'review_status': 'unreviewed'})
        qa.append({'id': acquisition_identity(gid, answer), 'unit_id': gid, 'event': event, 'answer': answer,
                   'questions': [f'Give the source value assigned to event {i}?', f'Name the recorded entry for event {i}?'],
                   'evidence': [excerpt], 'rationale': 'Synthetic source QA semantics',
                   'source_conflict_status': 'supported', 'review_status': 'unreviewed'})
        evaluation.append({'id': pid, 'opaque_probe_id': opaque, 'event': event, 'source_unit_id': uid,
                           'canonical_question': question, 'canonical_answer': answer, 'aliases': [],
                           'original_source_assertion': statement,
                           'paraphrase': f'State the value stored for event {i}?', 'evidence': [excerpt],
                           'source_ambiguity_status': 'supported', 'review_status': 'unreviewed'})
    catalog = seal({'schema': 'spacing-confirmation-source-catalog-v1', 'dataset_revision': REVISION,
                    'partition': partition, 'outer_partition_sha256': partition['sha256'],
                    'pinned_parquet_sha256': {'fict_qa': QA_SHA256, 'fictsheets': FACTSHEETS_SHA256},
                    'role_independent': True, 'human_review_complete': False, 'units': units, 'factsheets': factsheets,
                    'coverage': {'events': 80, 'source_units': 80, 'canonical_probes': 80, 'source_answer_targets': 80},
                    'test_fixture': True})
    probe_map = seal({'source_catalog_sha256': catalog['sha256'], 'probes': maps})
    packet = seal({'schema': 'source-only-confirmation-authoring-input-v1', 'source_catalog_sha256': catalog['sha256'],
                   'chunk_index': 1, 'events': partition['confirmation'], 'units': units, 'factsheets': factsheets})
    author = person('fixture-source-author')
    authored = seal({'schema': 'spacing-confirmation-authored-source-chunk-v1', 'dataset_revision': REVISION,
                     'source_catalog_sha256': catalog['sha256'], 'source_author_packet_sha256': packet['sha256'],
                     'chunk_index': 1, 'events': packet['events'], 'author': author,
                     'authoring_inputs': ['source_assertions', 'source_answer_labels', 'event_factsheets'],
                     'evaluation_question_text_included': False, 'model_outcomes_used': False,
                     'human_review_complete': False, 'independent_agent_review_complete': False,
                     'source_units': sources, 'groups': groups, 'acquisition_records': qa})
    review_packet = seal({'schema': 'spacing-source-independent-review-packet-v1',
                          'source_catalog_sha256': catalog['sha256'], 'source_author_packet_sha256': packet['sha256'],
                          'authored_chunk_sha256': authored['sha256'], 'chunk_index': 1, 'events': packet['events'],
                          'author_rationales_included': False, 'factsheets': factsheets,
                          'source_units': [{k: u[k] for k in ('unit_id', 'event', 'source_assertion', 'canonical_answer_labels', 'canonical_probes', 'sha256')}
                                           for u in units],
                          'groups': [source_review_content(row, 'groups') for row in groups],
                          'acquisition_records': [source_review_content(row, 'acquisition_records') for row in qa]})
    receipt = {'schema': 'spacing-source-semantic-review-v1', 'reviewer': person('fixture-source-reviewer'),
               'review_packet_sha256': review_packet['sha256'], 'author_rationales_seen': False, 'model_outcomes_seen': False}
    for field in ('groups', 'acquisition_records'):
        receipt[field] = [{'id': row['id'], 'status': 'approved', 'reviewed_content_sha256': digest(row),
                           'notes': 'Synthetic test review decision'} for row in review_packet[field]]
    seal(receipt)
    epacket = seal({'schema': 'source-only-evaluation-authoring-input-v1', 'records': [
        {key: row[key] for key in ('id', 'opaque_probe_id', 'event', 'source_unit_id', 'canonical_question', 'canonical_answer')}
        for row in evaluation]})
    eauthored = seal({'schema': 'spacing-confirmation-evaluation-catalog-v1', 'dataset_revision': REVISION,
                      'source_catalog_sha256': catalog['sha256'], 'outer_partition_sha256': partition['sha256'],
                      'evaluation_author_packet_sha256': epacket['sha256'], 'author': person('fixture-evaluation-author'),
                      'authoring_inputs': ['canonical_evaluation_questions', 'canonical_answers', 'original_source_assertions', 'event_factsheets'],
                      'acquisition_content_seen': False, 'model_outcomes_used': False, 'human_review_complete': False,
                      'independent_agent_review_complete': False, 'records': evaluation})
    erpacket = seal({'schema': 'spacing-evaluation-independent-review-packet-v1', 'source_catalog_sha256': catalog['sha256'],
                     'evaluation_catalog_sha256': eauthored['sha256'], 'author_packet_sha256': epacket['sha256'],
                     'author_rationales_included': False, 'acquisition_content_included': False, 'model_outcomes_included': False,
                     'records': [evaluation_review_content(row) for row in evaluation]})
    ereceipt = seal({'schema': 'spacing-evaluation-semantic-review-v1', 'reviewer': person('fixture-evaluation-reviewer'),
                     'review_packet_sha256': erpacket['sha256'], 'author_rationales_seen': False, 'model_outcomes_seen': False,
                     'acquisition_content_seen': False, 'human_review_complete': False,
                     'records': [{'id': row['id'], 'status': 'approved', 'reviewed_content_sha256': digest(row),
                                  'notes': 'Synthetic evaluation review decision'} for row in erpacket['records']]})
    candidates = seal({'schema': 'spacing-source-entity-candidates-v1', 'repeated_entity_candidates': [],
                       'answer_label_overlap_candidates': [], 'exact_assertion_candidates': []})
    event_rows = [{'event': event, 'factsheet': sheet, 'factsheet_metadata': {},
                   'source_assertions': [{'unit_id': unit['unit_id'], 'source_assertion': unit['source_assertion']}
                                         for unit in units if unit['event'] == event]} for event, sheet in factsheets.items()]
    event_rows += [{'event': event, 'factsheet': {'text': 'Development fixture.', 'sha256': digest('Development fixture.')},
                    'factsheet_metadata': {}, 'source_assertions': []} for event in partition['development']]
    cpacket = seal({'schema': 'source-only-cross-entity-authoring-input-v1', 'source_catalog_sha256': catalog['sha256'],
                    'dataset_revision': REVISION, 'outer_partition_sha256': partition['sha256'],
                    'entity_candidates_sha256': candidates['sha256'], 'events': event_rows})
    refs = [{'event': event, 'source_field': 'factsheet', 'line': 0, 'quote': factsheets[event]['text'],
             'source_field_sha256': factsheets[event]['sha256']} for event in ('event_000', 'event_071')]
    candidate = {'candidate_id': 'fixture-cross-link', 'events': ['event_000', 'event_071'], 'blocking_objection': True,
                 'disposition': 'unresolved_identity', 'rationale': 'Synthetic unresolved source identity', 'evidence_refs': refs}
    creview = seal({'schema': 'spacing-cross-entity-semantic-review-v1', 'reviewer': person('fixture-cross-reviewer'),
                    'source_packet_sha256': cpacket['sha256'], 'entity_candidates_sha256': candidates['sha256'],
                    'dataset_revision': REVISION, 'outer_partition_sha256': partition['sha256'],
                    'human_review_complete': False, 'model_outcomes_seen': False, 'evaluation_questions_seen': False,
                    'author_judgments_seen': False, 'independent_agent_review_complete': True, 'semantic_audit_complete': True,
                    'reviewed_event_ids': [row['event'] for row in event_rows], 'reviewed_events': 100,
                    'candidates': [], 'extra_candidates': [candidate], 'unresolved_same_proposition_objections': ['fixture-cross-link'],
                    'unresolved_identity_limitations': ['fixture-cross-link'],
                    'required_same_role_components': [{'events': ['event_000', 'event_071']}],
                    'unresolved_objections': True})
    adjudication = seal({'schema': 'spacing-cross-event-adjudication-v1', 'reviewer': person('fixture-cross-adjudicator'),
                         'source_packet_sha256': cpacket['sha256'], 'entity_candidates_sha256': candidates['sha256'],
                         'cross_review_sha256': creview['sha256'], 'human_review_complete': False, 'model_outcomes_seen': False,
                         'decisions': [{'candidate_id': candidate['candidate_id'], 'status': 'approved_with_same_role_protection',
                                        'candidate_content_sha256': digest(candidate), 'events': candidate['events'], 'notes': 'Synthetic protection approval',
                                        'source_identity_ambiguity_retained': True, 'merge_source_assertions': False,
                                        'change_gold_or_evaluation': False, 'evidence_refs': refs}]})
    audit = seal({'schema': AUDIT_SCHEMA, 'protocol_schema': PROTOCOL, 'human_review_complete': False, 'model_outcomes_used': False,
                  'source_catalog': catalog, 'probe_map': probe_map,
                  'source_chunks': [{'author_packet': packet, 'authored': authored, 'review_packet': review_packet, 'reviews': [receipt]}],
                  'evaluation_chunks': [{'author_packet': epacket, 'authored': eauthored, 'review_packet': erpacket, 'reviews': [ereceipt]}],
                  'cross': {'source_packet': cpacket, 'entity_candidates': candidates, 'review': creview, 'adjudication': adjudication},
                  'test_fixture': True})
    return facts, audit


class ConfirmationAuditTests(unittest.TestCase):
    def test_native_revision_task_binds_observed_result_and_rejects_channel_tampering(self):
        facts,audit=confirmation_fixture();chunk=audit['source_chunks'][0]
        who={'agent_id':'/root/test-native-source-author','provider':'openai','model':'gpt-6.1-sol'}
        chunk['authored']['revision_author']=who;seal(chunk['authored'])
        chunk['review_packet']['authored_chunk_sha256']=chunk['authored']['sha256'];seal(chunk['review_packet'])
        chunk['reviews'][0]['review_packet_sha256']=chunk['review_packet']['sha256'];seal(chunk['reviews'][0])
        result={'authored_sha256':chunk['authored']['sha256'],'author_packet_sha256':chunk['author_packet']['sha256'],
                'input_channel':'source_only','identity':who}
        task={'schema':'spacing-native-author-task-evidence-v1','canonical_task_name':who['agent_id'],'model':who['model'],
              'status':'completed','actual_result':result,'actual_result_text':'Explicit synthetic observed result.',
              'result_source':'Synthetic fixture, no actual certification','test_fixture':True}
        chunk['author_task_provenance']=[{'role':'revision_author','task_id':who['agent_id'],'task_record':task,**result}]
        audit['author_task_inventory']={'schema':'p4-actual-t3-author-task-inventory-v1','records':[task],'test_fixture':True};seal(audit)
        self.assertTrue(validate_confirmation_audit(audit,source_facts=facts)['independent_agent_review_complete'])
        task['actual_result']['input_channel']='evaluation_only';seal(audit)
        with self.assertRaisesRegex(ValueError,'native author result'):
            validate_confirmation_audit(audit,source_facts=facts)

    def test_derived_source_flag_cleanup_retains_prior_semantics_and_actual_task(self):
        facts,audit=confirmation_fixture();chunk=audit['source_chunks'][0]
        prior=copy.deepcopy(chunk['authored']);prior['source_units'][0]['nonliteral_answer_labels']=['incorrect-flag'];seal(prior)
        who={'agent_id':'fixture-derived-flag-author','provider':'openai','model':'gpt-6.1-sol',
             'role':'derived_flag_metadata_only','task_id':'test-source-derived-flags-cleanup-fixture'}
        current=chunk['authored'];current.update(prior_authored_sha256=prior['sha256'],revision_author=who,
            approval_granted=False,metadata_derivation={'revision_kind':'derived_flag_metadata_only','metadata_only':True,'task_id':who['task_id']})
        seal(current);chunk['prior_authored']=prior
        chunk['review_packet']['authored_chunk_sha256']=current['sha256'];seal(chunk['review_packet'])
        chunk['reviews'][0]['review_packet_sha256']=chunk['review_packet']['sha256'];seal(chunk['reviews'][0])
        task={'task_id':who['task_id'],'status':'completed','client_request_id':who['task_id'],'child_thread_id':'synthetic-thread',
              'child_run_id':'synthetic-run','model':who['model'],'test_fixture':True}
        chunk['author_task_provenance']=[{'role':'revision_author','task_id':task['task_id'],'task_record':task,'identity':who,
            'authored_sha256':current['sha256'],'author_packet_sha256':chunk['author_packet']['sha256'],'input_channel':'source_only'}]
        audit['author_task_inventory']={'schema':'p4-actual-t3-author-task-inventory-v1','records':[task],'test_fixture':True};seal(audit)
        self.assertTrue(validate_confirmation_audit(audit,source_facts=facts)['independent_agent_review_complete'])
        current['acquisition_records'][0]['questions'][0]='Changed wording during claimed metadata cleanup';seal(current);seal(audit)
        with self.assertRaisesRegex(ValueError,'flag cleanup changed semantic content'):
            validate_confirmation_audit(audit,source_facts=facts)

    def test_actual_visibility_aliases_and_named_reviewer_fields_remain_bound(self):
        facts,audit=confirmation_fixture();receipt=audit['source_chunks'][0]['reviews'][0]
        receipt['reviewer']={'review_name':'synthetic named source reviewer','provider':'anthropic','model':'fixture-model','human':False}
        receipt['disclosures']={'author_rationales_seen':receipt.pop('author_rationales_seen'),
                                'model_outcomes_seen':receipt.pop('model_outcomes_seen'),'human_review_complete':False}
        seal(receipt);seal(audit)
        self.assertTrue(validate_confirmation_audit(audit,source_facts=facts)['independent_agent_review_complete'])
        receipt['author_rationales_used']=True;seal(receipt);seal(audit)
        with self.assertRaisesRegex(ValueError,'visibility aliases contradict'):
            validate_confirmation_audit(audit,source_facts=facts)

    def test_transparent_metadata_annotation_cleanup_retains_all_semantic_versions(self):
        facts,audit=confirmation_fixture();chunk=audit['source_chunks'][0]
        original=copy.deepcopy(chunk['authored'])
        original['source_units'][0]['opaque_probe_ids'].append('duplicate-member-metadata');seal(original)
        metadata=copy.deepcopy(chunk['authored'])
        metadata['metadata_correction']={'kind':'canonical_opaque_probe_membership_only','original_authored_sha256':original['sha256']};seal(metadata)
        prior=copy.deepcopy(metadata)
        prior['groups'][0]['statement']='Source reports that '+prior['groups'][0]['statement'];seal(prior)
        current=copy.deepcopy(prior);current.pop('metadata_correction')
        person={'agent_id':'/root','provider':'openai','model':'gpt-6.1-sol','role':'provenance_metadata_only'}
        current.update(prior_authored_sha256=prior['sha256'],retained_metadata_correction={
            'metadata_only_payload_sha256':metadata['sha256'],'original_authored_sha256':original['sha256']},revision_author=person)
        seal(current);chunk.update(authored=current,prior_authored=prior,metadata_only_authored=metadata,original_authored=original)
        packet=chunk['review_packet'];packet['authored_chunk_sha256']=current['sha256']
        packet['groups']=[source_review_content(row,'groups') for row in current['groups']];seal(packet)
        receipt=chunk['reviews'][0];receipt['review_packet_sha256']=packet['sha256']
        receipt['groups']=[{'id':row['id'],'status':'approved','notes':'Synthetic review decision','reviewed_content_sha256':digest(row)} for row in packet['groups']];seal(receipt)
        task={'task_id':'test-metadata-task','status':'completed','client_request_id':'test-source01-metadata-provenance-cleanup-fixture',
              'child_thread_id':'test-thread','child_run_id':'test-run','model':'gpt-6.1-sol','test_fixture':True}
        chunk['author_task_provenance']=[{'role':'revision_author','task_id':task['task_id'],'task_record':task,'identity':person,
            'authored_sha256':current['sha256'],'author_packet_sha256':current['source_author_packet_sha256'],'input_channel':'source_only'}]
        audit['author_task_inventory']={'schema':'p4-actual-t3-author-task-inventory-v1','records':[task],'test_fixture':True};seal(audit)
        self.assertTrue(validate_confirmation_audit(audit,source_facts=facts)['independent_agent_review_complete'])
        current['groups'][0]['statement']='Different cleanup semantic payload';seal(current);seal(audit)
        with self.assertRaisesRegex(ValueError,'cleanup changed semantic payload'):
            validate_confirmation_audit(audit,source_facts=facts)

    def test_per_probe_original_assertion_is_exact_while_normalized_unit_representative_can_differ(self):
        facts,audit=confirmation_fixture();chunk=audit['evaluation_chunks'][0]
        raw=facts[0]['statement'].rstrip('.')
        facts[0]['statement']=raw
        chunk['authored']['records'][0]['original_source_assertion']=raw
        chunk['author_packet']['records'][0]['original_source_assertion']=raw
        seal(chunk['author_packet'])
        chunk['authored']['evaluation_author_packet_sha256']=chunk['author_packet']['sha256']
        chunk['review_packet']['author_packet_sha256']=chunk['author_packet']['sha256']
        self.reseal_evaluation_review(audit)
        self.assertTrue(validate_confirmation_audit(audit,source_facts=facts)['independent_agent_review_complete'])
        chunk['authored']['records'][0]['original_source_assertion']+='.'
        self.reseal_evaluation_review(audit)
        with self.assertRaisesRegex(ValueError,'source-assertion authority'):
            validate_confirmation_audit(audit,source_facts=facts)

    def test_generic_and_revision_authors_bind_actual_distinct_task_channels(self):
        facts,audit=confirmation_fixture()
        source=audit['source_chunks'][0];evaluation=audit['evaluation_chunks'][0]
        person={'agent_id':'/root','provider':'openai','model':'gpt-6.1-sol'}
        source['authored']['revision_author']=person
        seal(source['authored']);source['review_packet']['authored_chunk_sha256']=source['authored']['sha256'];seal(source['review_packet'])
        source['reviews'][0]['review_packet_sha256']=source['review_packet']['sha256'];seal(source['reviews'][0])
        evaluation['authored']['author']=person;self.reseal_evaluation_review(audit)
        records=[]
        for chunk,role,prefix in ((source,'revision_author','source-repair'),(evaluation,'author','evaluation-author')):
            task={'task_id':'explicit-synthetic-task-'+prefix,'status':'completed','client_request_id':'test-'+prefix+'-fixture',
                  'child_thread_id':'test-thread-'+prefix,'child_run_id':'test-run-'+prefix,'model':'gpt-6.1-sol','test_fixture':True}
            records.append(task)
            author_key='source_author_packet_sha256' if chunk is source else 'evaluation_author_packet_sha256'
            chunk['author_task_provenance']=[{'role':role,'identity':person,'authored_sha256':chunk['authored']['sha256'],
                'author_packet_sha256':chunk['authored'][author_key],'input_channel':'source_only' if chunk is source else 'evaluation_only',
                'task_id':task['task_id'],'task_record':task}]
        audit['author_task_inventory']={'schema':'p4-actual-t3-author-task-inventory-v1','records':records,'test_fixture':True};seal(audit)
        context=validate_confirmation_audit(audit,source_facts=facts)
        self.assertEqual(len(context['author_task_provenance']),2)
        self.assertTrue(any(row['revision_author']==person for row in context['author_identities']))
        evaluation['author_task_provenance'][0]['task_record']['status']='running';seal(audit)
        with self.assertRaisesRegex(ValueError,'task provenance differs'):
            validate_confirmation_audit(audit,source_facts=facts)
        evaluation['author_task_provenance']=[];seal(audit)
        with self.assertRaisesRegex(ValueError,'requires actual independent task provenance'):
            validate_confirmation_audit(audit,source_facts=facts)

    def test_receipt_packet_and_record_aliases_preserve_actual_bytes_and_reject_contradictions(self):
        facts,audit=confirmation_fixture();chunk=audit['evaluation_chunks'][0];receipt=chunk['reviews'][0]
        receipt['packet_sha256']=receipt.pop('review_packet_sha256')
        receipt['record_reviews']=receipt.pop('records')
        receipt['evaluation_catalog_sha256']=chunk['authored']['sha256'];seal(receipt);seal(audit)
        self.assertTrue(validate_confirmation_audit(audit,source_facts=facts)['independent_agent_review_complete'])
        receipt['review_packet_sha256']='contradictory';seal(receipt);seal(audit)
        with self.assertRaisesRegex(ValueError,'aliases.*contradictory'):
            validate_confirmation_audit(audit,source_facts=facts)
        receipt.pop('review_packet_sha256');receipt['evaluation_catalog_sha256']='different-catalog';seal(receipt);seal(audit)
        with self.assertRaisesRegex(ValueError,'provenance differs'):
            validate_confirmation_audit(audit,source_facts=facts)
        receipt['evaluation_catalog_sha256']=chunk['authored']['sha256'];receipt['records']=[];seal(receipt);seal(audit)
        with self.assertRaisesRegex(ValueError,'record aliases contradict'):
            validate_confirmation_audit(audit,source_facts=facts)

    def test_original_source_receipt_keeps_actual_identity_and_absent_declarations(self):
        facts,audit=confirmation_fixture()
        chunk=audit['source_chunks'][0]
        chunk['authored']['author']={'agent_id':'fixture-opus-author','provider':'ClaudeAlpha','model':'Opus5.5'}
        seal(chunk['authored'])
        chunk['review_packet']['authored_chunk_sha256']=chunk['authored']['sha256'];seal(chunk['review_packet'])
        receipt=chunk['reviews'][0]
        receipt.pop('reviewer');receipt.pop('author_rationales_seen');receipt.pop('model_outcomes_seen')
        receipt['reviewer_identity']={'provider':'openai','model':'gpt-6.1-sol'}
        receipt['human_review_complete']=False;receipt['review_packet_sha256']=chunk['review_packet']['sha256']
        seal(receipt);seal(audit)
        context=validate_confirmation_audit(audit,source_facts=facts)
        record=next(row for row in context['review_identities'] if row['receipt_sha256']==receipt['sha256'])
        self.assertEqual(record['reviewer'],receipt['reviewer_identity'])
        self.assertEqual(record['receipt_visibility_declarations'],{})
        self.assertNotIn('author_rationales_seen',receipt)
        receipt['reviewer_identity']['model']='Opus5.5';seal(receipt);seal(audit)
        with self.assertRaisesRegex(ValueError,'independent author channel'):
            validate_confirmation_audit(audit,source_facts=facts)

    def test_historical_approvals_cover_only_exact_unchanged_blind_records(self):
        facts, audit = confirmation_fixture()
        chunk = audit['evaluation_chunks'][0]
        previous = copy.deepcopy({key: chunk[key] for key in ('authored', 'review_packet', 'reviews')})
        chunk['history'] = [previous]
        chunk['authored']['records'][0]['paraphrase'] = 'Give the recorded value for event zero?'
        self.reseal_evaluation_review(audit)
        chunk['reviews'][0]['records'] = chunk['reviews'][0]['records'][:1]
        seal(chunk['reviews'][0]); seal(audit)
        context = validate_confirmation_audit(audit, source_facts=facts)
        self.assertEqual(context['evaluation_review_sha256'], [previous['reviews'][0]['sha256'], chunk['reviews'][0]['sha256']])
        chunk['reviews'] = []
        seal(audit)
        with self.assertRaisesRegex(ValueError, 'Unapproved current content'):
            validate_confirmation_audit(audit, source_facts=facts)

    def test_matching_current_rejection_needs_actual_explicit_supersession(self):
        facts, audit = confirmation_fixture()
        chunk = audit['evaluation_chunks'][0]
        previous = copy.deepcopy({key: chunk[key] for key in ('authored', 'review_packet', 'reviews')})
        previous['reviews'][0]['records'][0]['status'] = 'needs_revision'
        seal(previous['reviews'][0])
        chunk['history'] = [previous]
        # A new actual receipt from a distinct independent reviewer.
        chunk['reviews'][0]['reviewer'] = person('fixture-rereviewer')
        seal(chunk['reviews'][0]); seal(audit)
        with self.assertRaisesRegex(ValueError, 'unresolved semantic review rejection'):
            validate_confirmation_audit(audit, source_facts=facts)
        row = chunk['reviews'][0]['records'][0]
        chunk['reviews'][0]['supersedes'] = [{'review_receipt_sha256': previous['reviews'][0]['sha256'],
            'field': 'records', 'id': row['id'], 'reviewed_content_sha256': row['reviewed_content_sha256']}]
        seal(chunk['reviews'][0]); seal(audit)
        self.assertTrue(validate_confirmation_audit(audit, source_facts=facts)['independent_agent_review_complete'])
        chunk['reviews'][0]['supersedes'][0]['review_receipt_sha256'] = 'invented-prior-review'
        seal(chunk['reviews'][0]); seal(audit)
        with self.assertRaisesRegex(ValueError, 'supersession edge'):
            validate_confirmation_audit(audit, source_facts=facts)

    def test_changed_content_can_resolve_old_rejection_only_with_new_actual_approval(self):
        facts, audit = confirmation_fixture()
        chunk = audit['evaluation_chunks'][0]
        previous = copy.deepcopy({key: chunk[key] for key in ('authored', 'review_packet', 'reviews')})
        previous['reviews'][0]['records'][0]['status'] = 'needs_revision'
        seal(previous['reviews'][0]); chunk['history'] = [previous]
        chunk['authored']['records'][0]['paraphrase'] = 'Give the recorded value for event zero?'
        self.reseal_evaluation_review(audit)
        chunk['reviews'][0]['records'] = chunk['reviews'][0]['records'][:1]
        seal(chunk['reviews'][0]); seal(audit)
        self.assertTrue(validate_confirmation_audit(audit, source_facts=facts)['independent_agent_review_complete'])
        previous['review_packet']['records'][0]['paraphrase'] = 'Tampered reviewed wording'
        seal(previous['review_packet']); seal(audit)
        with self.assertRaisesRegex(ValueError, 'Historical evaluation receipt'):
            validate_confirmation_audit(audit, source_facts=facts)

    def reseal_evaluation_review(self, audit):
        chunk = audit['evaluation_chunks'][0]
        seal(chunk['authored'])
        packet = chunk['review_packet']
        packet['evaluation_catalog_sha256'] = chunk['authored']['sha256']
        packet['records'] = [evaluation_review_content(row) for row in chunk['authored']['records']]
        seal(packet)
        receipt = chunk['reviews'][0]
        receipt['review_packet_sha256'] = packet['sha256']
        receipt['records'] = [{'id': row['id'], 'status': 'approved', 'reviewed_content_sha256': digest(row),
                               'notes': 'Synthetic test review decision'} for row in packet['records']]
        seal(receipt); seal(audit)

    def test_empty_evidence_preserves_ambiguity_only_after_actual_blind_receipt(self):
        for status in ('supported', 'retained_source_ambiguity', 'retained_source_conflict'):
            facts, audit = confirmation_fixture()
            row = audit['evaluation_chunks'][0]['authored']['records'][0]
            row['source_ambiguity_status'], row['evidence'] = status, []
            self.reseal_evaluation_review(audit)
            with self.subTest(status=status):
                if status == 'supported':
                    with self.assertRaisesRegex(ValueError, 'evidence'):
                        validate_confirmation_audit(audit, source_facts=facts)
                else:
                    self.assertEqual(validate_confirmation_audit(audit, source_facts=facts)['evaluation_records'][0]['evidence'], [])
                    audit['evaluation_chunks'][0]['reviews'] = []
                    seal(audit)
                    with self.assertRaisesRegex(ValueError, 'Missing actual independent'):
                        validate_confirmation_audit(audit, source_facts=facts)

    def test_blind_review_rejects_author_judgments_even_with_fresh_approval(self):
        facts, audit = confirmation_fixture()
        chunk = audit['evaluation_chunks'][0]
        packet = chunk['review_packet']
        packet['records'][0]['source_ambiguity_status'] = 'supported'
        seal(packet)
        receipt = chunk['reviews'][0]
        receipt['review_packet_sha256'] = packet['sha256']
        receipt['records'][0]['reviewed_content_sha256'] = digest(packet['records'][0])
        seal(receipt); seal(audit)
        with self.assertRaisesRegex(ValueError, 'included author judgments'):
            validate_confirmation_audit(audit, source_facts=facts)

    def test_metadata_only_derivation_preserves_original_semantic_payload(self):
        facts, audit = confirmation_fixture()
        chunk = audit['source_chunks'][0]
        original = copy.deepcopy(chunk['authored'])
        original['source_units'][0]['opaque_probe_ids'].append('duplicate-cluster-metadata')
        seal(original)
        chunk['original_authored'] = original
        chunk['authored']['metadata_correction'] = {'kind': 'canonical_opaque_probe_membership_only',
                                                   'original_authored_sha256': original['sha256']}
        seal(chunk['authored'])
        chunk['review_packet']['authored_chunk_sha256'] = chunk['authored']['sha256']
        seal(chunk['review_packet'])
        chunk['reviews'][0]['review_packet_sha256'] = chunk['review_packet']['sha256']
        seal(chunk['reviews'][0]); seal(audit)
        self.assertTrue(validate_confirmation_audit(audit, source_facts=facts)['independent_agent_review_complete'])
        chunk['authored']['acquisition_records'][0]['answer'] = 'Changed semantic target'
        seal(chunk['authored']); seal(audit)
        with self.assertRaisesRegex(ValueError, 'changed semantic author content'):
            validate_confirmation_audit(audit, source_facts=facts)

    def test_complete_synthetic_audit_retains_human_false_and_identity_limits(self):
        facts, audit = confirmation_fixture()
        context = validate_confirmation_audit(audit, source_facts=facts)
        self.assertEqual(len(context['sources']), 80)
        self.assertEqual(len(context['evaluation_records']), 80)
        self.assertFalse(context['human_review_complete'])
        self.assertEqual(context['same_role_components'], [['event_000', 'event_071']])
        self.assertEqual(context['identity_limitations'], ['fixture-cross-link'])
        self.assertFalse(audit['source_chunks'][0]['authored']['independent_agent_review_complete'])

    def test_booleans_cannot_replace_missing_review_receipts(self):
        facts, audit = confirmation_fixture()
        audit['source_chunks'][0]['reviews'] = []
        audit['independent_agent_review_complete'] = True
        seal(audit)
        with self.assertRaisesRegex(ValueError, 'Missing actual independent'):
            validate_confirmation_audit(audit, source_facts=facts)

    def test_needs_revision_stale_receipt_and_same_author_are_rejected(self):
        for mutation in ('rejected', 'stale', 'author'):
            facts, audit = confirmation_fixture()
            chunk = audit['source_chunks'][0]
            receipt = chunk['reviews'][0]
            if mutation == 'rejected':
                receipt['groups'][0]['status'] = 'needs_revision'
            elif mutation == 'stale':
                receipt['acquisition_records'][0]['reviewed_content_sha256'] = 'changed-content'
            else:
                receipt['reviewer'] = chunk['authored']['author']
            seal(receipt); seal(audit)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_confirmation_audit(audit, source_facts=facts)

    def test_actual_evaluation_receipts_and_original_canonical_wording_are_required(self):
        for mutation in ('missing', 'alias', 'question', 'self-review', 'source-author'):
            facts, audit = confirmation_fixture()
            chunk = audit['evaluation_chunks'][0]
            if mutation == 'missing':
                chunk['reviews'] = []
            elif mutation == 'alias':
                chunk['authored']['records'][0]['aliases'] = ['invented alias']
                seal(chunk['authored'])
                chunk['review_packet']['evaluation_catalog_sha256'] = chunk['authored']['sha256']
                seal(chunk['review_packet'])
            elif mutation == 'question':
                facts[0]['question'] = 'Different original canonical question'
            elif mutation == 'self-review':
                chunk['reviews'][0]['reviewer'] = chunk['authored']['author']
                seal(chunk['reviews'][0])
            else:
                chunk['authored']['author'] = audit['source_chunks'][0]['authored']['author']
                seal(chunk['authored'])
                chunk['review_packet']['evaluation_catalog_sha256'] = chunk['authored']['sha256']
                seal(chunk['review_packet'])
            seal(audit)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_confirmation_audit(audit, source_facts=facts)

    def test_unresolved_proposition_without_adjudication_and_partition_crossing_block(self):
        for mutation in ('missing', 'rolecomponent', 'changedgold', 'self-review'):
            facts, audit = confirmation_fixture()
            if mutation == 'missing':
                audit['cross'].pop('adjudication')
            elif mutation == 'rolecomponent':
                audit['cross']['review']['required_same_role_components'][0]['events'].append('event_080')
                seal(audit['cross']['review'])
            elif mutation == 'changedgold':
                audit['cross']['adjudication']['decisions'][0]['change_gold_or_evaluation'] = True
                seal(audit['cross']['adjudication'])
            else:
                audit['cross']['adjudication']['reviewer'] = audit['cross']['review']['reviewer']
                seal(audit['cross']['adjudication'])
            seal(audit)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_confirmation_audit(audit, source_facts=facts)

    def test_catalog_full_coverage_and_source_byte_evidence_are_checked(self):
        facts, audit = confirmation_fixture()
        catalog = audit['source_catalog']
        validate_catalog(catalog, audit['probe_map'], facts)
        corrupted = copy.deepcopy(facts)
        corrupted[0]['statement'] = 'Unsupported original assertion'
        with self.assertRaisesRegex(ValueError, 'pinned source'):
            validate_catalog(catalog, audit['probe_map'], corrupted)
        with self.assertRaisesRegex(ValueError, 'pinned canonical probe'):
            validate_catalog(catalog, audit['probe_map'], facts[:-1])

    def test_metadata_roles_are_deterministic_protected_and_keep_geometry(self):
        facts, audit = confirmation_fixture()
        partition = audit['source_catalog']['partition']
        components = [['event_000', 'event_071']]
        roles = assign_confirmation_roles(facts, partition, 2026100801, components)
        self.assertEqual(roles['roles']['event_000'], roles['roles']['event_071'])
        self.assertEqual(roles, assign_confirmation_roles(facts, partition, 2026100801, components))
        self.assertEqual(roles['counts'], {'old': 20, 'new': 30, 'control': 20, 'qa': 10})
        with self.assertRaisesRegex(ValueError, 'partition amendment'):
            assign_confirmation_roles(facts, partition, 2026100801, [['event_000', 'event_080']])
