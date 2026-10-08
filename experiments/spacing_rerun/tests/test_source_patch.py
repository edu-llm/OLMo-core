"""Explicit synthetic source-patch proofs; never production author/review evidence."""
import copy
import hashlib
import json
import unittest

from spacing_rerun.source_patch import PATCH_EVIDENCE, PATCH_INPUT, normalize_task, reconstruct_question_patch
from spacing_rerun.confirmation import source_review_content, validate_confirmation_audit
from spacing_rerun.common import digest
from test_confirmation import confirmation_fixture, seal


def proof_fixture():
    _, audit = confirmation_fixture()
    base = copy.deepcopy(audit['source_chunks'][0]['authored'])
    selected = copy.deepcopy(base['acquisition_records'][:1]);selected[0]['question_indices_to_revise'] = [0]
    packet = seal({'schema': PATCH_INPUT, 'current_authored_sha256': base['sha256'],
        **{key: base[key] for key in ('source_catalog_sha256','source_author_packet_sha256','dataset_revision','chunk_index','events','groups','source_units')},
        'evaluation_question_text_included':False,'model_outcomes_included':False,'other_reviewer_judgments_included':False,
        'selected_acquisition_records':selected,'test_fixture':True})
    patch = copy.deepcopy(packet)
    author = {'agent_id':'synthetic-source-question-author','provider':'openai','model':'gpt-6.1-sol',
              'task_id':'synthetic-source-language-revision-fixture'}
    patch.update(prior_authored_sha256=base['sha256'],revision_author=author)
    patch['selected_acquisition_records'][0]['questions'][0] = 'Give the recorded source entry for event zero?'
    seal(patch)
    response = seal({'schema':'synthetic-source-correction-response','input_packet_sha256':packet['sha256'],
                    'prior_authored_sha256':base['sha256'],'output_authored_sha256':patch['sha256'],
                    'revision_author':author,'test_fixture':True})
    status = {'taskId':'synthetic:delegate-task:'+author['task_id'],'status':'completed','childRunId':'synthetic-run',
              'childThreadId':'synthetic-thread','model':author['model'],'summary':patch['sha256'],'test_fixture':True}
    inventory = {'schema':'synthetic-task-status-inventory','records':[status],'test_fixture':True}
    evidence = dict(schema=PATCH_EVIDENCE,base_authored=base,input_packet=packet,patch=patch,response=response,
                    actual_task_status=status,actual_task_status_inventory=inventory,task_record=normalize_task(status,author['task_id']),
                    test_fixture=True)
    refresh(evidence)
    return evidence


def refresh(evidence):
    def raw(value, key):
        text = json.dumps(value,sort_keys=True,indent=2)
        return {'utf8':text,'file_sha256':hashlib.sha256(text.encode()).hexdigest(),'path':'/tmp/synthetic-'+key+'.json',
                'canonical_sha256':value.get('sha256')}
    packet,patch,response = (evidence[key] for key in ('input_packet','patch','response'))
    seal(packet);seal(patch)
    response['input_packet_sha256'] = packet['sha256'];response['output_authored_sha256'] = patch['sha256']
    response['input_file_bytes_sha256'] = raw(packet,'input_packet')['file_sha256'];seal(response)
    evidence['actual_task_status']['summary'] = patch['sha256']
    evidence['task_record'] = normalize_task(evidence['actual_task_status'],patch['revision_author']['task_id'])
    evidence['artifact_files'] = {key:raw(evidence[key],key) for key in
        ('base_authored','input_packet','patch','response','actual_task_status_inventory')}
    seal(evidence)


class SourcePatchTests(unittest.TestCase):
    @staticmethod
    def reseal_response(evidence):
        response = seal(evidence['response'])
        text = json.dumps(response, sort_keys=True, indent=2)
        evidence['artifact_files']['response'].update(
            utf8=text, file_sha256=hashlib.sha256(text.encode()).hexdigest(), canonical_sha256=response['sha256'])
        seal(evidence)

    def test_actual_patch_named_response_aliases_preserve_exact_raw_response(self):
        evidence = proof_fixture();response = evidence['response']
        response['patch_sha256'] = response.pop('output_authored_sha256')
        response['base_sha256'] = response.pop('prior_authored_sha256')
        response['patch_file_sha256'] = evidence['artifact_files']['patch']['file_sha256']
        response['patch_path'] = evidence['artifact_files']['patch']['path']
        self.reseal_response(evidence)
        before = copy.deepcopy(response)
        merged = reconstruct_question_patch(evidence)
        self.assertEqual(response, before)
        self.assertEqual(merged['question_patch_provenance']['correction_response_sha256'], response['sha256'])
        self.assertEqual(merged['groups'], evidence['base_authored']['groups'])

    def test_all_present_response_aliases_must_bind_same_actual_base_and_patch(self):
        for key in ('patch_sha256', 'patch_file_sha256', 'base_sha256', 'patch_path', 'output_path'):
            with self.subTest(key=key):
                evidence = proof_fixture();response = evidence['response']
                response.update(patch_sha256=evidence['patch']['sha256'],
                                patch_file_sha256=evidence['artifact_files']['patch']['file_sha256'],
                                base_sha256=evidence['base_authored']['sha256'],
                                patch_path=evidence['artifact_files']['patch']['path'])
                response[key] = 'different actual artifact'
                self.reseal_response(evidence)
                with self.assertRaisesRegex(ValueError, 'declared provenance differs'):
                    reconstruct_question_patch(evidence)

    def test_response_cannot_omit_both_actual_base_digest_aliases(self):
        evidence = proof_fixture();evidence['response'].pop('prior_authored_sha256')
        self.reseal_response(evidence)
        with self.assertRaisesRegex(ValueError, 'prior_authored_sha256/base_sha256'):
            reconstruct_question_patch(evidence)

    def test_nested_actual_response_bindings_preserve_author_bytes(self):
        evidence = proof_fixture();response = evidence['response']
        response.pop('output_authored_sha256');response.pop('prior_authored_sha256')
        for key, target in (('authored_patch', 'patch'), ('exact_source_base', 'base_authored')):
            info = evidence['artifact_files'][target]
            response[key] = {'sha256': evidence[target]['sha256'], 'file_sha256': info['file_sha256'], 'path': info['path']}
        self.reseal_response(evidence);before = copy.deepcopy(response)
        merged = reconstruct_question_patch(evidence)
        self.assertEqual(response, before)
        self.assertEqual(merged['question_patch_provenance']['base_authored_sha256'], evidence['base_authored']['sha256'])

    def test_nested_response_bindings_require_actual_hash_bytes_path_and_alias_consistency(self):
        for binding, target in (('authored_patch', 'patch'), ('exact_source_base', 'base_authored')):
            for field in ('sha256', 'file_sha256', 'path', 'missing-file-sha', 'flat-conflict'):
                with self.subTest(binding=binding, field=field):
                    evidence = proof_fixture();response = evidence['response'];info = evidence['artifact_files'][target]
                    response[binding] = {'sha256': evidence[target]['sha256'], 'file_sha256': info['file_sha256'], 'path': info['path']}
                    if field == 'missing-file-sha':response[binding].pop('file_sha256')
                    elif field == 'flat-conflict':
                        response['patch_sha256' if target == 'patch' else 'base_sha256'] = 'different declared artifact'
                    else:response[binding][field] = 'different actual artifact'
                    self.reseal_response(evidence)
                    with self.assertRaises(ValueError):reconstruct_question_patch(evidence)

    def test_semantic_revision_retains_actual_patch_ancestor_and_author_channels(self):
        evidence=proof_fixture();facts,audit=confirmation_fixture();chunk=audit['source_chunks'][0]
        merged=reconstruct_question_patch(evidence);request='synthetic-source-repair-after-question-patch-fixture'
        who={'agent_id':request,'provider':'openai','model':'gpt-6.1-sol','task_id':request}
        task={'task_id':'synthetic-post-patch-repair-task','client_request_id':request,'child_thread_id':'synthetic-thread',
              'child_run_id':'synthetic-run','model':who['model'],'status':'completed','test_fixture':True}
        def provenance(authored,actual,role='revision_author'):
            return {'role':role,'identity':authored[role],'authored_sha256':authored['sha256'],
                    'author_packet_sha256':authored['source_author_packet_sha256'],'input_channel':'source_only',
                    'task_id':actual['task_id'],'task_record':actual}
        ancestor={'authored':merged,'question_patch_evidence':evidence,'base_author_task_provenance':[],
                  'author_task_provenance':[provenance(merged,evidence['task_record'])]}
        current=copy.deepcopy(merged);current.pop('question_patch_provenance')
        current.update(prior_authored_sha256=merged['sha256'],revision_author=who,
            retained_source_ancestor={'applies_to_authored_sha256':merged['sha256'],'question_patch_provenance':copy.deepcopy(merged['question_patch_provenance'])})
        current['acquisition_records'][0]['questions'][1]='Name the recorded value in the source event zero account?';seal(current)
        chunk.update(authored=current,source_ancestor_version=ancestor,author_task_provenance=[provenance(current,task)])
        packet=chunk['review_packet'];packet['authored_chunk_sha256']=current['sha256']
        packet['acquisition_records']=[source_review_content(row,'acquisition_records') for row in current['acquisition_records']];seal(packet)
        receipt=chunk['reviews'][0];receipt['review_packet_sha256']=packet['sha256']
        receipt['acquisition_records']=[{'id':row['id'],'status':'approved','notes':'Explicit synthetic test decision',
            'reviewed_content_sha256':digest(row)} for row in packet['acquisition_records']];seal(receipt)
        audit['author_task_inventory']={'schema':'p4-actual-t3-author-task-inventory-v1','records':[evidence['task_record'],task],'test_fixture':True};seal(audit)
        context=validate_confirmation_audit(audit,source_facts=facts)
        self.assertEqual(len(context['author_task_provenance']),2)
        current['retained_source_ancestor']['question_patch_provenance']['patch_sha256']='invented prior patch';seal(current)
        chunk['author_task_provenance'][0]['authored_sha256']=current['sha256'];seal(audit)
        with self.assertRaisesRegex(ValueError,'Retained source history'):
            validate_confirmation_audit(audit,source_facts=facts)

    def test_exact_allowlist_merge_keeps_original_semantics_and_author(self):
        evidence = proof_fixture();merged = reconstruct_question_patch(evidence);base = evidence['base_authored']
        self.assertEqual(merged['author'],base['author'])
        self.assertEqual(merged['source_units'],base['source_units'])
        self.assertEqual(merged['groups'],base['groups'])
        self.assertEqual(merged['acquisition_records'][1:],base['acquisition_records'][1:])
        self.assertEqual(merged['acquisition_records'][0]['questions'][1],base['acquisition_records'][0]['questions'][1])
        self.assertEqual(len(merged['question_patch_provenance']['changes']),1)
        self.assertTrue(merged['question_patch_provenance']['no_semantic_review_decisions_created'])

    def test_metadata_claims_are_scoped_to_the_exact_immutable_ancestor(self):
        evidence = proof_fixture();base = evidence['base_authored']
        base['metadata_derivation'] = {'revision_kind':'explicit-synthetic-ancestor-annotation'};seal(base)
        for key in ('input_packet','patch'):evidence[key]['current_authored_sha256'] = base['sha256']
        evidence['patch']['prior_authored_sha256'] = base['sha256'];evidence['response']['prior_authored_sha256'] = base['sha256']
        refresh(evidence);merged = reconstruct_question_patch(evidence)
        self.assertNotIn('metadata_derivation',merged)
        self.assertEqual(merged['retained_historical_annotations'],[{'applies_to_authored_sha256':base['sha256'],
                        'annotations':{'metadata_derivation':base['metadata_derivation']}}])

    def test_mutated_actual_author_result_cannot_change_untargeted_or_protected_data(self):
        for kind in ('untargeted','answer','context','membership','base-selection'):
            with self.subTest(kind=kind):
                evidence = proof_fixture();row = evidence['patch']['selected_acquisition_records'][0]
                if kind == 'untargeted':row['questions'][1] = 'Unauthorized other question'
                elif kind == 'answer':row['answer'] = 'changed target'
                elif kind == 'context':evidence['patch']['factsheets'] = {'invented':'context'};evidence['patch']['groups'][0]['statement'] = 'Invented source'
                elif kind == 'membership':evidence['patch']['selected_acquisition_records'].append(copy.deepcopy(row))
                else:evidence['input_packet']['selected_acquisition_records'][0]['answer'] = 'inconsistent selected base';row['answer'] = 'inconsistent selected base'
                refresh(evidence)
                with self.assertRaises(ValueError):reconstruct_question_patch(evidence)

    def test_raw_packet_completion_and_binding_mutations_fail_closed(self):
        for kind in ('bytes','task','binding'):
            with self.subTest(kind=kind):
                evidence = proof_fixture()
                if kind == 'bytes':evidence['artifact_files']['patch']['utf8'] += ' '
                elif kind == 'task':evidence['actual_task_status']['status'] = 'running'
                else:evidence['response']['input_file_bytes_sha256'] = 'different actual packet'
                seal(evidence)
                with self.assertRaises(ValueError):reconstruct_question_patch(evidence)
