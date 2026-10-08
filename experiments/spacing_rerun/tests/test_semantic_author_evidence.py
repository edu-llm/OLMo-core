"""Synthetic provenance mutation checks; these fixtures confer no real-data approval."""
import copy
import hashlib
import json
import unittest

from spacing_rerun.confirmation import (local_author_identity, semantic_execution_evidence,
    validate_retained_source_ancestor, source_preservation_context, validate_source_chunks, author_context)
from spacing_rerun.source_patch import normalize_task
from test_confirmation import confirmation_fixture, seal

SESSION = '01a11a8a-4936-7b60-b8c1-010a5a107ebe'


def raw(value, path):
    text = json.dumps(value, ensure_ascii=False)
    return {'path': path, 'utf8': text, 'file_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'canonical_sha256': value['sha256']}


def execution_fixture():
    _, audit = confirmation_fixture()
    chunk = copy.deepcopy(audit['source_chunks'][0])
    ancestor = copy.deepcopy(chunk['authored'])
    person = {'agent_id': '/root', 'provider': 'openai', 'model': 'fixture-model', 'task_id': '/root'}
    authored = chunk['authored']
    authored.update({'prior_authored_sha256': ancestor['sha256'], 'revision_author': person,
                     'retained_source_ancestor': {'applies_to_authored_sha256': ancestor['sha256']}})
    seal(authored)
    packet = seal({'schema': 'source-only-confirmation-review-and-structural-repair-input-v1',
        'source_author_packet': chunk['author_packet'], 'current_authored': ancestor,
        'current_authored_sha256': ancestor['sha256'], 'current_authored_path': '/synthetic/ancestor.json',
        'chunk_index': authored['chunk_index'], 'evaluation_question_text_included': False,
        'evaluation_identifiers_included': False, 'model_outcomes_included': False,
        'actual_source_review_materials': [{'receipt_path': '/synthetic/review.json',
            'review_packet_path': '/synthetic/review-packet.json', 'receipt': chunk['reviews'][0],
            'review_packet': chunk['review_packet']}]})
    response = seal({'schema': 'synthetic-correction-response-v1', 'revision_author': person,
        'authored_sha256': authored['sha256'], 'prior_authored_sha256': ancestor['sha256'],
        'input_packet_sha256': packet['sha256'], 'authored_path': '/synthetic/authored.json'})
    summary = 'Synthetic completed artifacts ' + authored['sha256'] + ' ' + response['sha256']
    status = {'status': 'completed', 'childRunId': 'synthetic-run', 'childThreadId': 'synthetic-thread',
        'taskId': 'node:delegated-task:command:synthetic:delegate-task:source-semantic-repair-synthetic',
        'model': 'fixture-model', 'providerInstanceId': 'codex-synthetic-fixture',
        'summary': summary, 'latestTerminalSummary': summary, 'latestTerminalStatus': 'completed',
        'hasPendingChildRuns': False}
    row = {'role': 'revision_author', 'task_record': normalize_task(status, 'source-semantic-repair-synthetic'),
        'identity': person, 'authored_sha256': authored['sha256'],
        'author_packet_sha256': authored['source_author_packet_sha256'], 'input_channel': 'source_only',
        'task_id': status['taskId'],
        'declared_task_identity_classification': 'runtime_agent_path',
        'semantic_execution_evidence': {'input_packet': packet, 'correction_response': response,
             'artifact_files': {'authored': raw(authored, '/synthetic/authored.json'),
                  'input_packet': raw(packet, '/synthetic/input.json'),
                  'correction_response': raw(response, '/synthetic/response.json')}}}
    chunk['author_task_provenance'] = [row]
    return chunk, row, person, ancestor


class SemanticAuthorEvidenceTests(unittest.TestCase):
    def test_question_patch_retains_full_semantic_base_execution_and_source_ancestry(self):
        from test_source_patch import proof_fixture, refresh
        from spacing_rerun.source_patch import reconstruct_question_patch
        base_version, _, _, ancestor = execution_fixture()
        base_version['source_ancestor_version'] = {'authored': ancestor}
        evidence = proof_fixture(); base = copy.deepcopy(base_version['authored'])
        evidence['base_authored'] = base
        for key in ('input_packet', 'patch'):
            evidence[key]['current_authored_sha256'] = base['sha256']
        evidence['patch']['prior_authored_sha256'] = base['sha256']
        evidence['response']['prior_authored_sha256'] = base['sha256']; refresh(evidence)
        merged = reconstruct_question_patch(evidence); task = evidence['task_record']
        row = {'role': 'revision_author', 'task_record': task, 'task_id': task['task_id'],
            'identity': merged['revision_author'], 'authored_sha256': merged['sha256'],
            'author_packet_sha256': merged['source_author_packet_sha256'], 'input_channel': 'source_only'}
        chunk = {'authored': merged, 'question_patch_evidence': evidence, 'author_task_provenance': [row],
                 'question_patch_base_version': base_version}
        context = author_context(chunk)
        self.assertIn('task:' + base_version['author_task_provenance'][0]['task_id'], context['channels'])
        self.assertIn('task:' + task['task_id'], context['channels'])
        for kind in ('missing-base', 'wrong-base', 'modified-ancestor'):
            changed = copy.deepcopy(chunk)
            if kind == 'missing-base': changed.pop('question_patch_base_version')
            elif kind == 'wrong-base': changed['question_patch_base_version']['authored']['chunk_index'] = 2
            else:
                old = changed['question_patch_base_version']['source_ancestor_version']['authored']
                old['groups'][0]['statement'] = 'Invented historic source'; seal(old)
            with self.subTest(kind=kind), self.assertRaises(ValueError): author_context(changed)

    def test_actual_source_result_binding_accepts_generic_path_but_rejects_mutations(self):
        chunk, row, person, _ = execution_fixture()
        semantic_execution_evidence(chunk, row, person)
        mutations = [
            lambda r: r['task_record']['actual_task_status'].update(latestTerminalStatus='running'),
            lambda r: r['task_record']['actual_task_status'].update(hasPendingChildRuns=True),
            lambda r: r['task_record'].update(provider_instance_id='foreign-provider'),
            lambda r: r.update(declared_task_identity_classification='invented'),
            lambda r: r['semantic_execution_evidence']['artifact_files']['authored'].update(file_sha256='0'*64),
            lambda r: r['semantic_execution_evidence']['artifact_files']['input_packet'].update(utf8='{}'),
            lambda r: r['semantic_execution_evidence']['correction_response'].update(input_packet_sha256='0'*64),
            lambda r: r['task_record']['actual_task_status'].update(latestTerminalSummary='unbound result'),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(row); mutate(changed)
                with self.assertRaises(ValueError): semantic_execution_evidence(chunk, changed, person)

    def test_disclosed_runtime_sessions_are_distinct_from_delegated_task_ids(self):
        authored = {'chunk_index': 6}
        person = {'agent_id': 'codex:' + SESSION, 'provider': 'openai', 'model': 'fixture-model',
                  'codex_thread_id': SESSION, 'runtime_agent_path': '/root', 'task_id': SESSION}
        self.assertEqual(local_author_identity(person, authored), 'runtime_session_label')
        for key, bad in [('task_id', 'node:delegated-task:foreign'), ('agent_id', 'codex:foreign'),
                         ('runtime_agent_path', '/root/foreign'), ('runtime_thread_id', 'another-session')]:
            with self.subTest(key=key):
                changed = {**person, key: bad}
                with self.assertRaises(ValueError): local_author_identity(changed, authored)
        local = {'agent_id': '/root', 'provider': 'openai', 'model': 'fixture-model',
            'task_id': '/root:chunk06-source-only-semantic-v4:' + SESSION,
            'identity_basis': 'local repair execution UUID, not an asserted T3 delegated-task ID'}
        self.assertEqual(local_author_identity(local, authored), 'disclosed_local_repair_uuid')
        with self.assertRaises(ValueError): local_author_identity(local, {'chunk_index': 7})

    def test_ancestor_copies_snapshots_and_annotation_ledger_are_exact(self):
        chunk, _, _, ancestor = execution_fixture()
        retained = chunk['authored']['retained_source_ancestor']
        retained.update({'author': ancestor['author'], 'authored_snapshot': copy.deepcopy(ancestor),
                         'authored_path': '/synthetic/ancestor.json'})
        group = chunk['authored']['groups'][0]; old = group['grouping_rationale']; group['grouping_rationale'] = 'Updated synthetic explanation'
        retained['replaced_author_annotations'] = [{'collection': 'groups', 'field': 'grouping_rationale',
            'id': group['id'], 'before': old, 'after': group['grouping_rationale'], 'source_only_reason': 'Synthetic reason'}]
        validate_retained_source_ancestor(chunk, ancestor)
        for mutate in [lambda r: r['authored_snapshot']['groups'][0].update(statement='Changed old proposition'),
                       lambda r: r['replaced_author_annotations'][0].update(before='Invented old rationale'),
                       lambda r: r.update(unknown_annotation='unsupported'),
                       lambda r: r.update(authored_path='/synthetic/foreign.json')]:
            changed = copy.deepcopy(chunk); mutate(changed['authored']['retained_source_ancestor'])
            with self.assertRaises(ValueError): validate_retained_source_ancestor(changed, ancestor)

    def test_preservation_context_cannot_add_author_rationale_or_weaken_rule(self):
        _, audit = confirmation_fixture(); chunk = audit['source_chunks'][0]
        packet = chunk['review_packet']; packet['source_preservation_protocol'] = source_preservation_context(chunk['authored']); seal(packet)
        receipt = chunk['reviews'][0]; receipt['review_packet_sha256'] = packet['sha256']; seal(receipt)
        validate_source_chunks([chunk], audit['source_catalog'])
        for key, bad in [('author_rationale', 'invented authority'), ('preserved_declaration_rule', 'Merges allowed'),
                         ('source_dispositions', [{'unit_id': 'foreign', 'status': 'preserve_source_conflict'}])]:
            changed = copy.deepcopy(chunk); changed['review_packet']['source_preservation_protocol'][key] = bad; seal(changed['review_packet'])
            with self.assertRaisesRegex(ValueError, 'preservation protocol'): validate_source_chunks([changed], audit['source_catalog'])
