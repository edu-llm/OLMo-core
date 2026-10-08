"""Synthetic provenance mutation checks; these fixtures confer no real-data approval."""
import copy
import hashlib
import json
import unittest

from spacing_rerun.confirmation import (local_author_identity, semantic_execution_evidence,
    validate_retained_source_ancestor, source_preservation_context, validate_source_chunks, author_context,
    declared_receipt_bindings, source_semantic_base_version, validate_confirmation_audit)
from spacing_rerun.common import digest
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
    def test_whole_prior_source_snapshots_bind_exact_raw_artifacts_and_completed_review(self):
        chunk, row, _, ancestor = execution_fixture()
        packet = row['semantic_execution_evidence']['input_packet']
        material = packet['actual_source_review_materials'][0]
        status = copy.deepcopy(row['task_record']['actual_task_status'])
        status['latestTerminalSummary'] = 'Synthetic completed review ' + material['receipt']['sha256']
        context = {'authored': ancestor, 'author_task_provenance': []}
        status_raw = {'task_statuses': [{'result': status}]}
        status_text = json.dumps(status_raw, ensure_ascii=False)
        snapshots = {'current_authored_raw_file': raw(ancestor, packet['current_authored_path']),
            'latest_actual_review_receipt_file': raw(material['receipt'], material['receipt_path']),
            'latest_actual_blind_packet_file': raw(material['review_packet'], material['review_packet_path']),
            'actual_latest_review_task_status': status,
            'actual_latest_review_task_status_file': {'path': '/synthetic/status.json', 'utf8': status_text,
                'file_sha256': hashlib.sha256(status_text.encode()).hexdigest(), 'canonical_sha256': None},
            'whole_prior_source_chunk_context': context,
            'whole_prior_context_binding': {'extracted_source_chunk_sha256': digest(context),
                'extracted_source_chunk_index': ancestor['chunk_index'],
                'other_source_or_evaluation_chunks_included': False}}
        packet.update(copy.deepcopy(snapshots))
        chunk['authored']['retained_source_ancestor'].update(copy.deepcopy(snapshots))
        validate_retained_source_ancestor(chunk, ancestor)
        for field in snapshots:
            changed = copy.deepcopy(chunk)
            changed['authored']['retained_source_ancestor'][field] = {'invented': True}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'exact source-only input'):
                validate_retained_source_ancestor(changed, ancestor)
        for field, mutate in (
                ('current_authored_raw_file', lambda value: value.update(file_sha256='0'*64)),
                ('whole_prior_context_binding', lambda value: value.update(extracted_source_chunk_sha256='0'*64)),
                ('actual_latest_review_task_status', lambda value: value.update(status='running'))):
            changed = copy.deepcopy(chunk)
            mutate(changed['author_task_provenance'][0]['semantic_execution_evidence']['input_packet'][field])
            changed['authored']['retained_source_ancestor'][field] = copy.deepcopy(
                changed['author_task_provenance'][0]['semantic_execution_evidence']['input_packet'][field])
            with self.subTest(invalid_source_field=field), self.assertRaises(ValueError):
                validate_retained_source_ancestor(changed, ancestor)

    def test_parent_inventory_covers_actual_author_beneath_computed_metadata_view(self):
        facts, audit = confirmation_fixture()
        chunk = audit['source_chunks'][0]
        base = copy.deepcopy(chunk)
        base['authored']['source_units'][0]['nonliteral_answer_labels'] = ['incorrect-flag']

        def attach_task(version, name, *, computed=False):
            authored = version['authored']
            who = {'agent_id': name, 'task_id': name, 'provider': 'openai', 'model': 'fixture-model'}
            if computed:
                who['role'] = 'derived_flag_metadata_only'
                authored.update(prior_authored_sha256=base['authored']['sha256'], approval_granted=False,
                    metadata_derivation={'revision_kind': 'derived_flag_metadata_only', 'metadata_only': True,
                                         'task_id': name})
            authored['revision_author'] = who
            seal(authored)
            result = {'authored_sha256': authored['sha256'],
                      'author_packet_sha256': authored['source_author_packet_sha256'],
                      'input_channel': 'source_only', 'identity': who}
            task = {'schema': 'spacing-native-author-task-evidence-v1', 'canonical_task_name': name,
                    'status': 'completed', 'model': who['model'], 'actual_result': result,
                    'actual_result_text': 'Synthetic completed result ' + authored['sha256'],
                    'result_source': 'Synthetic fixture, no actual certification'}
            version['author_task_provenance'] = [{'role': 'revision_author', 'task_id': name,
                                                 'task_record': task, **result}]
            return task

        base_task = attach_task(base, '/root/synthetic-semantic-base-author')
        chunk['authored'] = copy.deepcopy(base['authored'])
        chunk['authored']['source_units'][0]['nonliteral_answer_labels'] = []
        current_task = attach_task(chunk, '/root/synthetic-computed-flags-author', computed=True)
        chunk.update(prior_authored=base['authored'], metadata_derivation_base_version=base)
        chunk['review_packet']['authored_chunk_sha256'] = chunk['authored']['sha256']
        seal(chunk['review_packet'])
        chunk['reviews'][0]['review_packet_sha256'] = chunk['review_packet']['sha256']
        seal(chunk['reviews'][0])
        audit['author_task_inventory'] = {'schema': 'p4-actual-t3-author-task-inventory-v1',
                                          'records': [base_task, current_task], 'test_fixture': True}
        seal(audit)
        self.assertTrue(validate_confirmation_audit(audit, source_facts=facts)['independent_agent_review_complete'])
        audit['author_task_inventory']['records'].remove(base_task)
        seal(audit)
        with self.assertRaisesRegex(ValueError, 'actual parent task inventory'):
            validate_confirmation_audit(audit, source_facts=facts)

    def test_supplied_v4_snapshot_aliases_do_not_create_missing_v5_history(self):
        chunk, row, _, ancestor = execution_fixture()
        ancestor['retained_source_ancestor'] = {'question_patch_provenance': {'schema': 'synthetic-historical-patch'}}
        seal(ancestor); chunk['authored']['prior_authored_sha256'] = ancestor['sha256']
        packet = row['semantic_execution_evidence']['input_packet']
        packet.update(current_authored=ancestor, current_authored_path='/synthetic/chunk-04-semantic-v4.json')
        chunk['authored']['retained_source_ancestor'] = {'applies_to_authored_sha256': ancestor['sha256'],
            'authored_path': packet['current_authored_path'], 'snapshot': ancestor,
            'snapshot_canonical_sha256': ancestor['sha256'], 'supplied_base_version': 'v4',
            'question_patch_provenance_location': 'snapshot.retained_source_ancestor.question_patch_provenance',
            'question_patch_provenance_note': 'The supplied historical proof remains exactly in the V4 snapshot under its original ancestry. No V5 snapshot or V5 patch is supplied or created.'}
        validate_retained_source_ancestor(chunk, ancestor)
        mutations = [lambda r: r.update(snapshot_canonical_sha256='0'*64),
                     lambda r: r.update(supplied_base_version='v5'),
                     lambda r: r.update(question_patch_provenance_location='snapshot.question_patch_provenance'),
                     lambda r: r.update(question_patch_provenance_note='Invented V5 history'),
                     lambda r: r['snapshot'].update(approval=True)]
        for mutate in mutations:
            changed = copy.deepcopy(chunk); mutate(changed['authored']['retained_source_ancestor'])
            with self.assertRaises(ValueError): validate_retained_source_ancestor(changed, ancestor)

    def test_computed_flag_view_retains_complete_actual_semantic_author_context(self):
        base, _, _, ancestor = execution_fixture()
        base['source_ancestor_version'] = {'authored': ancestor}
        current = copy.deepcopy(base['authored'])
        who = {'agent_id': '/root/synthetic-source-derived-flags', 'task_id': '/root/synthetic-source-derived-flags',
               'provider': 'openai', 'model': 'fixture-model', 'role': 'derived_flag_metadata_only'}
        current.update(prior_authored_sha256=base['authored']['sha256'], revision_author=who,
                       approval_granted=False, metadata_derivation={'revision_kind': 'derived_flag_metadata_only',
                       'metadata_only': True, 'task_id': who['task_id']})
        seal(current)
        result = {'authored_sha256': current['sha256'], 'author_packet_sha256': current['source_author_packet_sha256'],
                  'input_channel': 'source_only', 'identity': who}
        task = {'schema': 'spacing-native-author-task-evidence-v1', 'canonical_task_name': who['agent_id'],
                'status': 'completed', 'model': who['model'], 'actual_result': result,
                'actual_result_text': 'Synthetic computed view ' + current['sha256'], 'result_source': 'Synthetic observed result'}
        chunk = {'authored': current, 'prior_authored': base['authored'], 'metadata_derivation_base_version': base,
                 'author_task_provenance': [{'role': 'revision_author', 'task_id': who['agent_id'], 'task_record': task, **result}]}
        context = author_context(chunk)
        self.assertIn('task:' + who['agent_id'], context['channels'])
        self.assertIn('task:' + base['author_task_provenance'][0]['task_id'], context['channels'])
        for kind in ('missing-base', 'wrong-base', 'changed-question', 'changed-source-flag', 'approval'):
            changed = copy.deepcopy(chunk)
            if kind == 'missing-base': changed.pop('metadata_derivation_base_version')
            elif kind == 'wrong-base': changed['metadata_derivation_base_version'] = {'authored': ancestor}
            elif kind == 'changed-question': changed['authored']['acquisition_records'][0]['questions'][0] = 'Changed semantics?'
            elif kind == 'changed-source-flag': changed['authored']['source_units'][0]['status'] = 'preserve_source_conflict'
            else: changed['authored']['approval_granted'] = True
            seal(changed['authored'])
            with self.subTest(kind=kind), self.assertRaises(ValueError): author_context(changed)

    def test_retained_actual_source_input_aliases_bind_exact_repair_packet(self):
        chunk, row, _, ancestor = execution_fixture()
        packet = row['semantic_execution_evidence']['input_packet']
        packet['history_requirements'] = {'prior_authored_sha256': ancestor['sha256']}
        chunk['authored']['retained_source_ancestor'].update(
            prior_full_authored_chunk=copy.deepcopy(ancestor), historical_only=True,
            current_review_approval_inferred=False,
            actual_source_review_materials=copy.deepcopy(packet['actual_source_review_materials']),
            history_requirements=copy.deepcopy(packet['history_requirements']))
        validate_retained_source_ancestor(chunk, ancestor)
        mutations = [lambda r: r['actual_source_review_materials'].clear(),
                     lambda r: r['history_requirements'].update(prior_authored_sha256='foreign'),
                     lambda r: r.update(current_review_approval_inferred=True),
                     lambda r: r.pop('prior_full_authored_chunk')]
        for mutate in mutations:
            changed = copy.deepcopy(chunk); mutate(changed['authored']['retained_source_ancestor'])
            with self.assertRaises(ValueError): validate_retained_source_ancestor(changed, ancestor)
        changed = copy.deepcopy(chunk); changed['author_task_provenance'] = []
        with self.assertRaises(ValueError): validate_retained_source_ancestor(changed, ancestor)

    def test_semantic_input_follows_all_exact_question_patch_ancestors(self):
        from test_source_patch import proof_fixture, refresh
        from spacing_rerun.source_patch import reconstruct_question_patch
        semantic, _, _, _ = execution_fixture()
        current = semantic
        for index in range(2):
            proof = proof_fixture(); base = copy.deepcopy(current['authored'])
            proof['base_authored'] = base
            selected = {**copy.deepcopy(base['acquisition_records'][0]), 'question_indices_to_revise': [0]}
            proof['input_packet']['selected_acquisition_records'] = [selected]
            proof['patch']['selected_acquisition_records'] = [copy.deepcopy(selected)]
            proof['patch']['selected_acquisition_records'][0]['questions'][0] = f'Name source entry zero using wording variant {index}?'
            for key in ('input_packet', 'patch'): proof[key]['current_authored_sha256'] = base['sha256']
            proof['patch']['prior_authored_sha256'] = base['sha256']
            proof['response']['prior_authored_sha256'] = base['sha256']; refresh(proof)
            current = {'authored': reconstruct_question_patch(proof), 'question_patch_evidence': proof,
                       'question_patch_base_version': current}
        self.assertIs(source_semantic_base_version(current), semantic)
        for kind in ('missing-link', 'misbound-link', 'missing-proof', 'cycle'):
            changed = copy.deepcopy(current)
            if kind == 'missing-link': changed.pop('question_patch_base_version')
            elif kind == 'misbound-link': changed['question_patch_base_version'] = semantic
            elif kind == 'missing-proof': changed['question_patch_base_version'].pop('question_patch_evidence')
            else: changed['question_patch_base_version'] = changed
            with self.subTest(kind=kind), self.assertRaises(ValueError): source_semantic_base_version(changed)

    def test_inherited_retained_history_is_exact_and_explicitly_historical(self):
        chunk, row, _, ancestor = execution_fixture()
        ancestor['retained_source_ancestor'] = {'applies_to_authored_sha256': 'synthetic-older-version'}
        seal(ancestor); authored = chunk['authored']
        authored['prior_authored_sha256'] = ancestor['sha256']
        authored['retained_source_ancestor'] = {'applies_to_authored_sha256': ancestor['sha256'],
            'authored_snapshot': ancestor, 'inherited_retained_source_ancestor': ancestor['retained_source_ancestor'],
            'historical_only': True,
            'history_scope': 'Exact supplied V5 snapshot and patch proof. All nested author, patch and review references apply to their original prior artifacts only.'}
        seal(authored)
        validate_retained_source_ancestor(chunk, ancestor)
        mutations = [lambda r: r['inherited_retained_source_ancestor'].update(approval=True),
                     lambda r: r.update(historical_only=False),
                     lambda r: r.update(history_scope='Prior approval applies to current content'),
                     lambda r: r['authored_snapshot'].update(completion=True)]
        for mutate in mutations:
            changed = copy.deepcopy(chunk); mutate(changed['authored']['retained_source_ancestor'])
            with self.assertRaises(ValueError): validate_retained_source_ancestor(changed, ancestor)

    def test_full_canonical_and_raw_byte_receipt_hashes_are_verified_separately(self):
        _, audit = confirmation_fixture(); chunk = audit['source_chunks'][0]
        packet = chunk['review_packet']; receipt = copy.deepcopy(chunk['reviews'][0])
        artifact = raw(packet, '/synthetic/packet.json')
        receipt.update(review_packet_canonical_full_sha256=digest(packet),
                       review_packet_raw_bytes_sha256=artifact['file_sha256'])
        version = {**chunk, 'artifact_files': {'review_packet': artifact}}
        declared_receipt_bindings(receipt, packet, [version])
        mutations = [
            lambda r: r.update(review_packet_canonical_full_sha256=packet['sha256']),
            lambda r: r.update(review_packet_raw_bytes_sha256=packet['sha256']),
            lambda r: r.update(review_packet_file_sha256='0'*64),
            lambda r: r.update(unverified_packet_sha256=digest(packet)),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(receipt); mutate(changed)
            with self.assertRaises(ValueError): declared_receipt_bindings(changed, packet, [version])
        changed = copy.deepcopy(version)
        changed['artifact_files']['review_packet']['utf8'] = '{}'
        with self.assertRaises(ValueError): declared_receipt_bindings(receipt, packet, [changed])

    def test_exact_completed_delegated_label_is_bound_separately_from_runtime_identity(self):
        chunk, row, person, _ = execution_fixture()
        task = row['task_record']
        for label, classification in ((task['task_id'], 'actual_delegated_task_id'),
                                      (task['client_request_id'], 'actual_delegated_request_label')):
            with self.subTest(classification=classification):
                actual = {**person, 'agent_id': label, 'task_id': label}
                self.assertEqual(local_author_identity(actual, chunk['authored'], task), classification)
                with self.assertRaises(ValueError):
                    local_author_identity(actual, chunk['authored'], {**task, 'status': 'running'})
        with self.assertRaises(ValueError):
            local_author_identity({**person, 'task_id': 'node:delegated-task:foreign'}, chunk['authored'], task)

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
