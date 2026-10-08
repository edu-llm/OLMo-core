"""Evidence-bound confirmation eligibility; no author payload is auto-approved."""
from __future__ import annotations

import collections
import copy
import datetime
import ast
import hashlib
import importlib.metadata
import os
from pathlib import Path
import random
import subprocess
import re
import uuid

from .common import digest, require, read_json
from .data import REVISION, normalize
from .grounding import FACTSHEETS_SHA256, group_identity
from .units import unit_identity

PROTOCOL = "spacing-confirmation-mixed-acquisition-agent-audit-v1"
AUDIT_SCHEMA = "spacing-confirmation-agent-audit-v1"
QA_SHA256 = "c178467bfd7db74b17c2d57333bb8908f881c0e1abfbe216c7db75f0934765b7"


def checked(payload, schema=None):
    require(isinstance(payload, dict) and payload.get("sha256") == digest({
        key: value for key, value in payload.items() if key != "sha256"}), "Confirmation artifact was modified or lacks its digest")
    require(schema is None or payload.get("schema") == schema, "Unsupported confirmation artifact schema")
    return payload


def identity(person):
    if isinstance(person, str):
        require(person.strip(), "Missing independent review identity")
        return person.strip()
    require(isinstance(person, dict), "Missing author/reviewer identity")
    if person.get("actual_identity") and person.get("model_family") and person.get("human") is False:
        return person["actual_identity"]
    name = person.get("agent_id") or person.get("identity") or person.get('review_task_id') or person.get('review_name') or person.get('name')
    require(name and (person.get("provider") or person.get("harness")) and
            (person.get("model") or person.get("model_id")), "Actual author/reviewer provider/model identity is required")
    return name


def receipt_identity(receipt, author=None):
    """Keep original receipt identity fields; never rewrite an actual report."""
    if receipt.get('reviewer') is not None:
        return identity(receipt['reviewer'])
    legacy = receipt.get('reviewer_identity')
    require(isinstance(legacy, dict) and legacy.get('provider') and legacy.get('model') and
            receipt.get('human_review_complete') is False,
            'Original receipt lacks actual provider/model identity or truthful human status')
    if any(legacy.get(key) for key in ('agent_id', 'identity', 'review_task_id', 'review_name', 'name')):
        return identity(legacy)
    if author is not None:
        # Without a named run identity, a separate model family is required.
        # The original SOL reviews were against Opus author packets.
        require(isinstance(author, dict) and 'opus' in str(author.get('model', '')).casefold() and
                'sol' in str(legacy['model']).casefold(),
                'Unnamed original reviewer does not establish an independent author channel')
    return 'original-reviewer-' + digest(legacy)


def receipt_declaration(receipt, key):
    aliases = {'author_rationales_seen': ('author_rationales_seen', 'author_rationales_used'),
               'model_outcomes_seen': ('model_outcomes_seen', 'model_outcomes_used'),
               'human_review_complete': ('human_review_complete',),
               'acquisition_content_seen': ('acquisition_content_seen',)}[key]
    values = [payload[alias] for payload in (receipt, receipt.get('disclosures', {}))
              for alias in aliases if alias in payload]
    require(not values or all(value is values[0] for value in values), 'Actual receipt visibility aliases contradict each other')
    return values[0] if values else None


def local_author_identity(person, authored, task=None):
    """Classify disclosed runtime labels without treating them as delegated task IDs."""
    label = person.get('task_id')
    if task is not None and label in (task.get('task_id'), task.get('client_request_id')):
        require(task.get('status') == 'completed' and task.get('child_thread_id') and task.get('child_run_id'),
                'Declared delegated author label lacks actual completed execution')
        return 'actual_delegated_task_id' if label == task['task_id'] else 'actual_delegated_request_label'
    name = identity(person)
    if label == name and name in ('/root', 'root'):
        return 'runtime_agent_path'
    sessions = [person[key] for key in ('runtime_thread_id', 'runtime_session_id', 'codex_thread_id', 'thread_id') if person.get(key)]
    if sessions:
        require(len(set(sessions)) == 1, 'Disclosed author runtime session labels disagree')
        session = sessions[0]
        require(str(uuid.UUID(session)) == session, 'Author runtime session is not a canonical UUID')
        allowed = {session, 'codex-thread:' + session, 'codex-thread:' + session + '/root',
                   'codex-thread:' + session + f'/root/chunk-{authored["chunk_index"]:02}-semantic-v4'}
        require(label in allowed and name in ('/root', 'root', 'codex:' + session, 'codex-thread:' + session + ':/root') and
                all(person[key] == '/root' for key in ('agent_path', 'runtime_agent_path') if key in person),
                'Declared author runtime label differs from its disclosed session/path')
        return 'runtime_session_label'
    match = re.fullmatch(r'/root:chunk([0-9]{2})-source-only-semantic-v4:([0-9a-f-]{36})', str(label))
    require(match and int(match[1]) == authored['chunk_index'] and str(uuid.UUID(match[2])) == match[2] and
            name == '/root' and 'local repair execution UUID' in person.get('identity_basis', '') and
            'not an asserted T3 delegated-task ID' in person['identity_basis'],
            'Declared author task ID differs from actual task execution or disclosed local identity')
    return 'disclosed_local_repair_uuid'


def semantic_execution_evidence(chunk, row, person):
    """Bind a local author label to exact completed task and source-only input/result bytes."""
    from .source_patch import actual_bytes, compatible_aliases, normalize_task
    authored = chunk['authored']
    proof = row.get('semantic_execution_evidence', {})
    task = row['task_record']
    status = task.get('actual_task_status', {})
    require(normalize_task(status, task['client_request_id']) == task and
            status.get('latestTerminalStatus') == 'completed' and not status.get('hasPendingChildRuns'),
            'Local author label lacks its exact terminal delegated task evidence')
    packet = checked(proof.get('input_packet'), 'source-only-confirmation-review-and-structural-repair-input-v1')
    response = checked(proof.get('correction_response'))
    for key, value in (('input_packet', packet), ('correction_response', response), ('authored', authored)):
        info = proof.get('artifact_files', {}).get(key, {})
        require(Path(info.get('path', '')).is_absolute(), 'Semantic author evidence lacks an absolute actual artifact path')
        actual_bytes(value, info)
    require(packet.get('evaluation_question_text_included') is False and packet.get('evaluation_identifiers_included') is False and
            packet.get('model_outcomes_included') is False and packet['source_author_packet']['sha256'] == authored['source_author_packet_sha256'] and
            packet['current_authored_sha256'] == authored.get('prior_authored_sha256') and
            packet['chunk_index'] == authored['chunk_index'] and response.get('revision_author') == person and
            response.get('authored_sha256') == authored['sha256'] and
            all(value in status['latestTerminalSummary'] for value in (authored['sha256'], response['sha256'])) and
            person.get('provider', '').casefold() in ('openai', 'codex') and
            str(task.get('provider_instance_id', '')).casefold().startswith('codex'),
            'Local author task is not bound to the exact source-only input, provider and completed result')
    compatible_aliases(response, ('input_packet_sha256', 'repair_input_sha256'), packet['sha256'], required=True)
    compatible_aliases(response, ('prior_authored_sha256',), authored['prior_authored_sha256'], required=True)
    compatible_aliases(response, ('authored_path',), proof['artifact_files']['authored']['path'])
    classification = local_author_identity(person, authored, task)
    require(row.get('declared_task_identity_classification') == classification,
            'Local author task-label classification differs from the preserved raw declaration')
    return packet


def validate_retained_source_input_snapshots(packet, ancestor):
    """Bind preserved actual input snapshots without inferring review approval."""
    import json
    from .source_patch import actual_bytes
    require(packet.get('current_authored') == ancestor,
            'Retained raw source input does not contain the exact immutable ancestor')
    raw_ancestor = packet['current_authored_raw_file']
    require(raw_ancestor.get('path') == packet['current_authored_path'],
            'Retained raw source ancestor path differs from actual input')
    actual_bytes(ancestor, raw_ancestor)
    receipt_file = packet['latest_actual_review_receipt_file']
    packet_file = packet['latest_actual_blind_packet_file']
    receipt = checked(json.loads(receipt_file['utf8']), 'spacing-source-semantic-review-v1')
    blind = checked(json.loads(packet_file['utf8']), 'spacing-source-independent-review-packet-v1')
    actual_bytes(receipt, receipt_file)
    actual_bytes(blind, packet_file)
    require(packet_binding(receipt) == blind['sha256'] and blind.get('authored_chunk_sha256') == ancestor['sha256'] and
            any(material.get('receipt') == receipt and material.get('review_packet') == blind
                for material in packet['actual_source_review_materials']),
            'Retained latest raw review snapshots differ from the actual source review input')
    status = packet['actual_latest_review_task_status']
    status_file = packet['actual_latest_review_task_status_file']
    raw_status = json.loads(status_file['utf8'])
    actual_bytes(raw_status, status_file)
    require(status in [row.get('result') for row in raw_status.get('task_statuses', [])] and
            status.get('status') == status.get('latestTerminalStatus') == 'completed' and
            status.get('hasPendingChildRuns') is False and receipt['sha256'] in status.get('latestTerminalSummary', ''),
            'Retained latest review snapshots lack exact completed actual task evidence')
    context = packet['whole_prior_source_chunk_context']
    binding = packet['whole_prior_context_binding']
    require(context.get('authored') == ancestor and binding.get('extracted_source_chunk_sha256') == digest(context) and
            binding.get('extracted_source_chunk_index') == ancestor['chunk_index'] and
            binding.get('other_source_or_evaluation_chunks_included') is False,
            'Retained whole prior source context differs from the exact input ancestor')


def validate_retained_source_ancestor(chunk, ancestor):
    authored = chunk['authored']
    retained = authored['retained_source_ancestor']
    required = {'applies_to_authored_sha256': ancestor['sha256'], **{
        key: ancestor[key] for key in ('question_patch_provenance', 'retained_historical_annotations') if key in ancestor}}
    require(all(retained.get(key) == value for key, value in required.items()) and
            authored.get('prior_authored_sha256') == ancestor['sha256'] and all(authored[key] == ancestor[key] for key in
                ('source_catalog_sha256', 'source_author_packet_sha256', 'dataset_revision', 'chunk_index', 'events')),
            'Retained source history differs from the exact immutable source-only ancestor')
    proof = next((r.get('semantic_execution_evidence') for r in chunk.get('author_task_provenance', [])
                  if r.get('role') == 'revision_author'), None)
    packet = proof.get('input_packet') if proof else None
    snapshot_fields = {'actual_latest_review_task_status', 'actual_latest_review_task_status_file',
                       'current_authored_raw_file', 'latest_actual_blind_packet_file',
                       'latest_actual_review_receipt_file', 'whole_prior_context_binding',
                       'whole_prior_source_chunk_context'}
    if snapshot_fields & retained.keys():
        require(packet and snapshot_fields <= retained.keys() and snapshot_fields <= packet.keys() and
                all(retained[key] == packet[key] for key in snapshot_fields),
                'Retained actual source snapshots differ from exact source-only input')
        validate_retained_source_input_snapshots(packet, ancestor)
    for key, value in retained.items():
        if key in required:
            continue
        if key in snapshot_fields:
            continue
        if key in ancestor:
            require(value == ancestor[key], 'Retained source ancestor field differs from its immutable original')
        elif key in ('prior_full_authored_chunk', 'authored_snapshot', 'snapshot'):
            require(value == ancestor, 'Retained full source snapshot differs from immutable ancestor')
        elif key == 'snapshot_canonical_sha256':
            require(retained.get('snapshot') == ancestor and value == ancestor['sha256'],
                    'Retained snapshot canonical seal differs from immutable ancestor')
        elif key == 'supplied_base_version':
            require(packet and retained.get('snapshot') == ancestor and value == 'v4' and
                    Path(packet['current_authored_path']).name.endswith('-semantic-v4.json'),
                    'Declared supplied base version differs from exact source-only input')
        elif key == 'question_patch_provenance_location':
            require(retained.get('snapshot') == ancestor and value ==
                    'snapshot.retained_source_ancestor.question_patch_provenance' and
                    ancestor.get('retained_source_ancestor', {}).get('question_patch_provenance'),
                    'Retained historical patch location differs from immutable snapshot')
        elif key == 'question_patch_provenance_note':
            require(retained.get('snapshot') == ancestor and retained.get('supplied_base_version') == 'v4' and value ==
                    'The supplied historical proof remains exactly in the V4 snapshot under its original ancestry. No V5 snapshot or V5 patch is supplied or created.',
                    'Retained historical patch note is unsupported')
        elif key == 'inherited_retained_source_ancestor':
            require('retained_source_ancestor' in ancestor and value == ancestor['retained_source_ancestor'],
                    'Inherited retained source history differs from immutable ancestor')
        elif key == 'historical_only':
            require(value is True and any(retained.get(name) == ancestor for name in
                    ('authored_snapshot', 'prior_full_authored_chunk')),
                    'Historical-only source annotation lacks its exact immutable snapshot')
        elif key in ('actual_source_review_materials', 'history_requirements'):
            require(packet and key in packet and value == packet[key],
                    'Retained source input field differs from exact actual repair packet: ' + key)
        elif key == 'current_review_approval_inferred':
            require(value is False and any(retained.get(name) == ancestor for name in
                    ('authored_snapshot', 'prior_full_authored_chunk')),
                    'Retained source history cannot infer current review approval')
        elif key == 'history_scope':
            require(retained.get('authored_snapshot') == ancestor and value ==
                    'Exact supplied V5 snapshot and patch proof. All nested author, patch and review references apply to their original prior artifacts only.',
                    'Retained source history scope is unsupported')
        elif key == 'historical_annotations':
            require(value == {k: v for k, v in ancestor.items() if k not in
                    ('source_units', 'groups', 'acquisition_records', 'sha256', 'question_patch_provenance')},
                    'Retained historical annotation projection differs from immutable ancestor')
        elif key == 'historical_annotation_scope':
            require(retained.get('authored_snapshot') == ancestor and value ==
                    'Every value in authored_snapshot, including review references, revision history, author metadata and record annotations, applies only to the immutable ancestor identified above.',
                    'Retained historical annotation scope is unsupported')
        elif key == 'replaced_author_annotations':
            require(isinstance(value, list), 'Invalid retained author annotation change ledger')
            seen = set()
            for change in value:
                collection, field = change.get('collection'), change.get('field')
                require((collection, field) in (('groups', 'grouping_rationale'), ('acquisition_records', 'rationale')) and
                        set(change) == {'collection', 'field', 'id', 'before', 'after', 'source_only_reason'},
                        'Retained author annotation ledger contains unsupported fields')
                before = {r['id']: r for r in ancestor[collection]}; after = {r['id']: r for r in authored[collection]}
                signature = collection, field, change['id']
                require(signature not in seen and change['id'] in before and change['id'] in after and
                        change['before'] == before[change['id']][field] and change['after'] == after[change['id']][field] and
                        change['source_only_reason'], 'Retained author annotation ledger differs from exact before/after content')
                seen.add(signature)
        elif key in ('actual_prior_review_references', 'source_review_history'):
            require(packet is not None and isinstance(value, list), 'Retained review references lack actual source-only input')
            for ref in value:
                material = next((m for m in packet['actual_source_review_materials'] if m['receipt']['sha256'] == ref.get('receipt_sha256')), None)
                allowed = {'receipt_path', 'receipt_sha256', 'review_packet_path', 'review_packet_sha256',
                           'applies_to_review_packet_sha256', 'reviewed_authored_sha256'}
                require(material and set(ref) <= allowed and all(ref.get(k) == material[k] for k in ('receipt_path', 'review_packet_path')) and
                        ref.get('review_packet_sha256') == material['review_packet']['sha256'] and
                        ref.get('applies_to_review_packet_sha256', material['review_packet']['sha256']) == material['review_packet']['sha256'] and
                        ref.get('reviewed_authored_sha256', material['review_packet']['authored_chunk_sha256']) == material['review_packet']['authored_chunk_sha256'],
                        'Retained review reference differs from exact actual source review material')
        elif key == 'authored_path':
            require(proof and packet['current_authored'] == ancestor and value == packet['current_authored_path'],
                    'Retained authored path differs from exact source-only ancestor input')
        elif key == 'input_packet_sha256':
            require(packet and value == packet['sha256'], 'Retained repair input hash differs from exact source-only evidence')
        else:
            require(False, 'Unsupported retained source ancestor field: ' + key)


def derived_source_metadata_base_version(chunk, authored=None):
    """Retain the entire semantic author version beneath a computed flag view."""
    authored = checked(authored or chunk['authored'], 'spacing-confirmation-authored-source-chunk-v1')
    prior = checked(chunk.get('prior_authored'), authored['schema'])
    derivation = authored.get('metadata_derivation', {})
    require(derivation.get('revision_kind') == 'derived_flag_metadata_only' and derivation.get('metadata_only') is True and
            authored.get('prior_authored_sha256') == prior['sha256'] and
            authored.get('revision_author', {}).get('role') == 'derived_flag_metadata_only' and
            derivation.get('task_id') == authored['revision_author'].get('task_id') and
            authored.get('approval_granted') is False,
            'Derived source-flag cleanup lost its actual prior artifact/task or claims approval')
    excluded = {'sha256', 'source_units', 'prior_authored_sha256', 'revision_author', 'metadata_derivation', 'approval_granted'}
    require({key: value for key, value in authored.items() if key not in excluded} ==
            {key: value for key, value in prior.items() if key not in excluded} and
            len(authored['source_units']) == len(prior['source_units']) and all(
                {key: value for key, value in current.items() if key != 'nonliteral_answer_labels'} ==
                {key: value for key, value in old.items() if key != 'nonliteral_answer_labels'}
                for current, old in zip(authored['source_units'], prior['source_units'])),
            'Derived source-flag cleanup changed semantic content or source membership')
    base = chunk.get('metadata_derivation_base_version', {'authored': prior,
        'author_task_provenance': chunk.get('prior_author_task_provenance', [])})
    require(base.get('authored') == prior, 'Derived source metadata base differs from exact immutable prior version')
    require(not (prior.get('retained_source_ancestor') or prior.get('question_patch_provenance') or
                 prior.get('metadata_derivation')) or 'metadata_derivation_base_version' in chunk,
            'Derived source metadata lost complete retained semantic ancestry')
    return base


def source_semantic_base_version(chunk):
    """Follow exact retained question-patch links to their semantic author base."""
    current, seen = chunk, set()
    while current['authored'].get('question_patch_provenance') or current['authored'].get('metadata_derivation'):
        authored = checked(current['authored'], 'spacing-confirmation-authored-source-chunk-v1')
        require(authored['sha256'] not in seen, 'Cyclic retained question patch ancestry')
        seen.add(authored['sha256'])
        if authored.get('metadata_derivation'):
            base = derived_source_metadata_base_version(current)
            require(base['authored']['sha256'] not in seen, 'Cyclic retained source metadata ancestry')
            current = base
            continue
        proof = current.get('question_patch_evidence')
        require(proof and isinstance(proof.get('base_authored'), dict), 'Missing exact retained question patch base')
        base = current.get('question_patch_base_version', {'authored': proof['base_authored'],
            'author_task_provenance': current.get('base_author_task_provenance', [])})
        prior = checked(base.get('authored'), authored['schema'])
        require(prior == proof['base_authored'] and
                prior['sha256'] == authored['question_patch_provenance'].get('base_authored_sha256'),
                'Question patch ancestry link differs from its exact immutable base')
        require(prior['sha256'] not in seen, 'Cyclic retained question patch ancestry')
        require(not prior.get('question_patch_provenance') or 'question_patch_base_version' in current,
                'Missing complete retained question patch ancestry')
        current = base
    return current


def author_context(chunk, *, evaluation=False):
    authored = chunk['authored']
    if authored.get('question_patch_provenance') or authored.get('metadata_derivation'):
        source_semantic_base_version(chunk)
    people = {'author': authored['author']}
    if authored.get('revision_author'):
        people['revision_author'] = authored['revision_author']
    provenance = chunk.get('author_task_provenance', [])
    by_role = {row['role']: row for row in provenance}
    require(len(by_role) == len(provenance) and set(by_role) <= set(people), 'Duplicate or unknown actual author-task provenance')
    channels = []
    for role, person in people.items():
        name = identity(person)
        row = by_role.get(role)
        generic = name == '/root' or name == 'root'
        require(row is not None or (not generic and role == 'author'),
                'Generic/revision author identity requires actual independent task provenance')
        if row is None:
            channels.append('named:' + name)
            continue
        task = row['task_record']
        author_key = 'evaluation_author_packet_sha256' if evaluation else 'source_author_packet_sha256'
        expected_channel = 'evaluation_only' if evaluation else 'source_only'
        model = str(person.get('model', person.get('model_id', ''))).casefold()
        actual_model = str(task.get('model', '')).casefold()
        native = task.get('schema') == 'spacing-native-author-task-evidence-v1'
        task_id = task.get('canonical_task_name') if native else task.get('task_id')
        execution_complete = (task.get('canonical_task_name', '').startswith('/root/') and task.get('actual_result') and
                              task.get('actual_result_text') and task.get('result_source')) if native else (task.get('child_thread_id') and task.get('child_run_id') and task.get('client_request_id'))
        require(row.get('identity') == person and row.get('authored_sha256') == authored['sha256'] and
                row.get('author_packet_sha256') == authored[author_key] and row.get('input_channel') == expected_channel and
                row.get('task_id') == task_id and task.get('status') == 'completed' and execution_complete and
                (model == actual_model or ('opus' in model and 'opus' in actual_model)),
                'Actual author/revision task provenance differs from authored source/identity/channel')
        if native:
            require(task['actual_result'] == {'authored_sha256': authored['sha256'], 'author_packet_sha256': authored[author_key],
                                             'input_channel': expected_channel, 'identity': person},
                    'Actual native author result is not bound to its source-only payload')
        else:
            request = task['client_request_id']
            metadata_only = person.get('role') in ('provenance_metadata_only', 'derived_flag_metadata_only')
            metadata_request = ('derived-flags-cleanup-' if person.get('role') == 'derived_flag_metadata_only'
                                else 'metadata-provenance-cleanup-')
            revision_request = any(token in request for token in ('repair-', 'revision-', 'revise-', 'wording-'))
            require(('evaluation' if evaluation else 'source') in request and
                    (('author-' in request or revision_request) if role == 'author' else
                     (metadata_request in request if metadata_only else revision_request)),
                    'Actual author task belongs to a different input channel or role')
            if person.get('task_id'):
                if person['task_id'] not in (task_id, task['client_request_id']):
                    require(role == 'revision_author' and not evaluation,
                            'Declared author task ID differs from actual task execution')
                    semantic_execution_evidence(chunk, row, person)
        channels.append('task:' + task_id)
    if authored.get('question_patch_provenance'):
        from .source_patch import reconstruct_question_patch
        patch_evidence = chunk.get('question_patch_evidence')
        require(patch_evidence and reconstruct_question_patch(patch_evidence) == authored,
                'Mechanical source question merge differs from its exact actual source-only patch')
        base_version = chunk.get('question_patch_base_version', {'authored': patch_evidence['base_authored'],
                                     'author_task_provenance': chunk.get('base_author_task_provenance', [])})
        require(base_version.get('authored') == patch_evidence['base_authored'],
                'Question patch retained base version differs from its immutable actual base')
        prior_context = author_context(base_version)
        people.update({'retained_' + key: person for key, person in prior_context['people'].items()})
        channels.extend(prior_context['channels'])
        provenance = list(provenance) + list(prior_context['task_provenance'])
    if authored.get('metadata_derivation'):
        base_version = derived_source_metadata_base_version(chunk)
        prior_context = author_context(base_version)
        people.update({'metadata_base_' + key: person for key, person in prior_context['people'].items()})
        channels.extend(prior_context['channels'])
        provenance = list(provenance) + list(prior_context['task_provenance'])
    if authored.get('retained_source_ancestor') and not authored.get('question_patch_provenance') and not authored.get('metadata_derivation'):
        ancestor_version = chunk.get('source_ancestor_version', {})
        ancestor = checked(ancestor_version.get('authored'), authored['schema'])
        validate_retained_source_ancestor(chunk, ancestor)
        prior_context = author_context(ancestor_version)
        people.update({'ancestor_' + key: person for key, person in prior_context['people'].items()})
        channels.extend(prior_context['channels'])
        provenance = list(provenance) + list(prior_context['task_provenance'])
    return {'author_context': True, 'people': people, 'channels': channels,
            'task_provenance': provenance, 'reviewer_task_provenance': chunk.get('reviewer_task_provenance', [])}


def packet_binding(receipt):
    names = [key for key in ('review_packet_sha256', 'packet_sha256') if key in receipt]
    require(names and len({receipt[key] for key in names}) == 1, 'Actual receipt packet aliases are missing or contradictory')
    return receipt[names[0]]


def decisions_for(receipt, field):
    aliases = {'groups': ('groups', 'group_reviews'), 'acquisition_records': ('acquisition_records', 'qa_reviews', 'acquisition_reviews'),
               'records': ('records', 'record_reviews')}[field]
    present = [receipt[key] for key in aliases if key in receipt]
    require(not present or all(value == present[0] for value in present), 'Actual receipt record aliases contradict each other')
    return present[0] if present else []


def raw_artifact(payload, provenance):
    require(provenance and isinstance(provenance.get('utf8'), str) and
            hashlib.sha256(provenance['utf8'].encode()).hexdigest() == provenance.get('file_sha256'),
            'Actual raw artifact bytes or file hash are missing or changed')
    import json
    require(json.loads(provenance['utf8']) == payload, 'Actual raw artifact bytes differ from embedded canonical artifact')
    return provenance['file_sha256']


def declared_receipt_bindings(receipt, packet, versions):
    require(packet_binding(receipt) == packet['sha256'], 'Actual receipt packet binding differs')
    direct = {'source_catalog_sha256': 'source_catalog_sha256', 'outer_partition_sha256': 'outer_partition_sha256',
              'evaluation_catalog_sha256': 'evaluation_catalog_sha256', 'author_packet_sha256': 'author_packet_sha256',
              'source_author_packet_sha256': 'source_author_packet_sha256', 'authored_chunk_sha256': 'authored_chunk_sha256'}
    for declared, actual in direct.items():
        if declared in receipt:
            require(actual in packet and receipt[declared] == packet[actual], 'Declared actual receipt provenance differs: ' + declared)
    if 'chunk_index' in receipt:
        require(receipt['chunk_index'] == packet.get('chunk_index'), 'Declared actual receipt chunk index differs')
    packet_versions = {version['review_packet']['sha256']: version for version in versions}
    bound_version = packet_versions[packet['sha256']]
    allowed_hashes = set(direct) | {'sha256', 'review_packet_sha256', 'packet_sha256', 'packet_file_sha256',
                                  'previous_packet_sha256', 'previous_packet_file_sha256', 'previous_review_packet_sha256',
                                  'author_packet_file_sha256', 'evaluation_catalog_file_sha256', 'authored_chunk_file_sha256',
                                  'review_packet_file_sha256', 'review_packet_canonical_full_sha256',
                                  'review_packet_raw_bytes_sha256'}
    require(all(not key.endswith('_sha256') or key in allowed_hashes for key in receipt),
            'Actual receipt has an unverified declared provenance hash')
    if 'review_packet_canonical_full_sha256' in receipt:
        require(receipt['review_packet_canonical_full_sha256'] == digest(packet),
                'Actual receipt full canonical packet hash differs')
    packet_file = bound_version.get('artifact_files', {}).get('review_packet')
    for path_key in ('packet_path', 'review_packet_path'):
        if path_key in receipt:
            declared_path = Path(receipt[path_key])
            actual_path = Path(packet_file['path']) if packet_file else None
            require(actual_path and ('..' not in declared_path.parts) and
                    (declared_path == actual_path if declared_path.is_absolute() else
                     actual_path.parts[-len(declared_path.parts):] == declared_path.parts),
                    'Declared actual packet path differs')
    raw_names = [key for key in ('packet_file_sha256', 'review_packet_file_sha256',
                                'review_packet_raw_bytes_sha256') if key in receipt]
    if raw_names:
        require(len({receipt[key] for key in raw_names}) == 1 and
                raw_artifact(packet, packet_file) == receipt[raw_names[0]],
                'Actual receipt raw packet hash differs')
    for declared, field, payload in (('author_packet_file_sha256', 'author_packet', bound_version.get('author_packet')),
                                    ('evaluation_catalog_file_sha256', 'authored', bound_version['authored']),
                                    ('authored_chunk_file_sha256', 'authored', bound_version['authored'])):
        if declared in receipt:
            require(raw_artifact(payload, bound_version.get('artifact_files', {}).get(field)) == receipt[declared],
                    'Declared actual raw author provenance differs')
    prior_names = [key for key in ('previous_packet_sha256', 'previous_review_packet_sha256') if key in receipt]
    previous_sha = None
    if prior_names:
        require(len({receipt[key] for key in prior_names}) == 1, 'Actual previous packet aliases contradict each other')
        previous_sha = receipt[prior_names[0]]
        require(previous_sha in packet_versions, 'Actual previous review packet is not retained')
        previous = packet_versions[previous_sha]
        previous_file = previous.get('artifact_files', {}).get('review_packet')
        if 'previous_packet_path' in receipt:
            require(previous_file and previous_file.get('path') == receipt['previous_packet_path'], 'Actual prior packet path differs')
        if 'previous_packet_file_sha256' in receipt:
            require(raw_artifact(previous['review_packet'], previous_file) == receipt['previous_packet_file_sha256'],
                    'Actual prior raw packet hash differs')
        if 'change_verification' in receipt:
            verification = receipt['change_verification']
            before = {row['id']: row for row in previous['review_packet']['records']}
            after = {row['id']: row for row in packet['records']}
            require(set(before) == set(after), 'Claimed record preservation changed packet membership')
            changed_fields = sorted({key for rid in before for key in set(before[rid]) | set(after[rid])
                                     if before[rid].get(key) != after[rid].get(key)})
            require(changed_fields == sorted(verification['changed_fields']) and all(
                    verification.get(key + '_preserved') is True and all(before[rid][key] == after[rid][key] for rid in before)
                    for key in ('canonical_question', 'canonical_answer', 'aliases', 'evidence')) and
                    verification.get('record_ids_unchanged') is True and all(verification.get(key + '_unchanged') is True and
                    previous['review_packet'][key] == packet[key] for key in ('source_assertions', 'events', 'factsheets')),
                    'Actual revision-preservation declarations differ from retained blind packet content')
    carry = receipt.get('unchanged_records_retained_from_original_receipt')
    if carry:
        historical = [version for version in versions if version['review_packet']['sha256'] == previous_sha]
        require(len(historical) == 1, 'Actual carryforward summary has no unique historical packet')
        old = historical[0]
        matches = [(review, info) for review in old.get('reviews', []) for info in old.get('artifact_files', {}).get('reviews', [])
                   if info.get('canonical_sha256') == review['sha256'] and info.get('file_sha256') == carry.get('receipt_file_sha256')]
        require(len(matches) == 1 and raw_artifact(*matches[0]) == carry['receipt_file_sha256'],
                'Actual carryforward summary refers to unretained raw review receipt')
        require(matches[0][1].get('path') == carry.get('receipt_path'), 'Actual carryforward receipt path differs')
        current = {row['id']: digest(row) for row in packet['records']}
        unchanged = sorted(row['id'] for row in old['review_packet']['records'] if current.get(row['id']) == digest(row))
        require(len(unchanged) == carry['unchanged_record_count'] and digest(unchanged) == carry['unchanged_record_ids_sha256'],
                'Actual unchanged-record inventory differs from blind packet history')


def evidence(entries, event, factsheets, *, allow_empty=False):
    require((entries or allow_empty) and event in factsheets, "Missing exact confirmation source evidence")
    lines = factsheets[event]["text"].splitlines()
    require(all(type(entry.get("line")) is int and 0 <= entry["line"] < len(lines) and
                entry.get("quote") == lines[entry["line"]] for entry in entries),
            "Confirmation evidence differs from the pinned source line")


def review_history(chunk, *, evaluation=False):
    """Bind previous author versions and their actual blind input projections."""
    current = chunk['authored']
    versions = []
    for version in chunk.get('history', []):
        authored = checked(version['authored'], current['schema'])
        packet = checked(version['review_packet'], chunk['review_packet']['schema'])
        validate_blind_packet(packet, evaluation=evaluation)
        require(authored['source_catalog_sha256'] == packet['source_catalog_sha256'] == current['source_catalog_sha256'] and
                authored['dataset_revision'] == current['dataset_revision'] and authored.get('human_review_complete') is False and
                authored.get('independent_agent_review_complete') is False and authored.get('model_outcomes_used') is False,
                'Historical review changed source/author provenance or claims completion')
        identity(authored['author'])
        if evaluation:
            require(authored['evaluation_author_packet_sha256'] == current['evaluation_author_packet_sha256'] ==
                    packet['author_packet_sha256'] and packet['evaluation_catalog_sha256'] == authored['sha256'] and
                    authored['outer_partition_sha256'] == current['outer_partition_sha256'] and
                    authored.get('acquisition_content_seen') is False and packet.get('author_rationales_included') is False and
                    packet.get('acquisition_content_included') is False and packet.get('model_outcomes_included') is False and
                    packet['records'] == [evaluation_review_content(row) for row in authored['records']],
                    'Historical evaluation receipt lacks its actual blind authored-content projection')
        else:
            require(authored['source_author_packet_sha256'] == current['source_author_packet_sha256'] ==
                    packet['source_author_packet_sha256'] and packet['authored_chunk_sha256'] == authored['sha256'] and
                    authored['events'] == current['events'] == packet['events'] and authored['chunk_index'] == current['chunk_index'] == packet['chunk_index'] and
                    authored.get('evaluation_question_text_included') is False and packet.get('author_rationales_included') is False and
                    packet['source_units'] == chunk['review_packet']['source_units'] and packet['factsheets'] == chunk['review_packet']['factsheets'] and
                    all(packet[field] == [source_review_content(row, field) for row in authored[field]]
                        for field in ('groups', 'acquisition_records')),
                    'Historical source receipt lacks its actual blind authored-content projection')
        versions.append((packet, version.get('reviews', []), author_context(version, evaluation=evaluation)))
    return versions


def reviewed_records(packet, receipts, field, author, *, schema, history=(), artifact_versions=None):
    """Every required record needs actual independent approved content receipts."""
    checked(packet)
    records = packet.get(field, [])
    require(records and len({record["id"] for record in records}) == len(records), "Missing or duplicated review packet records")
    versions = list(history) + [(packet, receipts, author)]
    require(any(actual_receipts for _, actual_receipts, _ in versions), "Missing actual independent semantic review receipts")
    judgments, actual_receipts = [], {}
    for bound_packet, version_receipts, version_author in versions:
        checked(bound_packet)
        by_id = {row['id']: row for row in bound_packet.get(field, [])}
        require(len(by_id) == len(bound_packet.get(field, [])), 'Duplicate historical review packet record')
        for receipt in version_receipts:
            checked(receipt, schema)
            if artifact_versions is not None:
                declared_receipt_bindings(receipt, bound_packet, artifact_versions)
            if version_author.get('author_context'):
                people = version_author['people']
                reviewer_name = receipt_identity(receipt, people['author'])
                reviewer_channel = 'named:' + reviewer_name
                generic_author = any(identity(person) in ('/root', 'root') for person in people.values())
                if reviewer_name in ('/root', 'root') and generic_author:
                    rows = [row for row in version_author['reviewer_task_provenance'] if row.get('receipt_sha256') == receipt['sha256']]
                    require(len(rows) == 1, 'Generic overlapping reviewer/author identities require actual separate task provenance')
                    row = rows[0]; task = row['task_record']
                    require(row.get('reviewer') == receipt.get('reviewer') and row.get('review_packet_sha256') == bound_packet['sha256'] and
                            row.get('task_id') == task.get('task_id') and task.get('status') == 'completed' and
                            task.get('child_thread_id') and task.get('child_run_id'), 'Actual reviewer task provenance differs')
                    reviewer_channel = 'task:' + task['task_id']
                require(reviewer_channel not in version_author['channels'] and all(
                        identity(person) != reviewer_name or reviewer_name in ('/root', 'root') for person in people.values()),
                        'Actual reviewer is an author/revision author in the same task channel')
            else:
                require(receipt_identity(receipt, version_author) != identity(version_author), "Review receipt packet/independent identity differs")
            require(packet_binding(receipt) == bound_packet['sha256'], 'Review receipt packet/independent identity differs')
            author_visibility = receipt_declaration(receipt, 'author_rationales_seen')
            outcome_visibility = receipt_declaration(receipt, 'model_outcomes_seen')
            if schema == 'spacing-evaluation-semantic-review-v1' or all(
                    value is not None for value in (author_visibility, outcome_visibility)):
                require(author_visibility is False and outcome_visibility is False,
                        "Independent reviewers received author rationales or model outcomes")
            else:
                require(schema == 'spacing-source-semantic-review-v1' and
                        bound_packet.get('author_rationales_included') is False and
                        receipt_declaration(receipt, 'human_review_complete') is False and
                        author_visibility is not True and outcome_visibility is not True,
                        'Original source receipt is not bound to its actual source-only blind packet')
            if schema == "spacing-evaluation-semantic-review-v1":
                require(receipt.get("acquisition_content_seen") is False and receipt.get("human_review_complete") is False,
                        "Evaluation semantic review did not preserve its separate input channel")
            decisions = decisions_for(receipt, field)
            require(len({row['id'] for row in decisions}) == len(decisions) and {row['id'] for row in decisions} <= set(by_id),
                    'Independent review decision lost or duplicated packet membership')
            require(all(row.get('status') in ('approved', 'needs_revision') and
                        row.get('reviewed_content_sha256') == digest(by_id[row['id']]) and row.get('notes') for row in decisions),
                    'Unapproved, stale or unexplained semantic review decision')
            actual_receipts[receipt['sha256']] = receipt
            judgments.extend((receipt['sha256'], row) for row in decisions)
    superseded = set()
    for receipt_sha, receipt in actual_receipts.items():
        for edge in receipt.get('supersedes', []):
            if edge.get('field') != field:
                continue
            prior_sha = edge.get('review_receipt_sha256')
            target = (edge.get('id'), edge.get('reviewed_content_sha256'))
            require(prior_sha in actual_receipts and prior_sha != receipt_sha and
                    any(sha == prior_sha and row['status'] == 'needs_revision' and (row['id'], row['reviewed_content_sha256']) == target
                        for sha, row in judgments) and
                    any(sha == receipt_sha and row['status'] == 'approved' and (row['id'], row['reviewed_content_sha256']) == target
                        for sha, row in judgments), 'Actual review supersession edge does not bind a prior rejection and new approval')
            superseded.add((prior_sha, *target))
    for record in records:
        target = record['id'], digest(record)
        relevant = [(sha, row) for sha, row in judgments if (row['id'], row['reviewed_content_sha256']) == target]
        require(any(row['status'] == 'approved' for _, row in relevant) and all(
            row['status'] == 'approved' or (sha, *target) in superseded for sha, row in relevant),
            'Unapproved current content or unresolved semantic review rejection')


def source_components(cross):
    """Read evidence-protected role components without certifying adjudication."""
    review = checked(cross["review"], "spacing-cross-entity-semantic-review-v1")
    components = [row["events"] for row in review.get("required_same_role_components", [])]
    components += [row["events"] for row in review.get("other_same_proposition_components_requiring_protection", [])]
    return [sorted(set(component)) for component in components]


def assign_confirmation_roles(facts, partition, seed, components):
    """Use the original 128 metadata candidates, retaining same-role components."""
    counts = {"old": 20, "new": 30, "control": 20, "qa": 10}
    events = list(partition["confirmation"])
    require(len(events) == 80 and all(component and set(component) <= set(events) for component in components),
            "Cross-partition role component requires an explicit partition amendment and fresh development checks")
    style_counts = {event: collections.Counter(fact["style"] for fact in facts if fact["event"] == event) for event in events}
    require(all(style_counts.values()), "Missing confirmation source metadata event")
    styles = sorted({style for values in style_counts.values() for style in values})
    rng, best, best_score = random.Random(seed), None, float("inf")
    eligible_candidates = 0
    for _ in range(128):
        rng.shuffle(events)
        offset, candidate, score = 0, {}, 0.
        for role, count in counts.items():
            chosen = events[offset:offset + count]
            candidate.update({event: role for event in chosen})
            totals = collections.Counter()
            for event in chosen:
                totals.update(style_counts[event])
            mean = sum(totals.values()) / max(len(styles), 1)
            score += sum((totals[style] - mean) ** 2 for style in styles) / count
            offset += count
        if any(len({candidate[event] for event in component}) != 1 for component in components):
            continue
        eligible_candidates += 1
        if score < best_score:
            best, best_score = candidate.copy(), score
    require(best is not None, "No prespecified metadata role candidate satisfies reviewed same-role protection")
    result = {"mode": "confirmation", "seed": seed, "counts": counts, "roles": best,
              "style_imbalance": best_score, "facts": [dict(fact, role=best[fact["event"]]) for fact in facts if fact["event"] in best],
              "outer_partition_sha256": partition["sha256"], "same_role_components": components,
              "role_assignment_policy": "original_128_metadata_candidates_with_same_role_components_v1",
              "eligible_metadata_candidates": eligible_candidates}
    result["sha256"] = digest(result)
    return result


def validate_catalog(catalog, probe_map, source_facts=None, pinned_factsheets=None):
    checked(catalog, "spacing-confirmation-source-catalog-v1")
    checked(catalog["partition"])
    checked(probe_map)
    require(catalog["dataset_revision"] == REVISION and catalog["outer_partition_sha256"] == catalog["partition"]["sha256"] and
            catalog["pinned_parquet_sha256"] == {"fict_qa": QA_SHA256, "fictsheets": FACTSHEETS_SHA256} and
            catalog.get("human_review_complete") is False and catalog.get("role_independent") is True,
            "Source inventory lacks pinned revision/partition/bytes or truthful review provenance")
    require(probe_map["source_catalog_sha256"] == catalog["sha256"], "Private probe mapping refers to a different source catalog")
    units = catalog["units"]
    require(len({unit["unit_id"] for unit in units}) == len(units), "Duplicated original confirmation source units")
    events = set(catalog["partition"]["confirmation"])
    require(len(events) == 80 and {unit["event"] for unit in units} == set(catalog["factsheets"]) == events,
            "Confirmation source catalog requires complete 80-event coverage")
    probes = {}
    for unit in units:
        checked(unit)
        require(unit["unit_id"] == unit_identity(unit["event"], unit["source_assertion"])[0], "Unstable original source-unit identity")
        for probe in unit["canonical_probes"]:
            opaque = probe["opaque_probe_id"]
            require(opaque not in probes, "Lost or duplicated canonical source probe")
            mapped = probe_map["probes"].get(opaque)
            require(mapped and mapped["source_unit_id"] == unit["unit_id"] and mapped["event"] == unit["event"] and
                    opaque == "probe-" + digest([REVISION, mapped["original_probe_id"]]), "Original/opaque probe membership differs")
            probes[opaque] = (unit, probe, mapped)
        require(sorted(set(unit["canonical_answer_labels"])) == sorted({p["source_answer_label"] for p in unit["canonical_probes"]}),
                "Source unit answer-target coverage differs")
    require(set(probes) == set(probe_map["probes"]), "Private probe mapping lost or added original canonical probes")
    for event, factsheet in catalog["factsheets"].items():
        require(factsheet["sha256"] == digest(factsheet["text"]), "Catalog factsheet was modified")
        require(pinned_factsheets is None or factsheet["text"] == pinned_factsheets[event], "Catalog factsheet differs from pinned parquet")
    coverage = {"events": len(events), "source_units": len(units), "canonical_probes": len(probes),
                "source_answer_targets": sum(len({normalize(answer) for answer in unit["canonical_answer_labels"]}) for unit in units)}
    require(catalog["coverage"] == coverage, "Source inventory coverage report differs from actual source content")
    if source_facts is not None:
        selected = {fact["id"]: fact for fact in source_facts if fact["event"] in events}
        require(set(selected) == {mapped["original_probe_id"] for _, _, mapped in probes.values()}, "Catalog does not preserve every pinned canonical probe")
        for unit, probe, mapped in probes.values():
            fact = selected[mapped["original_probe_id"]]
            require(fact["event"] == unit["event"] and normalize(fact.get("source_statement", fact["statement"])) == normalize(unit["source_assertion"]) and
                    fact["answer"] == probe["source_answer_label"] and fact["style"] == probe["source_style"] and
                    sorted("probe-" + digest([REVISION, member]) for member in fact["members"]) == sorted(probe["opaque_duplicate_cluster_members"]),
                    "Catalog source assertion/answer/style/duplicate-cluster differs from pinned source")
    return units, probes


def source_review_content(record, field):
    keys = ("id", "event", "source_unit_ids", "statement", "evidence") if field == "groups" else (
        "id", "unit_id", "event", "answer", "questions", "evidence")
    return {key: record[key] for key in keys}


def source_preservation_context(authored):
    """Fixed preservation obligations and bare dispositions, without author explanations."""
    return {'schema': 'spacing-source-preservation-protocol-context-v1',
            'preserved_declaration_rule': 'Every preserve_source_conflict or preserve_source_ambiguity unit must remain a singleton group whose declaration equals its raw original source_assertion literally.',
            'support_review_rule': 'Assess grounding, evidence, source limitations and question-to-target faithfulness independently. A preservation status does not assert factsheet support or resolve a source conflict.',
            'membership_review_rule': 'Preserved source conflicts and ambiguities cannot be merged or rewritten. Judge dependency and equivalence claims using the supplied source assertions and factsheets while retaining that obligation.',
            'visibility': 'Bare source disposition labels and fixed protocol obligations only; no author rationales, author judgments, prior reviewer findings, evaluator questions, model outcomes or completion claims.',
            'source_dispositions': [{'unit_id': row['unit_id'], 'status': row['status']} for row in authored['source_units']
                                    if row['status'] in ('preserve_source_conflict', 'preserve_source_ambiguity')]}


def validate_blind_packet(packet, *, evaluation=False):
    forbidden = {'author_rationales', 'author_judgments', 'author_notes', 'model_outputs', 'model_outcomes'}
    if evaluation:
        forbidden.update(('acquisition_records', 'acquisition_questions', 'acquisition_qa'))
    require(not forbidden & set(packet), 'Blind review packet contains prohibited author or outcome content')


def evaluation_review_content(record):
    hidden = {"source_ambiguity_status", "source_ambiguity_notes", "review_status", "notes", "rationale"}
    visible = {'id', 'opaque_probe_id', 'event', 'source_unit_id', 'canonical_question', 'canonical_answer',
               'original_source_assertion', 'aliases', 'paraphrase', 'evidence'}
    require(set(record) <= hidden | visible, 'Evaluation review record contains unknown potentially unblinded content')
    return {key: value for key, value in record.items() if key not in hidden}


def validate_source_chunks(chunks, catalog):
    from .acquisition import acquisition_identity
    raw = {unit["unit_id"]: unit for unit in catalog["units"]}
    all_sources, all_groups, all_qa, author_ids = [], [], [], set()
    for chunk in chunks:
        packet = checked(chunk["author_packet"], "source-only-confirmation-authoring-input-v1")
        authored = checked(chunk["authored"], "spacing-confirmation-authored-source-chunk-v1")
        current_authored = authored
        if authored.get('question_patch_provenance'):
            from .source_patch import reconstruct_question_patch
            patch_evidence = chunk.get('question_patch_evidence')
            require(patch_evidence and reconstruct_question_patch(patch_evidence) == authored,
                    'Mechanical source question merge differs from its exact actual source-only patch')
            authored = checked(patch_evidence['base_authored'], authored['schema'])
        if authored.get('metadata_derivation'):
            derived_source_metadata_base_version(chunk, authored)
        if authored.get('retained_metadata_correction'):
            prior = checked(chunk.get('prior_authored'), authored['schema'])
            metadata = checked(chunk.get('metadata_only_authored'), authored['schema'])
            original = checked(chunk.get('original_authored'), authored['schema'])
            retained = authored['retained_metadata_correction']
            require(authored.get('prior_authored_sha256') == prior['sha256'] and
                    retained.get('metadata_only_payload_sha256') == metadata['sha256'] and
                    retained.get('original_authored_sha256') == original['sha256'] and
                    all(authored[key] == prior[key] for key in ('source_units', 'groups', 'acquisition_records', 'author',
                        'source_catalog_sha256', 'source_author_packet_sha256', 'events', 'authoring_inputs')),
                    'Metadata annotation cleanup changed semantic payload or lost original provenance')
            if retained.get('annotation') is not None:
                require(retained['annotation'] == metadata.get('metadata_correction'), 'Retained metadata annotation differs from original correction')
            projected = copy.deepcopy(original)
            corrected_ids = {row['unit_id']: row['opaque_probe_ids'] for row in metadata['source_units']}
            for row in projected['source_units']:
                require(row['unit_id'] in corrected_ids, 'Retained metadata correction lost original source membership')
                row['opaque_probe_ids'] = corrected_ids[row['unit_id']]
            require({k: v for k, v in projected.items() if k != 'sha256'} ==
                    {k: v for k, v in metadata.items() if k not in ('sha256', 'metadata_correction')},
                    'Retained metadata-only ancestor changed original semantic content')
        if authored.get("metadata_correction"):
            original = checked(chunk.get("original_authored"), "spacing-confirmation-authored-source-chunk-v1")
            correction = authored["metadata_correction"]
            require(correction.get("kind") == "canonical_opaque_probe_membership_only" and
                    correction.get("original_authored_sha256") == original["sha256"], "Metadata correction lost original author provenance")
            projected = copy.deepcopy(original)
            corrected_ids = {row["unit_id"]: row["opaque_probe_ids"] for row in authored["source_units"]}
            for row in projected["source_units"]:
                require(row["unit_id"] in corrected_ids, "Metadata correction lost an original source unit")
                row["opaque_probe_ids"] = corrected_ids[row["unit_id"]]
            require({k: v for k, v in projected.items() if k != "sha256"} ==
                    {k: v for k, v in authored.items() if k not in ("sha256", "metadata_correction")},
                    "Metadata correction changed semantic author content")
        authored = current_authored
        review_packet = checked(chunk["review_packet"], "spacing-source-independent-review-packet-v1")
        validate_blind_packet(review_packet)
        if 'source_preservation_protocol' in review_packet:
            require(review_packet['source_preservation_protocol'] == source_preservation_context(authored),
                    'Blind source preservation protocol differs from exact bare dispositions and fixed obligations')
        history = review_history(chunk)
        authors = author_context(chunk)
        author_ids.update(authors['channels'])
        author_ids.update(channel for _, _, context in history for channel in context['channels'])
        require(packet["source_catalog_sha256"] == authored["source_catalog_sha256"] == review_packet["source_catalog_sha256"] == catalog["sha256"] and
                authored["source_author_packet_sha256"] == review_packet["source_author_packet_sha256"] == packet["sha256"] and
                review_packet["authored_chunk_sha256"] == authored["sha256"], "Source author/review packet provenance differs")
        inputs = ['source_assertions', 'source_answer_labels', 'event_factsheets']
        if authored.get('authoring_inputs') != inputs:
            require(authored.get('authoring_inputs') == inputs + ['current_source_only_artifact', 'actual_source_review_findings'],
                    'Authored source changed its disclosed source-only authoring inputs')
            input_version = source_semantic_base_version(chunk)
            revision_row = next((row for row in input_version.get('author_task_provenance', []) if row['role'] == 'revision_author'), None)
            require(revision_row is not None, 'Expanded source repair inputs lack exact actual revision evidence')
            semantic_execution_evidence(input_version, revision_row, input_version['authored']['revision_author'])
        require(authored.get("dataset_revision") == REVISION and authored.get("evaluation_question_text_included") is False and
                authored.get("model_outcomes_used") is False and authored.get("human_review_complete") is False and
                authored.get("independent_agent_review_complete") is False and review_packet.get("author_rationales_included") is False,
                "Authored source packet changed its blinded inputs or asserted review completion")
        require(packet["events"] == authored["events"] == review_packet["events"] and
                packet["chunk_index"] == authored["chunk_index"] == review_packet["chunk_index"], "Source chunk event/index provenance differs")
        packet_units = {unit["unit_id"]: unit for unit in packet["units"]}
        require(len(packet_units) == len(packet["units"]) and set(packet_units) <= set(raw) and
                all(unit == raw[sid] for sid, unit in packet_units.items()), "Source author packet differs from raw pinned units")
        require({unit["event"] for unit in packet_units.values()} == set(packet["events"]) and
                packet["factsheets"] == {event: catalog["factsheets"][event] for event in packet["events"]} and
                review_packet["factsheets"] == packet["factsheets"], "Source author/review factsheet coverage differs")
        expected_review_units = [{key: unit[key] for key in ("unit_id", "event", "source_assertion", "canonical_answer_labels", "canonical_probes", "sha256")}
                                 for unit in packet["units"]]
        require(collections.Counter(digest(row) for row in review_packet["source_units"]) ==
                collections.Counter(digest(row) for row in expected_review_units), "Independent source review omitted original assertion/answer/probe content")
        sources = authored["source_units"]
        require(len(sources) == len(packet_units) and {row["unit_id"] for row in sources} == set(packet_units),
                "Source author lost or duplicated original assertion units")
        groups = authored["groups"]
        by_group = {group["id"]: group for group in groups}
        require(len(by_group) == len(groups), "Duplicate authored grounded group")
        membership, text_signatures = [], set()
        for group in groups:
            ids = group["source_unit_ids"]
            require(ids == sorted(set(ids)) and ids and set(ids) <= set(packet_units) and group["id"] == group_identity(ids) and
                    all(raw[sid]["event"] == group["event"] for sid in ids), "Grounded group crossed source/event membership")
            text = group["statement"]
            require(group.get("review_status") == "unreviewed" and group.get("grouping_rationale") and text.strip() and
                    "?" not in text and "Question:" not in text and "Answer:" not in text, "Authored declaration is invalid or claims an approval")
            signature = group["event"], normalize(text)
            require(signature not in text_signatures, "Identical declarations require actual reviewed grouping")
            text_signatures.add(signature)
            evidence(group["evidence"], group["event"], catalog["factsheets"])
            membership.extend(ids)
        require(collections.Counter(membership) == collections.Counter(packet_units.keys()), "Grounded groups lost or multiply counted source units")
        for source in sources:
            original = raw[source["unit_id"]]
            group = by_group.get(source["group_id"])
            require(group and source["unit_id"] in group["source_unit_ids"] and source["event"] == original["event"] and
                    source["source_assertion"] == original["source_assertion"] and source["source_unit_sha256"] == original["sha256"] and
                    sorted(source["opaque_probe_ids"]) == sorted(p["opaque_probe_id"] for p in original["canonical_probes"]) and
                    source.get("review_status") == "unreviewed", "Authored source changed original content/membership or claimed approval")
            evidence(source["evidence"], source["event"], catalog["factsheets"])
            require(source["nonliteral_answer_labels"] == [answer for answer in sorted(set(original["canonical_answer_labels"]))
                    if normalize(answer) not in normalize(group["statement"])], "Source nonliteral-label flags lost original targets")
            require(source["status"] in ("grounded", "preserve_source_conflict", "preserve_source_ambiguity"), "Unknown source conflict disposition")
            if source["status"].startswith("preserve_source"):
                require(group["source_unit_ids"] == [source["unit_id"]] and group["statement"] == source["source_assertion"] and
                        source.get("conflict_rationale"), "Preserved source conflict was merged, rewritten or unexplained")
        targets = {(group["id"], normalize(answer)) for group in groups for sid in group["source_unit_ids"]
                   for answer in raw[sid]["canonical_answer_labels"]}
        seen, questions = [], set()
        for record in authored["acquisition_records"]:
            gid, answer = record["unit_id"], normalize(record["answer"])
            require((gid, answer) in targets and record["id"] == acquisition_identity(gid, answer) and
                    record["event"] == by_group[gid]["event"] and record.get("review_status") == "unreviewed" and record.get("rationale"),
                    "Authored acquisition target changed membership or claimed approval")
            forms = record["questions"]
            require(len(forms) == 2 and len({normalize(question) for question in forms}) == 2 and all(
                    question.strip() and "\n" not in question and "Question:" not in question and "Answer:" not in question
                    for question in forms), "Source acquisition requires two valid closed-book forms")
            declarations = [by_group[gid]["statement"]] + [raw[sid]["source_assertion"] for sid in by_group[gid]["source_unit_ids"]]
            for form in forms:
                require(normalize(form) not in questions and not any(normalize(text) in normalize(form) for text in declarations),
                        "Duplicate acquisition form or embedded declaration")
                questions.add(normalize(form))
            evidence(record["evidence"], record["event"], catalog["factsheets"])
            seen.append((gid, answer))
        require(collections.Counter(seen) == collections.Counter(targets), "Acquisition source targets lost or duplicated")
        for field, authored_field in (("groups", "groups"), ("acquisition_records", "acquisition_records")):
            require(review_packet[field] == [source_review_content(row, field) for row in authored[authored_field]],
                    "Stripped independent review content differs from authored payload")
            reviewed_records(review_packet, chunk.get("reviews", []), field, authors, schema="spacing-source-semantic-review-v1", history=history,
                             artifact_versions=list(chunk.get('history', [])) + [chunk])
        all_sources.extend(sources)
        all_groups.extend(groups)
        all_qa.extend(authored["acquisition_records"])
    require(len(all_sources) == len(raw) and {source["unit_id"] for source in all_sources} == set(raw),
            "Incomplete role-independent source authoring/review coverage")
    require(len({group["id"] for group in all_groups}) == len(all_groups) and len({record["id"] for record in all_qa}) == len(all_qa),
            "Source chunks overlap or duplicate grounded/acquisition identities")
    return all_sources, all_groups, all_qa, author_ids


def validate_evaluation_chunks(chunks, catalog, probes, source_authors, acquisition_records, source_facts=None):
    records, eval_authors = [], set()
    acq_forms = {normalize(form) for record in acquisition_records for form in record["questions"]}
    source_by_id = {fact["id"]: fact for fact in source_facts or []}
    for chunk in chunks:
        packet = checked(chunk["author_packet"])
        authored = checked(chunk["authored"], "spacing-confirmation-evaluation-catalog-v1")
        review_packet = checked(chunk["review_packet"], "spacing-evaluation-independent-review-packet-v1")
        validate_blind_packet(review_packet, evaluation=True)
        history = review_history(chunk, evaluation=True)
        authors = author_context(chunk, evaluation=True)
        author = identity(authored["author"])
        require(not set(authors['channels']) & source_authors, "Evaluation and acquisition authors must use separate author channels")
        require(all(not set(context['channels']) & source_authors for _, _, context in history),
                "Historical evaluation and acquisition authors must use separate author channels")
        eval_authors.add(author)
        require(authored["source_catalog_sha256"] == review_packet["source_catalog_sha256"] == catalog["sha256"] and
                authored["evaluation_author_packet_sha256"] == packet["sha256"] and
                review_packet["evaluation_catalog_sha256"] == authored["sha256"] and review_packet["author_packet_sha256"] == packet["sha256"] and
                authored["dataset_revision"] == REVISION and authored["outer_partition_sha256"] == catalog["outer_partition_sha256"],
                "Evaluation source/author/review provenance differs")
        require(authored["authoring_inputs"] == ["canonical_evaluation_questions", "canonical_answers", "original_source_assertions", "event_factsheets"] and
                authored.get("acquisition_content_seen") is False and authored.get("model_outcomes_used") is False and
                authored.get("human_review_complete") is False and authored.get("independent_agent_review_complete") is False,
                "Evaluation author inputs or truthful review state changed")
        author_rows = {row["id"]: row for row in packet["records"]}
        require(len(author_rows) == len(packet["records"]) == len(authored["records"]), "Incomplete evaluation author packet membership")
        for row in authored["records"]:
            unit, probe, mapped = probes.get(row["opaque_probe_id"], (None, None, None))
            require(unit and row["id"] == mapped["original_probe_id"] and row["event"] == unit["event"] and
                    row["source_unit_id"] == unit["unit_id"] and row["canonical_answer"] == probe["source_answer_label"] and
                    row["aliases"] == [] and row.get("review_status") == "unreviewed", "Evaluation altered original probe/gold or added aliases/approval")
            original = author_rows.get(row["id"])
            require(original and all(original[key] == row[key] for key in ("id", "opaque_probe_id", "event", "source_unit_id", "canonical_question", "canonical_answer")),
                    "Evaluation catalog changed author packet canonical fields")
            require(row["paraphrase"].strip() and normalize(row["paraphrase"]) != normalize(row["canonical_question"]) and
                    normalize(row["paraphrase"]) not in acq_forms and normalize(row["canonical_question"]) not in acq_forms,
                    "Canonical/paraphrase evaluation wording entered acquisition or no independent paraphrase exists")
            require(row["source_ambiguity_status"] in ("supported", "retained_source_ambiguity", "retained_source_conflict"), "Unknown evaluation source-ambiguity disposition")
            require(normalize(row.get("original_source_assertion", '')) == normalize(unit["source_assertion"]) and
                    (not original.get('original_source_assertion') or row['original_source_assertion'] == original['original_source_assertion']),
                    "Evaluation source-assertion authority changed")
            evidence(row["evidence"], row["event"], catalog["factsheets"],
                     allow_empty=row["source_ambiguity_status"] in ("retained_source_ambiguity", "retained_source_conflict"))
            if source_facts is not None:
                require(row["id"] in source_by_id and row["canonical_question"] == source_by_id[row["id"]]["question"] and
                        row["canonical_answer"] == source_by_id[row["id"]]["answer"] and
                        row['original_source_assertion'] == source_by_id[row['id']].get('source_statement', source_by_id[row['id']]['statement']),
                        "Evaluation catalog differs from pinned canonical question/answer/source assertion")
        require(review_packet.get("author_rationales_included") is False and review_packet.get("acquisition_content_included") is False and
                review_packet.get("model_outcomes_included") is False and
                review_packet["records"] == [evaluation_review_content(row) for row in authored["records"]],
                "Independent evaluation review included author judgments or changed authored content")
        reviewed_records(review_packet, chunk.get("reviews", []), "records", authors, schema="spacing-evaluation-semantic-review-v1", history=history,
                         artifact_versions=list(chunk.get('history', [])) + [chunk])
        records.extend(authored["records"])
    require(len(records) == len(probes) and len({row["id"] for row in records}) == len(records) and
            {row["opaque_probe_id"] for row in records} == set(probes), "Missing or duplicated evaluation paraphrase/alias review coverage")
    return records


def cross_evidence(refs, events):
    require(refs, "Missing cross-event candidate/adjudication evidence")
    for ref in refs:
        require(ref.get("event") in events, "Cross-event evidence references an unknown source event")
        event = events[ref["event"]]
        field = ref.get("source_field")
        if field == "source_assertions":
            require(any(row["unit_id"] == ref.get("unit_id") and row["source_assertion"] == ref.get("source_assertion")
                        for row in event["source_assertions"]), "Cross-event assertion evidence differs from pinned source")
        else:
            text = event["factsheet"]["text"] if field == "factsheet" else event["factsheet_metadata"].get(field)
            require(text is not None and ref.get("source_field_sha256") == digest(text) and
                    type(ref.get("line")) is int and 0 <= ref["line"] < len(text.splitlines()) and
                    ref.get("quote") == text.splitlines()[ref["line"]], "Cross-event quotation differs from exact pinned source")


def validate_cross(cross, catalog):
    packet = checked(cross["source_packet"], "source-only-cross-entity-authoring-input-v1")
    candidates = checked(cross["entity_candidates"], "spacing-source-entity-candidates-v1")
    review = checked(cross["review"], "spacing-cross-entity-semantic-review-v1")
    identity(review.get("reviewer"))
    require(packet["source_catalog_sha256"] == catalog["sha256"] and packet["dataset_revision"] == review["dataset_revision"] == REVISION and
            packet["outer_partition_sha256"] == review["outer_partition_sha256"] == catalog["outer_partition_sha256"] and
            packet["entity_candidates_sha256"] == review["entity_candidates_sha256"] == candidates["sha256"] and
            review["source_packet_sha256"] == packet["sha256"], "Cross-event audit input provenance differs")
    require(review.get("human_review_complete") is False and review.get("model_outcomes_seen") is False and
            review.get("evaluation_questions_seen") is False and review.get("author_judgments_seen") is False and
            review.get("independent_agent_review_complete") is True and review.get("semantic_audit_complete") is True,
            "Cross-event review is incomplete or its input/review claims changed")
    event_rows = packet["events"]
    events = {row["event"]: row for row in event_rows}
    require(len(events) == len(event_rows) == 100 and set(events) == set(catalog["partition"]["confirmation"] + catalog["partition"]["development"]) and
            set(review["reviewed_event_ids"]) == set(events) and review["reviewed_events"] == 100,
            "Cross-event semantic audit requires every development and confirmation event")
    for event, row in events.items():
        require(row["factsheet"]["sha256"] == digest(row["factsheet"]["text"]), "Cross source factsheet was modified")
        if event in catalog["factsheets"]:
            require(row["factsheet"] == catalog["factsheets"][event], "Cross and confirmation factsheet inventories differ")
    listed = [row for key in ("repeated_entity_candidates", "answer_label_overlap_candidates", "exact_assertion_candidates")
              for row in candidates.get(key, [])]
    judgments = review.get("candidates", [])
    require(len(judgments) == len(listed) and len({row["candidate_id"] for row in judgments}) == len(judgments) and
            {row["candidate_id"] for row in judgments} == {row["candidate_id"] for row in listed}, "Cross-event listed candidates lack exhaustive review")
    all_judgments = judgments + review.get("extra_candidates", [])
    require(len({row["candidate_id"] for row in all_judgments}) == len(all_judgments), "Duplicated cross-event semantic candidate")
    for row in all_judgments:
        require(row.get("rationale") and row.get("disposition") not in (None, "unreviewed") and
                set(row["events"]) <= set(events), "Unreviewed cross-event candidate disposition")
        cross_evidence(row["evidence_refs"], events)
    blockers = {row["candidate_id"]: row for row in all_judgments if row.get("blocking_objection") is True}
    require(set(review.get("unresolved_same_proposition_objections", [])) <= set(blockers), "Cross-event unresolved objections omitted from candidate ledger")
    components = source_components(cross)
    confirmation_events = set(catalog["partition"]["confirmation"])
    require(all(len(component) >= 2 and set(component) <= confirmation_events for component in components),
            "A cross-partition factual component requires a partition amendment and fresh development checks")
    if blockers:
        adjudication = checked(cross.get("adjudication"))
        if adjudication.get("schema") == "spacing-cross-entity-semantic-adjudication-v1":
            validate_actual_adjudication(adjudication, packet, candidates, review, blockers, components, events)
            return components
        require(adjudication.get("schema") == "spacing-cross-event-adjudication-v1", "Unsupported cross-event protection adjudication schema")
        require(identity(adjudication.get("reviewer")) != identity(review["reviewer"]) and
                adjudication.get("source_packet_sha256") == packet["sha256"] and
                adjudication.get("entity_candidates_sha256") == candidates["sha256"] and
                adjudication.get("cross_review_sha256") == review["sha256"] and
                adjudication.get("human_review_complete") is False and adjudication.get("model_outcomes_seen") is False,
                "Independent cross-event protection adjudication is missing or refers to different inputs")
        decisions = adjudication.get("decisions", [])
        require(len(decisions) == len(blockers) and {row["candidate_id"] for row in decisions} == set(blockers),
                "Unresolved cross-event factual objections lack independent adjudication")
        for row in decisions:
            candidate = blockers[row["candidate_id"]]
            require(row.get("status") == "approved_with_same_role_protection" and
                    row.get("candidate_content_sha256") == digest(candidate) and row.get("notes") and
                    set(row["events"]) == set(candidate["events"]) and
                    any(set(row["events"]) <= set(component) for component in components) and
                    row.get("source_identity_ambiguity_retained") is True and row.get("merge_source_assertions") is False and
                    row.get("change_gold_or_evaluation") is False, "Cross-event factual objection was not actually protected and preserved")
            cross_evidence(row["evidence_refs"], events)
    elif review.get("unresolved_objections"):
        require(review.get("unresolved_identity_limitations") and not review.get("unresolved_same_proposition_objections"),
                "Unclassified cross-event objection remains blocking")
    return components


def validate_actual_adjudication(adjudication, packet, candidates, review, blockers, components, events):
    """Consume the original Opus report without changing its clearance flags."""
    require(identity(adjudication.get("reviewer")) != identity(review["reviewer"]) and adjudication.get("human_review_complete") is False and
            adjudication["dataset_revision"] == REVISION and adjudication["outer_partition_sha256"] == packet["outer_partition_sha256"],
            "Actual independent cross-event adjudication identity/source scope differs")
    bindings = adjudication["input_bindings"]
    for key, artifact in (("source_packet", packet), ("entity_candidates", candidates), ("prior_review_sol", review)):
        require(bindings[key].get("embedded_canonical_sha256") == bindings[key].get("recomputed_canonical_sha256") == artifact["sha256"] and
                bindings[key].get("embedded_matches_recomputed") is True, "Actual cross adjudication refers to different input content")
    flags = adjudication["status_flags"]
    require(flags.get("human_review_complete") is False and flags.get("clearance_granted") is False and
            flags.get("partition_cleanliness_certified") is False and flags.get("objections_fully_resolved") is False and
            flags.get("independent_review_coverage_complete") is True and not adjudication["cross_partition_blocking_links"] and
            adjudication["cross_partition_same_propositions_found"] == 0 and adjudication["scope"]["events"] == 100 and
            adjudication["scope"]["factsheets_read_in_full"] == 100 and adjudication["scope"]["source_assertions_read_in_full"] == packet["coverage"]["source_units"],
            "Actual cross adjudication is incomplete, blocking, or falsely claims source clearance")
    unseen = set(adjudication["not_seen"])
    require({"canonical_evaluation_wording", "paraphrase_evaluation_wording", "model_outputs", "acquisition_QA"} <= unseen,
            "Actual cross adjudication viewed prohibited evaluation/acquisition outcomes")
    detail = adjudication["adjudication_event_000_event_071"]
    require(set(blockers) == {detail["original_objection_id"]} and detail["events"] == ["event_000", "event_071"] and
            detail["role_contamination_analysis"].get("conservative_protection_valid") is True and
            detail["independent_semantic_determination"].get("identity_flag_preserved") is True and
            detail["objection_status_after_this_review"].get("clearance_granted") is False and
            sorted(adjudication["required_same_role_components"]) == sorted(components),
            "Actual cross adjudication does not protect every outstanding role-contamination objection")
    for row in adjudication["required_same_role_components_detail"]:
        require(row.get("merge_source_assertions") is False and row.get("change_gold_or_evaluation") is False and
                row.get("claims_identity") is False, "Actual adjudication changes gold/source or asserts unknown identity")
    refs = [ref for value in detail["exact_evidence"].values() if isinstance(value, list) for ref in value]
    cross_evidence(refs, events)
    listed = adjudication["reviewed_candidates"]
    require(len(listed) == len(review["candidates"]) and {row["candidate_id"] for row in listed} == {row["candidate_id"] for row in review["candidates"]} and
            all(row.get("blocking") is False and row.get("same_proposition") is False and row.get("rationale") for row in listed),
            "Actual second-family candidate adjudications are missing or blocking")
    for row in adjudication.get("additional_links_found", []):
        require(row.get("blocking") is False and row.get("same_proposition") is False and row.get("determination"),
                "Additional actual cross-event link lacks a reviewed nonblocking disposition")
        if row.get("evidence"):
            cross_evidence(row["evidence"], events)


def validate_confirmation_audit(audit, *, source_facts=None, pinned_factsheets=None):
    checked(audit, AUDIT_SCHEMA)
    require(audit.get("protocol_schema") == PROTOCOL and audit.get("human_review_complete") is False and
            audit.get("model_outcomes_used") is False, "Confirmation requires the explicit agent-audit protocol and truthful human/outcome status")
    def author_rows(version):
        rows = list(version.get('author_task_provenance', [])) + list(version.get('base_author_task_provenance', []))
        if version.get('question_patch_base_version'):
            rows.extend(author_rows(version['question_patch_base_version']))
        if version.get('source_ancestor_version'):
            rows.extend(author_rows(version['source_ancestor_version']))
        if version.get('metadata_derivation_base_version'):
            rows.extend(author_rows(version['metadata_derivation_base_version']))
        return rows
    provenance = [row for kind in ('source_chunks', 'evaluation_chunks') for chunk in audit.get(kind, [])
                  for version in list(chunk.get('history', [])) + [chunk] for row in author_rows(version)]
    if provenance:
        inventory = audit.get('author_task_inventory', {})
        require(inventory.get('schema') == 'p4-actual-t3-author-task-inventory-v1' and
                all(row['task_record'] in inventory.get('records', []) for row in provenance),
                'Author/revision task provenance is not retained in the actual parent task inventory')
    catalog = audit["source_catalog"]
    units, probes = validate_catalog(catalog, audit["probe_map"], source_facts, pinned_factsheets)
    sources, groups, qa, authors = validate_source_chunks(audit.get("source_chunks", []), catalog)
    evaluations = validate_evaluation_chunks(audit.get("evaluation_chunks", []), catalog, probes, authors, qa, source_facts)
    components = validate_cross(audit["cross"], catalog)
    # Keep all original ambiguity flags and objections. Protection is not a
    # finding that the underlying source identities or claims are clean.
    return {"protocol_schema": PROTOCOL, "audit_sha256": audit["sha256"], "source_catalog_sha256": catalog["sha256"],
            "outer_partition_sha256": catalog["outer_partition_sha256"], "sources": sources, "groups": groups,
            "acquisition_records": qa, "evaluation_records": evaluations, "same_role_components": components,
            "human_review_complete": False, "independent_agent_review_complete": True,
            "source_review_sha256": [review["sha256"] for chunk in audit["source_chunks"] for version in
                                     list(chunk.get('history', [])) + [chunk] for review in version.get('reviews', [])],
            "evaluation_review_sha256": [review["sha256"] for chunk in audit["evaluation_chunks"] for version in
                                         list(chunk.get('history', [])) + [chunk] for review in version.get('reviews', [])],
            "review_identities": [{"receipt_sha256": review["sha256"],
                                   "reviewer_identity_field": 'reviewer' if 'reviewer' in review else 'reviewer_identity',
                                   "reviewer": review.get('reviewer', review.get('reviewer_identity')),
                                   "review_packet_sha256": packet_binding(review),
                                   "receipt_visibility_declarations": {key: review[key] for key in
                                       ('author_rationales_seen', 'model_outcomes_seen', 'acquisition_content_seen') if key in review}}
                                  for kind in ('source_chunks', 'evaluation_chunks') for chunk in audit[kind]
                                  for version in list(chunk.get('history', [])) + [chunk] for review in version.get('reviews', [])],
            "author_task_provenance": [row for kind in ('source_chunks', 'evaluation_chunks') for chunk in audit[kind]
                                       for version in list(chunk.get('history', [])) + [chunk]
                                       for row in author_rows(version)],
            "author_identities": [{"authored_sha256": version['authored']['sha256'], "author": version['authored']['author'],
                                   "revision_author": version['authored'].get('revision_author')}
                                  for kind in ('source_chunks', 'evaluation_chunks') for chunk in audit[kind]
                                  for version in list(chunk.get('history', [])) + [chunk]],
            "source_review_limitations": [{"receipt_sha256": review['sha256'], "field": field, "content": review[field]}
                                          for chunk in audit['source_chunks'] for version in list(chunk.get('history', [])) + [chunk]
                                          for review in version.get('reviews', [])
                                          for field in ('source_unit_issues', 'source_units', 'missing_original_answer_labels', 'retained_source_ambiguities')
                                          if review.get(field)],
            "cross_review_sha256": audit["cross"]["review"]["sha256"],
            "cross_adjudication_sha256": audit["cross"].get("adjudication", {}).get("sha256"),
            "identity_limitations": audit["cross"]["review"].get("unresolved_identity_limitations", [])}


def validate_roles(context, roles, partition_hash):
    require(context["outer_partition_sha256"] == partition_hash and set(roles) == {
        row["event"] for row in context["sources"]} and collections.Counter(roles.values()) == {
            "old": 20, "new": 30, "control": 20, "qa": 10}, "Confirmation audit role/partition geometry differs")
    require(all(len({roles[event] for event in component}) == 1 for component in context["same_role_components"]),
            "Reviewed factual component crossed confirmation event roles")


def validate_confirmation_source_bundle(bundle, audit, facts, source_registry, partition_hash, roles):
    context = validate_confirmation_audit(audit)
    validate_roles(context, roles, partition_hash)
    require(bundle.get("confirmation_audit_sha256") == audit["sha256"] and bundle.get("source_catalog_sha256") ==
            context["source_catalog_sha256"] and bundle.get("human_review_complete") is False, "Derived grounding audit/source provenance differs")
    selected = {row["unit_id"]: row for row in context["sources"] if roles[row["event"]] in ("old", "new")}
    groups = {row["id"]: row for row in context["groups"] if roles[row["event"]] in ("old", "new")}
    require({row["unit_id"] for row in bundle["source_units"]} == set(selected) and {row["id"] for row in bundle["groups"]} == set(groups),
            "Derived grounding catalog membership differs from actual reviewed source")
    for row in bundle["groups"]:
        original = groups[row["id"]]
        require(all(row[key] == original[key] for key in ("id", "event", "source_unit_ids", "statement", "grouping_rationale", "evidence")) and
                row["review_status"] == "approved_confirmation", "Derived grounding differs from actually approved content")
    for row in bundle["source_units"]:
        original = selected[row["unit_id"]]
        require(row["source_statement"] == original["source_assertion"] and row["event_id"] == original["event"] and
                all(row[key] == original[key] for key in ("group_id", "status", "conflict_rationale", "nonliteral_answer_labels", "evidence")) and
                row["review_status"] == "approved_confirmation", "Derived source flags/evidence differ from reviewed author payload")
    return context


def validate_confirmation_acquisition_bundle(bundle, source_input, audit, roles):
    context = validate_confirmation_audit(audit)
    require(bundle.get("confirmation_audit_sha256") == source_input.get("confirmation_audit_sha256") == audit["sha256"] and
            bundle.get("source_catalog_sha256") == source_input.get("source_catalog_sha256") == context["source_catalog_sha256"] and
            bundle.get("human_review_complete") is False, "Derived acquisition catalog/audit binding differs")
    expected = {row["id"]: row for row in context["acquisition_records"] if roles[row["event"]] == "old"}
    require(len(bundle["records"]) == len(expected) and {row["id"] for row in bundle["records"]} == set(expected),
            "Derived old acquisition catalog lost or added reviewed targets")
    for row in bundle["records"]:
        original = expected[row["id"]]
        require(all(row[key] == original[key] for key in ("id", "unit_id", "event", "answer", "questions", "evidence", "rationale", "source_conflict_status")) and
                row["review_status"] == "approved_confirmation", "Derived acquisition differs from actually approved source-QA content")
    return context


NUMERICAL_SCOPE = {
    "training.py": ("capture_rng", "restore_rng", "seed_all", "load_model", "make_optimizer", "save_checkpoint",
                    "load_checkpoint", "train_update", "recover_exposure_log", "Session.__init__", "Session.resume",
                    "Session.update", "Session.audit_exposures", "Session.evaluate"),
    "aa_check.py": ("compare_tree", "run_aa"),
    "evaluation.py": ("conditional_scores", "generated_answers"),
}
AA_CONFIG_KEYS = ('batch_size', 'microbatch_size', 'context_length', 'loss_tokens_per_example',
                  'learning_rate', 'warmup_steps', 'eval_context_length', 'eval_batch_size',
                  'generation_batch_size', 'max_answer_tokens', 'generic_eval_sequences',
                  'qa_per_step', 'review_cap', 'cohort_size')


def numerical_sources(text, names):
    tree = ast.parse(text)
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[node.name] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    definitions[node.name + "." + child.name] = ast.get_source_segment(text, child)
    require(set(names) <= definitions.keys(), "Numerical compatibility scope lost a required implementation")
    return {name: digest(definitions[name]) for name in names}


def current_numerical_hashes():
    root = Path(__file__).resolve().parent
    return {filename: numerical_sources((root / filename).read_text(), names) for filename, names in NUMERICAL_SCOPE.items()}


def build_aa_compatibility(aa_report, launch_commit, repository, aa_manifest_path=None):
    """Compare actual git versions; this creates no independent review claim."""
    require(aa_report.get("schema") == "spacing-aa-replay-v1" and aa_report.get("passed") is True,
            "Code compatibility requires an actual passing A/A report")
    commits = (aa_report["code_commit"], launch_commit)
    require(all(len(commit) == 40 and all(char in "0123456789abcdef" for char in commit) for commit in commits),
            "A/A compatibility requires actual full code commits")
    versions = []
    for commit in commits:
        version = {}
        for filename, names in NUMERICAL_SCOPE.items():
            path = "experiments/spacing_rerun/spacing_rerun/" + filename
            text = subprocess.check_output(["git", "show", commit + ":" + path], cwd=repository, text=True)
            version[filename] = numerical_sources(text, names)
        versions.append(version)
    require(versions[0] == versions[1], "Relevant numerical/A-A implementations changed; repeat the actual GPU A/A check")
    reference = Path(aa_manifest_path or Path(aa_report["prepared"]) / "manifest.json")
    reference_bytes = reference.read_bytes()
    manifest = read_json(reference)
    require(hashlib.sha256(reference_bytes).hexdigest() == aa_report["bound_prepared_artifacts"]["manifest.json"]["sha256"] and
            manifest["sha256"] == aa_report["manifest_sha256"], "A/A reference manifest differs from actual checked preparation")
    backend = {"model": manifest["model"], "model_revision": manifest["model_revision"],
               "tokenizer_sha256": manifest["tokenizer_sha256"], "software": aa_report["software"],
               "device": aa_report["device"], "gpu": aa_report["gpu"],
               "load_backend": "hf_olmo_transformers_fp32_parameters_bf16_cuda_autocast_deterministic_no_tf32"}
    packet = {"schema": "spacing-aa-code-compatibility-v1", "aa_report_sha256": digest(aa_report),
              "aa_code_commit": commits[0], "launch_code_commit": commits[1], "method": "actual_git_function_source_identity",
              "numerical_scope": {key: list(value) for key, value in NUMERICAL_SCOPE.items()}, "function_sha256": versions[0],
              "reference_manifest_path": str(reference.resolve()), "reference_manifest_sha256": manifest["sha256"],
              "reference_manifest_file_sha256": hashlib.sha256(reference_bytes).hexdigest(), "backend": backend,
              'reference_numerical_config': {key: manifest['config'][key] for key in AA_CONFIG_KEYS}}
    packet["sha256"] = digest(packet)
    return packet


def verify_confirmation_preregistration(prereg, manifest, *, check_code=True):
    from .acquisition import SOURCE_QA_ACQUISITION_POLICY
    from .teaching import SOURCE_QA_POLICY
    from .units import GROUNDED_POLICY, GROUNDED_METRICS
    from .final_span import verify_final_span_report
    checked(prereg, "spacing-confirmation-preregistration-v1")
    require(manifest.get("confirmation_protocol_schema") == prereg.get("confirmation_protocol_schema") == PROTOCOL and
            manifest.get("audit_complete") is True and manifest.get("rehearsal_unit_policy") == prereg.get("rehearsal_unit_policy") == GROUNDED_POLICY and
            prereg.get("metric_schema") == GROUNDED_METRICS and prereg.get("acquisition_policy") == SOURCE_QA_ACQUISITION_POLICY and
            prereg.get("qa_teaching_policy") == SOURCE_QA_POLICY, "Confirmation protocol/recipe/metric preregistration differs")
    context = validate_confirmation_audit(manifest["confirmation_audit"], source_facts=manifest["split"]["facts"])
    require(prereg.get("confirmation_audit_sha256") == context["audit_sha256"] and
            prereg.get("source_catalog_sha256") == context["source_catalog_sha256"] and prereg.get("human_review_complete") is False,
            "Preregistration does not bind actual complete source/evaluation/cross audits")
    frozen = datetime.datetime.fromisoformat(prereg["frozen_utc"])
    require(frozen.tzinfo is not None and frozen <= datetime.datetime.now(datetime.timezone.utc), "Confirmation freeze timestamp is absent, naive or in the future")
    if manifest.get("created_utc"):
        created = datetime.datetime.fromisoformat(manifest["created_utc"])
        require(created.tzinfo is not None and created <= frozen, "Confirmation manifest was prepared after preregistration")
    if prereg.get('final_span_preflight_path') is not None:
        require(prereg.get('final_span_preflight_sha256'), 'Verified final-span preflight must be frozen by its actual receipt digest')
        report = verify_final_span_report(prereg['final_span_report_path'],
            preflight_path=prereg['final_span_preflight_path'],
            expected_preflight_sha256=prereg['final_span_preflight_sha256'])
    else:
        require(not prereg.get('final_span_preflight_sha256'), 'Final-span preflight hash has no actual receipt path')
        report = verify_final_span_report(prereg["final_span_report_path"])
    require(prereg["final_span_report_sha256"] == report["sha256"] and report["assay_usable"] is True and
            report["cost"].get("accounting_complete") is True, "Final-span sensitivity or actual allocation accounting remains incomplete")
    require(prereg["E"] == report["selected_E"] and prereg["n"] == report["precision_plan"]["chosen_n"] and
            prereg["inference_claim"] == report["precision_plan"]["claim"] and prereg["planning_policy_sha256"] == report["planning_policy_sha256"] and
            prereg["power_scenarios"] == report["precision_plan"], "Confirmation changed common acquisition dose, finite-t sample size or planning claim")
    require(type(prereg["n"]) is int and type(prereg["E"]) is int and 2 <= prereg["n"] <= 32 and
            prereg["E"] in (2, 3, 4, 6, 8) and prereg["loss_margin"] == .02 and prereg["primary_delay"] == 84 and
            prereg["common_span"] == 252 and prereg["buffer_steps"] == 168 and prereg["cohort_size"] == 2 and prereg["review_cap"] == 8 and
            prereg["secondary_holm_family"] == ["H1", "H3", "HG"] and prereg.get("missing_pair_rule") and
            prereg["maximum_attempts"] == manifest["config"]["max_attempts"], "Confirmation estimand, multiplicity, geometry or attempt policy changed")
    require(prereg.get("event_cluster_sensitivity_policy") == "same_role_components_as_single_cluster_then_equal_clusters_v1" and
            prereg.get("same_role_components") == context["same_role_components"], "Protected-event cluster sensitivity is not preregistered")
    require(prereg["model_revision"] == manifest["model_revision"] == manifest["config"]["model_revision"] and
            prereg["tokenizer_sha256"] == manifest["tokenizer_sha256"] and prereg["measured_cost"]["final_span_report_sha256"] == report["sha256"] and
            prereg["measured_cost"].get("confirmation_projection") and prereg["measured_cost"].get("content_scale_limitations"),
            "Confirmation model/tokenizer or measured forecast/scale limitations are not frozen")
    hashes = prereg["replicate_manifest_sha256"]
    designs = prereg["replicate_designs"]
    require(len(hashes) == len(set(hashes)) == len(designs) == prereg["n"] and
            {row["manifest_sha256"] for row in designs} == set(hashes) and manifest["sha256"] in hashes,
            "Missing, duplicate or unregistered independent replicate manifests")
    own = next(row for row in designs if row["manifest_sha256"] == manifest["sha256"])
    require(all(own[key] == manifest["config"][key] for key in ("split_seed", "order_seed", "train_seed", "eval_seed")) and
            own["roles_sha256"] == digest(manifest["split"]["roles"]) and
            all(len({row[key] for row in designs}) == prereg['n'] and all(type(row[key]) is int and row[key] > 0 for row in designs)
                for key in ('split_seed', 'order_seed', 'train_seed', 'eval_seed')) and
            all(2026100801 <= row["split_seed"] <= 2026100832 for row in designs), "Confirmation role/seed manifest bindings differ")
    aa = read_json(prereg["aa_report_path"])
    require(digest(aa) == prereg["aa_report_sha256"] and aa.get("schema") == "spacing-aa-replay-v1" and
            aa.get("passed") is True and aa.get("status") == "passed" and aa.get("device") == "cuda" and
            aa.get("source_artifacts_unchanged") is True and aa.get("gpu") == prereg["hardware"]["gpu"],
            "Actual passing GPU A/A/backend evidence is missing or modified")
    comparisons = aa.get("comparisons", {})
    require(set(comparisons) == {"model", "optimizer", "rng", "progress_and_clocks", "training_metrics", "probe_observations", "realized_exposures"} and
            all(value.get("bitwise_equal") is True and value.get("mismatch_count") == 0 for value in comparisons.values()),
            "Actual A/A comparisons or exposure/RNG/numerical checks are incomplete")
    compatibility = checked(prereg["aa_code_compatibility"], "spacing-aa-code-compatibility-v1")
    require(compatibility.get("aa_report_sha256") == digest(aa) and compatibility.get("aa_code_commit") == aa["code_commit"] and
            compatibility.get("launch_code_commit") == prereg["code_commit"] and compatibility.get("method") == "actual_git_function_source_identity" and
            compatibility.get("numerical_scope") == {key: list(value) for key, value in NUMERICAL_SCOPE.items()} and
            compatibility.get("function_sha256") == current_numerical_hashes(), "Actual A/A does not cover the current numerical implementation")
    reference_path = Path(compatibility["reference_manifest_path"])
    reference = read_json(reference_path)
    require(hashlib.sha256(reference_path.read_bytes()).hexdigest() == compatibility["reference_manifest_file_sha256"] ==
            aa["bound_prepared_artifacts"]["manifest.json"]["sha256"] and
            reference["sha256"] == compatibility["reference_manifest_sha256"] == aa["manifest_sha256"],
            "A/A reference preparation changed")
    backend = compatibility["backend"]
    require(compatibility.get('reference_numerical_config') == {key: reference['config'][key] for key in AA_CONFIG_KEYS} ==
            {key: manifest['config'][key] for key in AA_CONFIG_KEYS} and manifest['config']['microbatch_size'] == 2,
            'Actual A/A training/evaluation configuration differs from the frozen confirmation recipe')
    require(backend == {"model": reference["model"], "model_revision": reference["model_revision"],
            "tokenizer_sha256": reference["tokenizer_sha256"], "software": aa["software"], "device": aa["device"], "gpu": aa["gpu"],
            "load_backend": "hf_olmo_transformers_fp32_parameters_bf16_cuda_autocast_deterministic_no_tf32"} and
            all(backend[key] == manifest[key] for key in ("model", "model_revision", "tokenizer_sha256")) and
            all(importlib.metadata.version(name) == version for name, version in backend["software"].items()) and
            set(backend["software"]) == {"torch", "transformers", "ai2-olmo", "numpy"},
            "Actual A/A model/tokenizer/software/backend differs from confirmation launch")
    if check_code:
        commit = os.environ.get("SPACING_CODE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        require(prereg["code_commit"] == commit, "Confirmation launch code differs from preregistration")
    return prereg
