"""Strict mechanical application of actual source-only question-author patches."""
from __future__ import annotations

import copy
import hashlib
import json
from urllib.parse import unquote

from .common import digest, require

PATCH_INPUT = 'source-only-confirmation-targeted-qa-revision-input-v1'
PATCH_EVIDENCE = 'spacing-source-question-patch-evidence-v1'
PATCH_PROVENANCE = 'spacing-mechanical-source-question-patch-v1'
ANNOTATIONS = ('metadata_correction', 'retained_metadata_correction', 'metadata_derivation')


def sealed(value):
    require(isinstance(value, dict) and value.get('sha256') == digest({key: item for key, item in value.items() if key != 'sha256'}),
            'Source question patch artifact is missing or has changed its digest')
    return value


def actual_bytes(value, file_info):
    require(isinstance(file_info.get('utf8'), str) and json.loads(file_info['utf8']) == value and
            hashlib.sha256(file_info['utf8'].encode()).hexdigest() == file_info.get('file_sha256') and
            file_info.get('canonical_sha256') == value.get('sha256'),
            'Source question patch raw bytes differ from the actual artifact')
    return file_info['file_sha256']


def compatible_aliases(value, names, expected, *, required=False):
    present = [value[name] for name in names if name in value]
    require((present or not required) and all(item == expected for item in present),
            'Source question patch declared provenance differs: ' + '/'.join(names))


def normalize_task(status, request):
    require(status.get('status') == 'completed' and status.get('childRunId') and status.get('childThreadId') and
            status.get('taskId') and unquote(status['taskId']).endswith(':delegate-task:' + request) and
            status.get('model') and status.get('summary'), 'Source question patch lacks its actual completed author task')
    return {'task_id': status['taskId'], 'client_request_id': request, 'child_thread_id': status['childThreadId'],
            'child_run_id': status['childRunId'], 'status': status['status'], 'model': status['model'],
            'provider_instance_id': status.get('providerInstanceId'), 'actual_task_status': status}


def reconstruct_question_patch(evidence):
    sealed(evidence)
    require(evidence.get('schema') == PATCH_EVIDENCE, 'Unsupported source question patch proof')
    base, packet, patch, response = [sealed(evidence[key]) for key in ('base_authored', 'input_packet', 'patch', 'response')]
    files = evidence['artifact_files']
    hashes = {key: actual_bytes(evidence[key], files[key]) for key in ('base_authored', 'input_packet', 'patch', 'response')}
    task_inventory = evidence['actual_task_status_inventory']
    actual_bytes(task_inventory, files['actual_task_status_inventory'])
    actual_status = evidence['actual_task_status']
    require(actual_status in task_inventory.get('records', []), 'Source question patch task is absent from the actual task-status inventory')
    author = patch['revision_author']
    task = normalize_task(actual_status, author['task_id'])
    require(task == evidence['task_record'] and author['model'] == task['model'] and patch['sha256'] in actual_status['summary'],
            'Source question patch identity/result differs from actual task completion')
    require(base['schema'] == 'spacing-confirmation-authored-source-chunk-v1' and packet['schema'] == patch['schema'] == PATCH_INPUT and
            packet['current_authored_sha256'] == patch['current_authored_sha256'] == patch['prior_authored_sha256'] == base['sha256'],
            'Source question patch is not bound to the exact immutable base')
    for key in ('source_catalog_sha256', 'source_author_packet_sha256', 'dataset_revision', 'chunk_index', 'events', 'groups', 'source_units'):
        require(base[key] == packet[key] == patch[key], 'Source question patch changed source context: ' + key)
    require(packet.get('evaluation_question_text_included') is False and packet.get('model_outcomes_included') is False and
            packet.get('other_reviewer_judgments_included') is False, 'Source question patch input violates channel isolation')
    require(all(patch[key] == value for key, value in packet.items() if key not in ('sha256', 'selected_acquisition_records')),
            'Source question patch altered its supplied non-question input')
    compatible_aliases(patch, ('source_revision_packet_sha256',), packet['sha256'])
    compatible_aliases(patch, ('source_revision_packet_file_sha256', 'source_revision_packet_bytes_sha256'), hashes['input_packet'])
    compatible_aliases(response, ('source_revision_packet_sha256', 'input_packet_sha256'), packet['sha256'], required=True)
    compatible_aliases(response, ('source_revision_packet_file_sha256', 'source_revision_packet_bytes_sha256', 'input_file_bytes_sha256'),
                       hashes['input_packet'], required=True)
    compatible_aliases(response, ('authored_sha256', 'artifact_sha256', 'output_authored_sha256'), patch['sha256'], required=True)
    compatible_aliases(response, ('authored_file_sha256', 'artifact_file_sha256', 'authored_file_bytes_sha256'), hashes['patch'])
    for key in ('source_catalog_sha256', 'source_author_packet_sha256'):
        compatible_aliases(response, (key,), base[key])
    require(response['prior_authored_sha256'] == base['sha256'] and response['revision_author'] == author,
            'Source question correction response changed its actual prior artifact/author')
    if 'input_path' in response:
        require(response['input_path'] == files['input_packet']['path'], 'Source question correction response input path differs')
    if 'output_path' in response:
        require(response['output_path'] == files['patch']['path'], 'Source question correction response output path differs')
    original = {row['id']: row for row in base['acquisition_records']}
    selected = {row['id']: row for row in packet['selected_acquisition_records']}
    changed = {row['id']: row for row in patch['selected_acquisition_records']}
    require(len(original) == len(base['acquisition_records']) and len(selected) == len(packet['selected_acquisition_records']) and
            len(changed) == len(patch['selected_acquisition_records']) and set(changed) == set(selected) <= set(original),
            'Source question patch lost, duplicated or added selected/base records')
    result = copy.deepcopy(base)
    rows = {row['id']: row for row in result['acquisition_records']}
    changes = []
    for rid, before in selected.items():
        after = changed[rid]
        require({key: value for key, value in before.items() if key != 'question_indices_to_revise'} == original[rid] and
                {key: value for key, value in after.items() if key != 'questions'} ==
                {key: value for key, value in before.items() if key != 'questions'},
                'Source question patch altered record metadata or selected an inconsistent base')
        indices = before['question_indices_to_revise']
        require(indices == sorted(set(indices)) and indices and set(indices) <= {0, 1} and
                len(after['questions']) == len(before['questions']) == 2,
                'Source question patch has invalid question-index allowlist')
        for index, form in enumerate(after['questions']):
            require(isinstance(form, str) and form.strip() and '\n' not in form and
                    (index in indices or form == before['questions'][index]),
                    'Source question patch altered an untargeted question or supplied invalid wording')
            if form != before['questions'][index]:
                rows[rid]['questions'][index] = form
                changes.append({'id': rid, 'question_index': index, 'before': before['questions'][index], 'after': form})
    require(changes, 'Source question patch made no authorized question changes')
    retained = {key: result.pop(key) for key in ANNOTATIONS if key in result}
    if retained:
        result.setdefault('retained_historical_annotations', []).append({'applies_to_authored_sha256': base['sha256'], 'annotations': retained})
    result['prior_authored_sha256'] = base['sha256']
    result['revision_author'] = copy.deepcopy(author)
    result['question_patch_provenance'] = {'schema': PATCH_PROVENANCE, 'evidence_sha256': evidence['sha256'],
        'base_authored_sha256': base['sha256'], 'input_packet_sha256': packet['sha256'], 'patch_sha256': patch['sha256'],
        'correction_response_sha256': response['sha256'], 'actual_task_id': task['task_id'], 'changes': changes,
        'no_semantic_review_decisions_created': True}
    result['sha256'] = digest({key: value for key, value in result.items() if key != 'sha256'})
    return result
