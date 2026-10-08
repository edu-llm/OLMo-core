"""Explicit synthetic source-patch proofs; never production author/review evidence."""
import copy
import hashlib
import json
import shlex
import unittest

from spacing_rerun.source_patch import (AUTHOR_RATIONALE_FIELDS, SOURCE_BLIND_METADATA_FIELDS, PATCH_EVIDENCE, PATCH_INPUT,
    normalize_task, reconstruct_question_patch, source_only_projection)
from spacing_rerun.confirmation import (source_review_content, validate_confirmation_audit, validate_source_chunks,
                                        review_history, author_context)
from spacing_rerun.common import digest
from test_confirmation import confirmation_fixture, seal


def native_trace_fixture(status, artifact, info):
    """Synthetic isolated commands whose stdout is retained separately from source."""
    row={'path':info['path'],'sha256':artifact['sha256'],'file_sha256':info['file_sha256']}
    creation={**row,'mode':'0444','canonical_verified':True}
    verification={'artifacts':[{**row,'canonical_seal_independently_verified':True,
        'raw_written_bytes_independently_verified':True,'readonly_mode':'0444','chmod_444_after_final_write_verified':True}],
        'independent_verification':{'verification_performed_no_writes':True}}
    prefix="import hashlib, json, stat\nfrom pathlib import Path\npatch_path=Path("+repr(info['path'])+")\n"
    check=("raw=patch_path.read_bytes();obj=json.loads(raw)\n"
        "canonical=hashlib.sha256(json.dumps({k:v for k,v in obj.items() if k!='sha256'},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()\n"
        "assert canonical==obj['sha256']=="+repr(artifact['sha256'])+"\n"
        "assert hashlib.sha256(raw).hexdigest()=="+repr(info['file_sha256'])+"\n"
        "assert stat.S_IMODE(patch_path.stat().st_mode)==0o444\n")
    source=(prefix+"import os\nraw="+repr(info['utf8'])+".encode()\n"
        "with patch_path.open('xb') as stream:\n    stream.write(raw);stream.flush();os.fsync(stream.fileno())\n"
        "os.chmod(patch_path,0o444)\n"+check+"print(json.dumps("+repr(creation)+",indent=2))")
    verify=prefix+check+"print(json.dumps("+repr(verification)+",indent=2))"
    def item(position,source,report):
        command="python -I -S -B - <<'PY'\n"+source+"\nPY"
        return {'position':position,'sourceThreadId':status['childThreadId'],'runId':status['childRunId'],
            'type':'command_execution','status':'completed','visibility':'local','textTruncated':False,'nextTextOffset':None,
            'text':'$ /bin/zsh -c '+shlex.quote(command)+'\n'+json.dumps(report,indent=2)+'\n'}
    return {'thread':{'threadId':status['childThreadId'],'status':'completed','latestRunId':status['childRunId'],
        'activeRunId':None,'pendingRequestCount':0,'model':status['model']},
        'recentRuns':[{'runId':status['childRunId'],'status':'completed','model':status['model']}],
        'items':[item(12,source,creation),item(16,verify,verification)]}


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
              'childThreadId':'synthetic-thread','model':author['model'],'summary':patch['sha256'],'test_fixture':True,
              'latestTerminalStatus':'completed','hasPendingChildRuns':False,'latestTerminalRunId':'synthetic-run',
              'latestTerminalSummary':patch['sha256']}
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
    evidence['actual_task_status']['latestTerminalSummary'] = patch['sha256']
    evidence['task_record'] = normalize_task(evidence['actual_task_status'],patch['revision_author']['task_id'])
    evidence['artifact_files'] = {key:raw(evidence[key],key) for key in
        ('base_authored','input_packet','patch','response','actual_task_status_inventory')}
    seal(evidence)


class SourcePatchTests(unittest.TestCase):
    def test_separate_original_contract_helper_custody_is_bound_without_restamping_proof(self):
        evidence=proof_fixture();info=evidence['artifact_files']['patch'];status=evidence['actual_task_status']
        helper_text='Synthetic retained helper bytes; not executed.\n';helper_path='/tmp/synthetic-original-helper.py'
        contract=seal({'schema':'synthetic-original-author-contract','patch_output_path':info['path'],
            'authorized_offline_packing_script_path':helper_path,
            'authorized_offline_packing_script_file_sha256':hashlib.sha256(helper_text.encode()).hexdigest()})
        contract_text=json.dumps(contract);contract_path='/tmp/synthetic-original-contract.json'
        declared={'path':contract_path,'sha256':contract['sha256'],'file_sha256':hashlib.sha256(contract_text.encode()).hexdigest()}
        evidence['response']['author_output_contract']=declared;refresh(evidence);info=evidence['artifact_files']['patch']
        status['latestTerminalSummary']='Completed original patch '+info['path'];seal(evidence['actual_task_status_inventory'])
        text=json.dumps(evidence['actual_task_status_inventory'],sort_keys=True,indent=2)
        evidence['artifact_files']['actual_task_status_inventory'].update(utf8=text,file_sha256=hashlib.sha256(text.encode()).hexdigest(),canonical_sha256=evidence['actual_task_status_inventory']['sha256'])
        evidence['task_record']=normalize_task(status,evidence['patch']['revision_author']['task_id'])
        trace=native_trace_fixture(status,evidence['patch'],info)
        item=trace['items'][0];boundary=item['text'].rindex('\n{');command=shlex.split(item['text'][:boundary].removeprefix('$ '))[2]
        replacement=("contract_path=Path("+repr(contract_path)+")\ncontract=json.loads(contract_path.read_bytes())\n"
            "helper=Path(contract['authorized_offline_packing_script_path'])\n"
            "assert hashlib.sha256(helper.read_bytes()).hexdigest()==contract['authorized_offline_packing_script_file_sha256']\n"
            "patch_path=Path(contract['patch_output_path'])")
        item['command']=command.replace('patch_path=Path('+repr(info['path'])+')',replacement);item['stdout']=item['text'][boundary+1:];item.pop('text')
        text=json.dumps({'isError':False,'structuredContent':trace,'content':[{'type':'text','text':json.dumps(trace)}]})
        evidence['completion_trace_file']={'utf8':text,'file_sha256':hashlib.sha256(text.encode()).hexdigest(),'trace_json_pointer':'/structuredContent'};seal(evidence)
        inputs={'author_output_contract':{'path':contract_path,'utf8':contract_text,'file_sha256':declared['file_sha256'],'canonical_sha256':declared['sha256']},
            'packing_helper':{'path':helper_path,'utf8':helper_text,'file_sha256':contract['authorized_offline_packing_script_file_sha256']}}
        immutable=copy.deepcopy(evidence);reconstruct_question_patch(evidence,original_trace_input_files=inputs);self.assertEqual(evidence,immutable)
        for kind in ('no-inputs','missing-contract','missing-helper','wrong-contract-path','wrong-contract-raw','wrong-contract-seal',
                     'wrong-helper-path','wrong-helper-raw','helper-canonical-alias','unrecognized-input','extra-contract-descriptor',
                     'extra-helper-descriptor','wrong-ancestor-contract'):
            with self.subTest(kind=kind):
                changed=copy.deepcopy(inputs)
                if kind=='missing-contract':changed.pop('author_output_contract')
                elif kind=='missing-helper':changed.pop('packing_helper')
                elif kind=='wrong-contract-path':changed['author_output_contract']['path']='/tmp/another-contract.json'
                elif kind=='wrong-contract-raw':changed['author_output_contract']['utf8']+=' '
                elif kind=='wrong-contract-seal':changed['author_output_contract']['canonical_sha256']='0'*64
                elif kind=='wrong-helper-path':changed['packing_helper']['path']='/tmp/another-helper.py'
                elif kind=='wrong-helper-raw':changed['packing_helper']['utf8']+=' '
                elif kind=='helper-canonical-alias':changed['packing_helper']['sha256']=changed['packing_helper']['file_sha256']
                elif kind=='unrecognized-input':changed['parent_approval']={'approved':True}
                elif kind=='extra-contract-descriptor':changed['author_output_contract']['parent_approval']=True
                elif kind=='extra-helper-descriptor':changed['packing_helper']['parent_approval']=True
                elif kind=='wrong-ancestor-contract':
                    other=copy.deepcopy(contract);other['patch_output_path']='/tmp/another-ancestor.json';seal(other)
                    changed['author_output_contract']['utf8']=json.dumps(other)
                    changed['author_output_contract']['file_sha256']=hashlib.sha256(changed['author_output_contract']['utf8'].encode()).hexdigest()
                    changed['author_output_contract']['canonical_sha256']=other['sha256']
                with self.assertRaises(ValueError):reconstruct_question_patch(evidence,original_trace_input_files=None if kind=='no-inputs' else changed)
                self.assertEqual(evidence,immutable)
        # Exact existing custody must also reach historical author validation,
        # source validation and its audit caller, without adding proof fields.
        facts,audit=confirmation_fixture();current=copy.deepcopy(audit['source_chunks'][0]);historical=copy.deepcopy(current)
        merged=reconstruct_question_patch(evidence,original_trace_input_files=inputs)
        row={'role':'revision_author','identity':merged['revision_author'],'authored_sha256':merged['sha256'],
            'author_packet_sha256':merged['source_author_packet_sha256'],'input_channel':'source_only',
            'task_id':evidence['task_record']['task_id'],'task_record':evidence['task_record']}
        historical.update(authored=merged,question_patch_evidence=evidence,author_task_provenance=[row],reviews=[])
        packet=historical['review_packet'];packet['authored_chunk_sha256']=merged['sha256']
        packet['acquisition_records']=[source_review_content(record,'acquisition_records') for record in merged['acquisition_records']];seal(packet)
        current['history']=[historical];mapping={evidence['sha256']:inputs}
        author_context(historical,original_trace_input_files_by_proof=mapping)
        review_history(current,original_trace_input_files_by_proof=mapping)
        validate_source_chunks([current],audit['source_catalog'],original_trace_input_files_by_proof=mapping)
        audit['source_chunks']=[current];audit['author_task_inventory']={'schema':'p4-actual-t3-author-task-inventory-v1',
            'records':[evidence['task_record']],'test_fixture':True};seal(audit)
        validate_confirmation_audit(audit,source_facts=facts,original_trace_input_files_by_proof=mapping)
        for validate in (lambda:review_history(current),lambda:validate_source_chunks([current],audit['source_catalog']),
                         lambda:validate_confirmation_audit(audit,source_facts=facts)):
            with self.assertRaisesRegex(ValueError,'custody'):validate()
        self.assertEqual(evidence,immutable)

    def test_supported_execution_values_preserve_branches_types_membership_and_hashes(self):
        evidence=proof_fixture();status=evidence['actual_task_status'];info=evidence['artifact_files']['patch']
        status['latestTerminalSummary']='Completed original patch '+info['path']
        inventory=evidence['actual_task_status_inventory'];seal(inventory)
        text=json.dumps(inventory,sort_keys=True,indent=2)
        evidence['artifact_files']['actual_task_status_inventory'].update(utf8=text,file_sha256=hashlib.sha256(text.encode()).hexdigest(),canonical_sha256=inventory['sha256'])
        evidence['task_record']=normalize_task(status,evidence['patch']['revision_author']['task_id'])
        trace=native_trace_fixture(status,evidence['patch'],info)
        for item in trace['items']:
            boundary=item['text'].rindex('\n{');item['command']=shlex.split(item['text'][:boundary].removeprefix('$ '))[2]
            item['stdout']=item['text'][boundary+1:];item.pop('text')
        def source(item):return item['command'].split('\n',1)[1].rsplit('\nPY',1)[0]
        def bind(value):
            text=json.dumps({'isError':False,'structuredContent':value,'content':[{'type':'text','text':json.dumps(value)}]})
            evidence['completion_trace_file']={'utf8':text,'file_sha256':hashlib.sha256(text.encode()).hexdigest(),'trace_json_pointer':'/structuredContent'};seal(evidence)
        bind(trace);baseline=reconstruct_question_patch(evidence)
        cases=[
            ('builtin-equality','', 'str==str',True),
            ('builtin-inequality','', 'str!=str',False),
            ('function-equality','', 'json.dumps==json.dumps',True),
            ('function-inequality','', 'json.dumps!=json.dumps',False),
            ('builtin-alias','first=str\nsecond=str\n','first==second',True),
            ('function-alias','import json as serialization\nfirst=serialization.dumps\n','first==json.dumps',True),
            ('module-alias','import json as serialization\n','serialization==json',True),
            ('distinct-functions','','json.dumps!=json.loads',True),
            ('distinct-types','','str!=set',True),
            ('membership','','str in [str]',True),
            ('membership-wrong','','str not in [str]',False),
            ('set-cardinality','','len(set([str,str]))==1',True),
            ('set-cardinality-wrong','','len(set([str,str]))==2',False),
            ('dict-key','first=str\npayloads={first:raw}\nassert payloads[str]==raw\n','True',True),
            ('isinstance-str','','isinstance("text",str)',True),
            ('isinstance-class-negative','','isinstance(str,str)',False),
            ('isinstance-path','','isinstance(patch_path,Path)',True),
            ('isinstance-tuple','','isinstance("text",(str,Path))',True),
            ('isinstance-range','','isinstance(range(2),range)',True),
            ('isinstance-enumerate','','isinstance(enumerate([1]),enumerate)',True),
            ('range-wrong-type-branch','','not isinstance(range(2),range)',False),
            ('enumerate-wrong-type-branch','','not isinstance(enumerate([1]),enumerate)',False),
            ('range-value-text','','str(range(2))=="range(0, 2)"',True),
            ('unsupported-function-str','str(str)\n','True',False),
            ('unsupported-nested-function-str','str([str])\n','True',False),
            ('unsupported-nested-path-str','str([patch_path])\n','True',False),
            ('unsupported-bound-method-value','method="text".strip\n','method==method',False),
            ('unsupported-missing-module-attribute','missing=json.strip\n','[missing]==[missing]',False),
        ]
        for builtin in ('str','len','set','all','any','sorted','enumerate','range','isinstance','print'):
            cases.append(('stable-'+builtin,'',builtin+'=='+builtin,True))
        for phase in ('creation','verification'):
            for name,prelude,condition,expected in cases:
                with self.subTest(phase=phase,kind=name):
                    current=copy.deepcopy(trace);creation=source(current['items'][0]);verification=source(current['items'][1])
                    if phase=='creation':
                        creation=creation.replace('with patch_path.open',prelude+'with patch_path.open')
                        creation=creation.replace('    stream.write(raw);stream.flush();os.fsync(stream.fileno())',
                            '    if '+condition+':\n        stream.write(raw)\n    else:\n        stream.write(b\'\')\n    stream.flush();os.fsync(stream.fileno())')
                    else:
                        lines=verification.splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('assert '))
                        end=max(i for i,line in enumerate(lines) if line.startswith('assert '))+1
                        verification='\n'.join(lines[:start])+ '\n'+prelude+'if '+condition+':\n'+ '\n'.join('    '+line for line in lines[start:end])+ '\n'+ '\n'.join(lines[end:])
                    current['items'][0]['command']="python -I -S -B - <<'PY'\n"+creation+'\nPY'
                    current['items'][1]['command']="python -I -S -B - <<'PY'\n"+verification+'\nPY'
                    bind(current)
                    authored=copy.deepcopy(baseline);authored['question_patch_provenance']['evidence_sha256']=evidence['sha256'];seal(authored)
                    provenance={'role':'revision_author','identity':authored['revision_author'],'authored_sha256':authored['sha256'],
                        'author_packet_sha256':authored['source_author_packet_sha256'],'input_channel':'source_only',
                        'task_id':evidence['task_record']['task_id'],'task_record':evidence['task_record']}
                    chunk={'authored':authored,'question_patch_evidence':evidence,'author_task_provenance':[provenance]}
                    if expected:reconstruct_question_patch(evidence);author_context(chunk)
                    else:
                        with self.assertRaises(ValueError):reconstruct_question_patch(evidence)
                        with self.assertRaises(ValueError):author_context(chunk)

    def test_ordinary_function_deepcopy_preserves_identity_in_containers_and_executing_helpers(self):
        evidence=proof_fixture();status=evidence['actual_task_status'];info=evidence['artifact_files']['patch']
        status['latestTerminalSummary']='Completed original patch '+info['path']
        inventory=evidence['actual_task_status_inventory'];seal(inventory)
        text=json.dumps(inventory,sort_keys=True,indent=2)
        evidence['artifact_files']['actual_task_status_inventory'].update(utf8=text,file_sha256=hashlib.sha256(text.encode()).hexdigest(),canonical_sha256=inventory['sha256'])
        evidence['task_record']=normalize_task(status,evidence['patch']['revision_author']['task_id'])
        trace=native_trace_fixture(status,evidence['patch'],info)
        for item in trace['items']:
            boundary=item['text'].rindex('\n{');item['command']=shlex.split(item['text'][:boundary].removeprefix('$ '))[2]
            item['stdout']=item['text'][boundary+1:];item.pop('text')
        creation=trace['items'][0]['command'].split('\n',1)[1].rsplit('\nPY',1)[0]
        def bind(value):
            text=json.dumps({'isError':False,'structuredContent':value,'content':[{'type':'text','text':json.dumps(value)}]})
            evidence['completion_trace_file']={'utf8':text,'file_sha256':hashlib.sha256(text.encode()).hexdigest(),'trace_json_pointer':'/structuredContent'};seal(evidence)
        bind(trace);baseline=reconstruct_question_patch(evidence)
        cases=[
            ('direct-alias','original=seal\n','original==seal',True),
            ('direct-deepcopy','','seal==copy.deepcopy(seal)',True),
            ('false-deepcopy-inequality','','seal!=copy.deepcopy(seal)',False),
            ('alias-deepcopy','original=seal\ncloned=copy.deepcopy(original)\n','cloned==seal',True),
            ('repeated-copy','','copy.deepcopy(seal)==copy.deepcopy(copy.deepcopy(seal))',True),
            ('distinct-function','','seal!=json.dumps',True),
            ('list-membership','values=copy.deepcopy([seal])\n','seal in values',True),
            ('false-list-nonmembership','values=copy.deepcopy([seal])\n','seal not in values',False),
            ('list-cardinality','values=copy.deepcopy([seal,seal])\n','len(set([seal]+values))==1',True),
            ('false-list-cardinality','values=copy.deepcopy([seal,seal])\n','len(set([seal]+values))==2',False),
            ('dict-value','values=copy.deepcopy({"function":seal})\n','values["function"]==seal',True),
            ('false-dict-value','values=copy.deepcopy({"function":seal})\n','values["function"]!=seal',False),
            ('dict-key','values=copy.deepcopy({seal:raw})\n','seal in values',True),
            ('false-dict-key-missing','values=copy.deepcopy({seal:raw})\n','seal not in values',False),
            ('dict-key-lookup','values=copy.deepcopy({seal:raw})\n','values[seal]==raw',True),
            ('dict-key-value-alias','values=copy.deepcopy({seal:seal})\n','values[seal]==seal',True),
            ('set-membership','values=copy.deepcopy(set([seal]))\n','seal in values',True),
            ('false-set-nonmembership','values=copy.deepcopy(set([seal]))\n','seal not in values',False),
            ('set-cardinality','values=copy.deepcopy(set([seal,seal]))\n','len(set([seal]+[item for item in values]))==1',True),
            ('false-set-cardinality','values=copy.deepcopy(set([seal,seal]))\n','len(set([seal]+[item for item in values]))==2',False),
            ('nested-tuple','values=copy.deepcopy((seal,[seal],{seal:seal}))\n','values[0]==seal and values[1][0]==seal and values[2][seal]==seal',True),
            ('list-isolation','values=[seal]\ncloned=copy.deepcopy(values)\nvalues.append(str)\n','len(cloned)==1 and cloned[0]==seal',True),
            ('dict-isolation','values={seal:[seal]}\ncloned=copy.deepcopy(values)\nvalues[seal].append(str)\n','len(cloned[seal])==1 and cloned[seal][0]==seal',True),
            ('shared-container-alias','shared=[seal]\ncloned=copy.deepcopy([shared,shared])\ncloned[0].append(str)\n','len(cloned[1])==2 and len(shared)==1 and cloned[1][0]==seal',True),
            ('nested-mixed','values=copy.deepcopy({"functions":set([seal]),"one":seal})\n','seal in values["functions"] and values["one"]==seal',True),
            ('unsupported-function-text','str(copy.deepcopy(seal))\n','True',False),
            ('unsupported-nested-function-text','str(copy.deepcopy([seal]))\n','True',False),
        ]
        lines=creation.splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('with patch_path.open'))
        end=next(i for i,line in enumerate(lines) if line.startswith('os.chmod'))+1
        for executing_helper in (False,True):
            for name,prelude,condition,expected in cases:
                with self.subTest(executing_helper=executing_helper,kind=name):
                    body=(prelude+"with patch_path.open('xb') as stream:\n    if "+condition+
                        ":\n        stream.write(raw)\n    else:\n        stream.write(b'')\n"
                        "    stream.flush();os.fsync(stream.fileno())\nos.chmod(patch_path,0o444)")
                    if executing_helper:
                        middle='def seal():\n'+'\n'.join('    '+line for line in body.splitlines())+'\nseal()'
                    else:middle='def seal():\n    return True\n'+body
                    source='\n'.join(lines[:start])+'\nimport copy\n'+middle+'\n'+'\n'.join(lines[end:])
                    current=copy.deepcopy(trace);current['items'][0]['command']="python -I -S -B - <<'PY'\n"+source+'\nPY'
                    bind(current)
                    authored=copy.deepcopy(baseline);authored['question_patch_provenance']['evidence_sha256']=evidence['sha256'];seal(authored)
                    provenance={'role':'revision_author','identity':authored['revision_author'],'authored_sha256':authored['sha256'],
                        'author_packet_sha256':authored['source_author_packet_sha256'],'input_channel':'source_only',
                        'task_id':evidence['task_record']['task_id'],'task_record':evidence['task_record']}
                    chunk={'authored':authored,'question_patch_evidence':evidence,'author_task_provenance':[provenance]}
                    if expected:reconstruct_question_patch(evidence);author_context(chunk)
                    else:
                        with self.assertRaises(ValueError):reconstruct_question_patch(evidence)
                        with self.assertRaises(ValueError):author_context(chunk)

    def test_trace_dataflow_binds_written_bytes_read_path_expected_hashes_mode_and_execution(self):
        evidence=proof_fixture();status=evidence['actual_task_status'];info=evidence['artifact_files']['patch']
        status['latestTerminalSummary']='Completed original patch '+info['path']
        inventory=evidence['actual_task_status_inventory'];seal(inventory)
        text=json.dumps(inventory,sort_keys=True,indent=2)
        evidence['artifact_files']['actual_task_status_inventory'].update(utf8=text,file_sha256=hashlib.sha256(text.encode()).hexdigest(),canonical_sha256=inventory['sha256'])
        evidence['task_record']=normalize_task(status,evidence['patch']['revision_author']['task_id'])
        trace=native_trace_fixture(status,evidence['patch'],info)
        for item in trace['items']:
            boundary=item['text'].rindex('\n{');item['command']=shlex.split(item['text'][:boundary].removeprefix('$ '))[2]
            item['stdout']=item['text'][boundary+1:];item.pop('text')
        def source(item):return item['command'].split('\n',1)[1].rsplit('\nPY',1)[0]
        def bind(value):
            text=json.dumps({'isError':False,'structuredContent':value,'content':[{'type':'text','text':json.dumps(value)}]})
            evidence['completion_trace_file']={'utf8':text,'file_sha256':hashlib.sha256(text.encode()).hexdigest(),'trace_json_pointer':'/structuredContent'};seal(evidence)
        bind(trace);baseline_merged=reconstruct_question_patch(evidence)
        for kind in ('unrelated-payload','empty-write','wrong-computed-target','wrong-read-path','canonical-unanchored',
                     'raw-tautology','canonical-str-self','canonical-strip-self','raw-str-self','raw-strip-self',
                     'nested-function-closure','chained-comparison-short-circuit','constant-string-identity-branch',
                     'str-encoding-branch','items-extra-argument','canonical-or-true','raw-or-true','mode-or-true',
                     'posonly-arity','parameter-annotation','return-annotation','type-parameters','duplicate-parameters',
                     'duplicate-keywords','fsync-keyword','encode-positional-keyword-duplicate','return-outside-function',
                     'extra-command-alias','extra-stdout-alias','extra-combined-alias','extra-incomplete-target',
                     'extra-unrelated-alias-conflict','extra-malformed-json','valid-extra-unrelated-command',
                     'missing-creation-import','missing-verification-import','dead-imports','valid-import-aliases',
                     'late-local-path','late-local-raw','dead-local-for','dead-local-with','dead-local-import',
                     'valid-function','valid-comprehension-shadow','unused-write-generator','all-shortcircuit-write-generator',
                     'valid-consumed-write-generator','valid-pure-generator','valid-deferred-generator-values',
                     'valid-dict-items','valid-dict-values','valid-dict-keys','valid-items-live-value-update',
                     'valid-items-live-size-update-before-iterator','valid-view-comparisons',
                     'dict-view-join-exceeds-bound',
                     'items-view-changed-value','items-generator-changed-value','items-generator-changed-size',
                     'items-generator-changed-order','values-generator-changed-value','keys-generator-changed-size',
                     'function-generator-late-global-change',
                     'extra-target-wrong-raw','extra-target-missing-path','extra-target-missing-flag',
                     'valid-extra-packing-summary','extra-packing-wrong-target-raw','extra-packing-missing-target-path',
                     'extra-packing-wrong-response-raw','extra-packing-wrong-response-seal','extra-packing-wrong-response-path',
                     'extra-packing-response-is-patch','extra-packing-patch-is-response','extra-packing-both-unbound',
                     'extra-packing-missing-response','extra-packing-missing-patch',
                     'function-return-before-write','mode-tautology','wrong-mode','dead-checks','duplicate-completed-run','duplicate-failed-run',
                     'distinct-old-run','exact-computed-target'):
            with self.subTest(kind=kind):
                value=copy.deepcopy(trace);creation=source(value['items'][0]);verify=source(value['items'][1])
                if kind=='unrelated-payload':creation=creation.replace(repr(info['utf8']),repr('{"sha256":"'+('0'*64)+'"}'))
                elif kind=='empty-write':creation=creation.replace('stream.write(raw)',"stream.write(b'')")
                elif kind in ('wrong-computed-target','exact-computed-target'):
                    path=info['path'] if kind=='exact-computed-target' else '/tmp/unrelated-original-target.json'
                    creation=creation.replace('patch_path=Path('+repr(info['path'])+')','bound_path='+repr(path)+'\npatch_path=Path(bound_path)')
                elif kind=='wrong-read-path':verify=verify.replace(repr(info['path']),repr('/tmp/unrelated-original-read.json'))
                elif kind=='canonical-unanchored':verify=verify.replace("assert canonical==obj['sha256']=="+repr(evidence['patch']['sha256']),"assert canonical==obj['sha256']")
                elif kind=='raw-tautology':verify=verify.replace('assert hashlib.sha256(raw).hexdigest()=='+repr(info['file_sha256']),'assert hashlib.sha256(raw).hexdigest()==hashlib.sha256(raw).hexdigest()')
                elif kind in ('canonical-str-self','canonical-strip-self'):
                    expected_expr="str(obj['sha256'])" if kind=='canonical-str-self' else "obj['sha256'].strip()"
                    verify=verify.replace("assert canonical==obj['sha256']=="+repr(evidence['patch']['sha256']),'assert canonical=='+expected_expr)
                elif kind in ('raw-str-self','raw-strip-self'):
                    expected_expr='str(hashlib.sha256(raw).hexdigest())' if kind=='raw-str-self' else 'hashlib.sha256(raw).hexdigest().strip()'
                    verify=verify.replace('assert hashlib.sha256(raw).hexdigest()=='+repr(info['file_sha256']),'assert hashlib.sha256(raw).hexdigest()=='+expected_expr)
                elif kind=='function-return-before-write':
                    lines=creation.splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('with patch_path.open'))
                    end=next(i for i,line in enumerate(lines) if line.startswith('os.chmod'))
                    creation='\n'.join(lines[:start])+"\ndef seal():\n    if True:\n        return None\n"+'\n'.join('    '+line for line in lines[start:end+1])+"\nseal()\n"+'\n'.join(lines[end+1:])
                elif kind=='nested-function-closure':
                    lines=creation.splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('with patch_path.open'))
                    end=next(i for i,line in enumerate(lines) if line.startswith('os.chmod'))
                    creation='\n'.join(lines[:start])+"\ndef seal():\n    raw=b'wrong closure payload'\n    def seal():\n"+'\n'.join('        '+line for line in lines[start:end+1])+"\n    seal()\nseal()\n"+'\n'.join(lines[end+1:])
                elif kind=='chained-comparison-short-circuit':
                    verify=verify.replace('canonical=hashlib',"if 1==2==obj.pop('sha256'):\n    pass\ncanonical=hashlib").replace("{k:v for k,v in obj.items() if k!='sha256'}",'obj').replace("assert canonical==obj['sha256']==",'assert canonical==')
                elif kind=='constant-string-identity-branch':creation=creation.replace('with patch_path.open',"if 'same' is 'same':\n    raw=b'wrong actual identity branch'\nwith patch_path.open")
                elif kind=='str-encoding-branch':creation=creation.replace('with patch_path.open',"if str(b'x','utf-8')=='x':\n    raw=b'wrong actual decode branch'\nwith patch_path.open")
                elif kind=='items-extra-argument':verify=verify.replace('obj.items()','obj.items(123)')
                elif kind in ('canonical-or-true','raw-or-true','mode-or-true'):
                    prefix='assert canonical' if kind=='canonical-or-true' else 'assert hashlib' if kind=='raw-or-true' else 'assert stat'
                    line=next(line for line in verify.splitlines() if line.startswith(prefix));verify=verify.replace(line,line+' or True')
                elif kind in ('posonly-arity','parameter-annotation','return-annotation','type-parameters','duplicate-parameters'):
                    lines=creation.splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('with patch_path.open'))
                    end=next(i for i,line in enumerate(lines) if line.startswith('os.chmod'))
                    signature={'posonly-arity':'def seal(required, /):','parameter-annotation':'def seal(unused: unavailable_type):',
                        'return-annotation':'def seal() -> unavailable_type:','type-parameters':'def seal[T]():',
                        'duplicate-parameters':'def seal(unused,unused):'}[kind]
                    invocation='seal(None)' if kind=='parameter-annotation' else 'seal(None,None)' if kind=='duplicate-parameters' else 'seal()'
                    creation='\n'.join(lines[:start]+[signature]+['    '+line for line in lines[start:end+1]]+[invocation]+lines[end+1:])
                elif kind=='duplicate-keywords':verify=verify.replace('sort_keys=True,separators=','sort_keys=False,sort_keys=True,separators=')
                elif kind=='fsync-keyword':creation=creation.replace('os.fsync(stream.fileno())','os.fsync(stream.fileno(),bogus=True)')
                elif kind=='encode-positional-keyword-duplicate':creation=creation.replace('.encode()',".encode('utf-8',encoding='utf-8')")
                elif kind=='return-outside-function':creation='if False:\n    return None\n'+creation
                elif kind in ('missing-creation-import','missing-verification-import','dead-imports','valid-import-aliases'):
                    if kind=='missing-creation-import':creation=creation.replace('from pathlib import Path\n','')
                    elif kind=='missing-verification-import':verify=verify.replace('from pathlib import Path\n','')
                    elif kind=='dead-imports':
                        creation=creation.replace('from pathlib import Path','if False:\n    from pathlib import Path')
                        verify=verify.replace('from pathlib import Path','if False:\n    from pathlib import Path')
                    else:
                        creation=creation.replace('from pathlib import Path','from pathlib import Path as BoundPath').replace('Path(','BoundPath(')
                        verify=verify.replace('from pathlib import Path','from pathlib import Path as BoundPath').replace('Path(','BoundPath(')
                elif kind in ('late-local-path','late-local-raw','dead-local-for','dead-local-with','dead-local-import',
                              'valid-function','valid-comprehension-shadow'):
                    lines=creation.splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('with patch_path.open'))
                    end=next(i for i,line in enumerate(lines) if line.startswith('os.chmod'));body=lines[start:end+1]
                    if kind=='late-local-path':body.append('patch_path=Path('+repr(info['path'])+')')
                    elif kind=='late-local-raw':body.append("raw=b'late local'")
                    elif kind=='dead-local-for':body.extend(['if False:','    for raw in []:','        None'])
                    elif kind=='dead-local-with':body.extend(['if False:',"    with patch_path.open('xb') as raw:",'        None'])
                    elif kind=='dead-local-import':body.extend(['if False:','    import os'])
                    elif kind=='valid-comprehension-shadow':body.append("(raw for raw in [b'unconsumed'])")
                    creation='\n'.join(lines[:start]+['def seal():']+['    '+line for line in body]+['seal()']+lines[end+1:])
                elif kind=='unused-write-generator':creation=creation.replace('stream.write(raw)','(stream.write(raw) for unused in [0])')
                elif kind=='all-shortcircuit-write-generator':creation=creation.replace('stream.write(raw)',"all(stream.write(piece) for piece in [b'',raw])")
                elif kind=='valid-consumed-write-generator':creation=creation.replace('stream.write(raw)','all(stream.write(piece) for piece in [raw])')
                elif kind=='valid-pure-generator':creation=creation.replace('with patch_path.open',"assert all(value==1 for value in [1])\nwith patch_path.open")
                elif kind=='valid-deferred-generator-values':
                    creation=creation.replace('with patch_path.open',"later=b'wrong'\ndeferred=(later for unused in [0])\nlater=raw\nwith patch_path.open").replace('stream.write(raw)','all(stream.write(piece) for piece in deferred)')
                elif kind=='dict-view-join-exceeds-bound':
                    creation=creation.replace('with patch_path.open',"keys={str(index):'' for index in range(2000)}\nkeys['overflow']=''\n''.join(keys.keys())\nwith patch_path.open")
                elif kind in ('valid-dict-items','valid-dict-values','valid-dict-keys','valid-items-live-value-update',
                              'valid-items-live-size-update-before-iterator','valid-view-comparisons',
                              'items-view-changed-value','items-generator-changed-value','items-generator-changed-size',
                              'items-generator-changed-order','values-generator-changed-value','keys-generator-changed-size'):
                    prelude="payloads={'piece':raw}\n"
                    if kind=='valid-dict-values':prelude+='deferred=payloads.values()\n'
                    elif kind=='valid-dict-keys':prelude+='deferred=(payloads[key] for key in payloads.keys())\n'
                    elif kind=='valid-items-live-value-update':prelude="payloads={'piece':b'wrong'}\nview=payloads.items()\npayloads['piece']=raw\ndeferred=(piece for key,piece in view)\n"
                    elif kind=='valid-items-live-size-update-before-iterator':prelude+="view=payloads.items()\npayloads['extra']=b''\ndeferred=(piece for key,piece in view)\n"
                    elif kind=='valid-view-comparisons':prelude+="view=payloads.values()\nassert view==view\nassert view!=payloads.values()\nassert payloads.keys()==set(['piece'])\nassert payloads.items()==set([('piece',raw)])\nassert ''.join(payloads.keys())=='piece'\ndeferred=(piece for key,piece in payloads.items())\n"
                    elif kind=='items-view-changed-value':prelude+="view=payloads.items()\npayloads['piece']=b''\ndeferred=(piece for key,piece in view)\n"
                    elif kind=='values-generator-changed-value':prelude+="deferred=(piece for piece in payloads.values())\npayloads['piece']=b''\n"
                    elif kind=='keys-generator-changed-size':prelude+="deferred=(payloads[key] for key in payloads.keys())\npayloads['extra']=b''\n"
                    else:
                        prelude+='deferred=(piece for key,piece in payloads.items())\n'
                        if kind=='items-generator-changed-value':prelude+="payloads['piece']=b''\n"
                        elif kind=='items-generator-changed-size':prelude+="payloads['extra']=b''\n"
                        elif kind=='items-generator-changed-order':prelude+="payloads.pop('piece')\npayloads['other']=b''\n"
                    creation=creation.replace('with patch_path.open',prelude+'with patch_path.open').replace('stream.write(raw)','all(stream.write(piece) for piece in deferred)')
                elif kind=='function-generator-late-global-change':
                    creation=creation.replace('with patch_path.open',"def seal():\n    return (raw for unused in [0])\ndeferred=seal()\nraw=b'wrong later global'\nwith patch_path.open").replace('stream.write(raw)','all(stream.write(piece) for piece in deferred)')
                elif kind.startswith('extra-') or kind in ('valid-extra-unrelated-command','valid-extra-packing-summary'):
                    extra=copy.deepcopy(value['items'][1]);extra['position']=20
                    if kind=='extra-command-alias':extra['cmd']=extra['command'].replace('patch_path','conflicting_target')
                    elif kind=='extra-stdout-alias':extra['standard_output']='{}'
                    elif kind=='extra-combined-alias':extra['text']='$ '+extra['command']+'\n{}'
                    elif kind=='extra-incomplete-target':extra.pop('command')
                    elif kind=='extra-malformed-json':extra['stdout']=extra['stdout'].rstrip()[:-1]
                    elif kind.startswith('extra-target-'):
                        report=json.loads(extra['stdout']);row=report['artifacts'][0]
                        if kind=='extra-target-wrong-raw':row['file_sha256']='0'*64
                        elif kind=='extra-target-missing-path':row.pop('path')
                        else:row.pop('raw_written_bytes_independently_verified')
                        extra['stdout']=json.dumps(report,indent=2)
                        extra_source=source(extra);extra_source=extra_source[:extra_source.rindex('print(json.dumps(')]+'print(json.dumps('+repr(report)+',indent=2))'
                        extra['command']="python -I -S -B - <<'PY'\n"+extra_source+'\nPY'
                    elif 'packing' in kind:
                        descriptor={'path':info['path'],'sha256':evidence['patch']['sha256'],'file_sha256':info['file_sha256']}
                        response_info=evidence['artifact_files']['response']
                        response_descriptor={'path':response_info['path'],'sha256':evidence['response']['sha256'],
                                             'file_sha256':response_info['file_sha256'],'mode':'0444'}
                        report={'packing_output':{},'patch':descriptor,
                            'response':response_descriptor,
                            'canonical_verified':True,'packing_import_profile_event_count':0,'packing_stderr_non_import_lines':[]}
                        if kind=='extra-packing-wrong-target-raw':descriptor['file_sha256']='0'*64
                        elif kind=='extra-packing-missing-target-path':descriptor.pop('path')
                        elif kind=='extra-packing-wrong-response-raw':response_descriptor['file_sha256']='0'*64
                        elif kind=='extra-packing-wrong-response-seal':response_descriptor['sha256']='0'*64
                        elif kind=='extra-packing-wrong-response-path':response_descriptor['path']='/tmp/unrelated-response.json'
                        elif kind=='extra-packing-response-is-patch':report['response']=copy.deepcopy(descriptor)
                        elif kind=='extra-packing-patch-is-response':report['patch']=copy.deepcopy(response_descriptor)
                        elif kind=='extra-packing-both-unbound':
                            descriptor['path']='/tmp/unbound-patch.json';response_descriptor['path']='/tmp/unbound-response.json'
                        elif kind=='extra-packing-missing-response':report.pop('response')
                        elif kind=='extra-packing-missing-patch':report.pop('patch')
                        extra.update(command="python -I -S -B - <<'PY'\nimport json\nprint(json.dumps("+repr(report)+",indent=2))\nPY",
                                     stdout=json.dumps(report,indent=2))
                    else:
                        extra.update(command='pwd',stdout='/tmp')
                        if kind=='extra-unrelated-alias-conflict':extra['cmd']='ls'
                    value['items'].append(extra)
                elif kind=='mode-tautology':verify=verify.replace('assert stat.S_IMODE(patch_path.stat().st_mode)==0o444','assert stat.S_IMODE(patch_path.stat().st_mode)==stat.S_IMODE(patch_path.stat().st_mode)')
                elif kind=='wrong-mode':verify=verify.replace('==0o444','==0o644')
                elif kind=='dead-checks':
                    lines=verify.splitlines();verify='\n'.join(lines[:3])+"\nnever=1==2\nif never:\n"+'\n'.join('    '+line for line in lines[3:-1])+'\n'+lines[-1]
                elif kind in ('duplicate-completed-run','duplicate-failed-run'):value['recentRuns'].append({**value['recentRuns'][0],'status':'failed' if 'failed' in kind else 'completed'})
                elif kind=='distinct-old-run':value['recentRuns'].append({'runId':'different-old-run','model':'different-old-model','status':'failed'})
                value['items'][0]['command']="python -I -S -B - <<'PY'\n"+creation+'\nPY'
                value['items'][1]['command']="python -I -S -B - <<'PY'\n"+verify+'\nPY'
                bind(value)
                merged=copy.deepcopy(baseline_merged);merged['question_patch_provenance']['evidence_sha256']=evidence['sha256'];seal(merged)
                row={'role':'revision_author','identity':merged['revision_author'],'authored_sha256':merged['sha256'],
                    'author_packet_sha256':merged['source_author_packet_sha256'],'input_channel':'source_only',
                    'task_id':evidence['task_record']['task_id'],'task_record':evidence['task_record']}
                chunk={'authored':merged,'question_patch_evidence':evidence,'author_task_provenance':[row]}
                if kind in ('distinct-old-run','exact-computed-target','valid-extra-unrelated-command','valid-import-aliases',
                            'valid-function','valid-comprehension-shadow','valid-consumed-write-generator','valid-pure-generator',
                            'valid-deferred-generator-values','valid-extra-packing-summary'):
                    reconstruct_question_patch(evidence);author_context(chunk)
                elif kind in ('valid-dict-items','valid-dict-values','valid-dict-keys','valid-items-live-value-update',
                              'valid-items-live-size-update-before-iterator','valid-view-comparisons'):
                    reconstruct_question_patch(evidence);author_context(chunk)
                else:
                    with self.assertRaises((ValueError,RuntimeError)):reconstruct_question_patch(evidence)
                    with self.assertRaises((ValueError,RuntimeError)):author_context(chunk)

    def test_latest_failed_pending_stale_run_or_summary_cannot_qualify_patch(self):
        for kind in ('failed','pending','pending-integer','stale-run','missing-run','stale-summary'):
            with self.subTest(kind=kind):
                evidence=proof_fixture();status=evidence['actual_task_status']
                if kind=='failed':status['latestTerminalStatus']='failed'
                elif kind=='pending':status['hasPendingChildRuns']=True
                elif kind=='pending-integer':status['hasPendingChildRuns']=0
                elif kind=='stale-run':status['latestTerminalRunId']='another-run'
                elif kind=='missing-run':status.pop('latestTerminalRunId')
                else:status['latestTerminalSummary']='Latest completed result does not bind patch'
                inventory=evidence['actual_task_status_inventory'];seal(inventory)
                text=json.dumps(inventory,sort_keys=True,indent=2)
                evidence['artifact_files']['actual_task_status_inventory'].update(utf8=text,
                    file_sha256=hashlib.sha256(text.encode()).hexdigest(),canonical_sha256=inventory['sha256'])
                seal(evidence)
                with self.assertRaises(ValueError):reconstruct_question_patch(evidence)

    def test_original_same_run_trace_binds_patch_without_restamping_latest_summary(self):
        evidence=proof_fixture();status=evidence['actual_task_status'];info=evidence['artifact_files']['patch']
        status['latestTerminalSummary']='Original completed patch '+info['path']+'; hashes printed in original command'
        inventory=evidence['actual_task_status_inventory'];seal(inventory)
        text=json.dumps(inventory,sort_keys=True,indent=2)
        evidence['artifact_files']['actual_task_status_inventory'].update(utf8=text,
            file_sha256=hashlib.sha256(text.encode()).hexdigest(),canonical_sha256=inventory['sha256'])
        evidence['task_record']=normalize_task(status,evidence['patch']['revision_author']['task_id'])
        trace=native_trace_fixture(status,evidence['patch'],info)
        def bind(value):
            envelope={'structuredContent':value,'isError':False,'content':[{'type':'text','text':json.dumps(value)}]}
            text=json.dumps(envelope)
            evidence['completion_trace_file']={'utf8':text,'file_sha256':hashlib.sha256(text.encode()).hexdigest(),
                                               'trace_json_pointer':'/structuredContent'}
            seal(evidence)
        bind(trace);original_summary=status['latestTerminalSummary'];reconstruct_question_patch(evidence)
        self.assertEqual(status['latestTerminalSummary'],original_summary)
        for kind in ('wrong-run','truncated','wrong-bytes','wrong-seal','not-readonly'):
            with self.subTest(kind=kind):
                changed=copy.deepcopy(trace);item=changed['items'][0]
                if kind=='wrong-run':item['runId']='another-run'
                elif kind=='truncated':item['textTruncated']=True
                elif kind=='wrong-bytes':item['text']=item['text'].replace(info['file_sha256'],'0'*64)
                elif kind=='wrong-seal':item['text']=item['text'].replace(evidence['patch']['sha256'],'0'*64)
                else:item['text']=item['text'].replace('0o444','0o644')
                bind(changed)
                with self.assertRaises(ValueError):reconstruct_question_patch(evidence)

    def test_native_json_creation_and_verification_outputs_bind_exact_latest_patch(self):
        evidence=proof_fixture();status=evidence['actual_task_status'];info=evidence['artifact_files']['patch']
        status['latestTerminalSummary']='Completed patch '+info['path']+'; full hashes printed in commands'
        inventory=evidence['actual_task_status_inventory'];seal(inventory)
        text=json.dumps(inventory,sort_keys=True,indent=2)
        evidence['artifact_files']['actual_task_status_inventory'].update(utf8=text,
            file_sha256=hashlib.sha256(text.encode()).hexdigest(),canonical_sha256=inventory['sha256'])
        evidence['task_record']=normalize_task(status,evidence['patch']['revision_author']['task_id'])
        trace=native_trace_fixture(status,evidence['patch'],info)
        def bind(value,extra_text=None):
            envelope={'structuredContent':value,'isError':False,'content':[{'type':'text','text':json.dumps(value)}]}
            if extra_text is not None:envelope['content'].append({'type':'text','text':json.dumps(extra_text)})
            text=json.dumps(envelope)
            evidence['completion_trace_file']={'utf8':text,'file_sha256':hashlib.sha256(text.encode()).hexdigest(),
                                               'trace_json_pointer':'/structuredContent'};seal(evidence)
        bind(trace);reconstruct_question_patch(evidence)
        explicit=copy.deepcopy(trace)
        for item in explicit['items']:
            boundary=item['text'].rindex('\n{')
            item['command']=item['text'][:boundary]
            item['stdout']=item['text'][boundary+1:]
            item['exit_code']=0
            item.pop('text')
        bind(explicit);reconstruct_question_patch(evidence)
        for kind in ('creation-missing','creation-after-verification','creation-wrong-bytes','verification-wrong-seal',
                     'verification-bool-integer','verification-writes','duplicate-verification','command-source-only',
                     'chmod-comment-only','verification-write-call','creation-bool-position','verification-bool-position',
                     'failed-exit','failed-exit-bool','contradictory-MCP-text','contradictory-command-alias',
                     'contradictory-stdout-alias','verification-report-only'):
            with self.subTest(kind=kind):
                changed=copy.deepcopy(trace)
                if kind=='creation-missing':changed['items'].pop(0)
                elif kind=='creation-after-verification':changed['items'][0]['position']=17
                elif kind=='creation-wrong-bytes':changed['items'][0]['text']=changed['items'][0]['text'].replace(info['file_sha256'],'0'*64)
                elif kind=='verification-wrong-seal':changed['items'][1]['text']=changed['items'][1]['text'].replace(evidence['patch']['sha256'],'0'*64)
                elif kind=='verification-bool-integer':changed['items'][1]['text']=changed['items'][1]['text'].replace('"raw_written_bytes_independently_verified": true','"raw_written_bytes_independently_verified": 1')
                elif kind=='verification-writes':changed['items'][1]['text']=changed['items'][1]['text'].replace('"verification_performed_no_writes": true','"verification_performed_no_writes": false')
                elif kind=='duplicate-verification':changed['items'].append(copy.deepcopy(changed['items'][1]))
                elif kind=='command-source-only':changed['items'][1]['text']=changed['items'][1]['text'].split('\n{')[0]
                elif kind=='chmod-comment-only':changed['items'][0]['text']=changed['items'][0]['text'].replace('os.chmod(patch_path,0o444)','# os.chmod(patch_path,0o444)')
                elif kind=='verification-write-call':changed['items'][1]['text']=changed['items'][1]['text'].replace('raw=patch_path.read_bytes()','patch_path.write_text("bad");raw=patch_path.read_bytes()')
                elif kind=='creation-bool-position':changed['items'][0]['position']=False
                elif kind=='verification-bool-position':changed['items'][1]['position']=True
                elif kind=='failed-exit':changed['items'][1]['exit_code']=7
                elif kind=='failed-exit-bool':changed['items'][1]['exit_code']=False
                elif kind in ('contradictory-command-alias','contradictory-stdout-alias','verification-report-only'):
                    changed=copy.deepcopy(explicit);item=changed['items'][1]
                    if kind=='contradictory-command-alias':item['cmd']=item['command'].replace('patch_path','another_path')
                    elif kind=='contradictory-stdout-alias':item['standard_output']=item['stdout'].replace('"readonly_mode": "0444"','"readonly_mode": "0644"')
                    else:item['command']="python -I -S -B - <<'PY'\nimport json\nprint(json.dumps("+repr(json.loads(item['stdout']))+"))\nPY"
                bind(changed,{} if kind=='contradictory-MCP-text' else None)
                with self.assertRaises(ValueError):reconstruct_question_patch(evidence)

    def test_selected_event_projection_hides_probe_metadata_and_restores_entire_locked_base(self):
        evidence=proof_fixture();base=evidence['base_authored']
        base['source_units'][0].update(opaque_probe_ids=['hidden-evaluator-id'],nonliteral_answer_labels=['hidden-author-classification'])
        unrelated=copy.deepcopy(base['source_units'][0]);unrelated['event']='unrelated-event';unrelated['unit_id']='unrelated-unit'
        base['source_units'].append(unrelated);base['events'].append('unrelated-event');seal(base)
        omissions=list(AUTHOR_RATIONALE_FIELDS)+list(SOURCE_BLIND_METADATA_FIELDS)
        for key in ('input_packet','patch'):
            value=evidence[key];value['current_authored_sha256']=base['sha256']
            if key=='patch':value['prior_authored_sha256']=base['sha256']
            selected_event=value['selected_acquisition_records'][0]['event']
            value['events']=[e for e in base['events'] if e==selected_event]
            value['groups']=source_only_projection([g for g in base['groups'] if g['event']==selected_event],omissions)
            value['source_units']=source_only_projection([u for u in base['source_units'] if u['event']==selected_event],omissions)
            value['source_context_scope']='selected_events_only'
            evidence[key]=source_only_projection(value,omissions)
            evidence[key]['locked_author_metadata_omission_fields']=list(AUTHOR_RATIONALE_FIELDS)
            evidence[key]['locked_source_metadata_omission_fields']=list(SOURCE_BLIND_METADATA_FIELDS)
        evidence['response']['prior_authored_sha256']=base['sha256'];refresh(evidence)
        merged=reconstruct_question_patch(evidence)
        self.assertEqual(merged['source_units'],base['source_units'])
        self.assertEqual(merged['events'],base['events'])
        self.assertNotIn('opaque_probe_ids',evidence['input_packet']['source_units'][0])
        for kind in ('hide-gold','alter-source','omit-own-unit','invent-scope'):
            with self.subTest(kind=kind):
                changed=copy.deepcopy(evidence)
                for key in ('input_packet','patch'):
                    if kind=='hide-gold':changed[key]['locked_source_metadata_omission_fields'].append('answer')
                    elif kind=='alter-source':changed[key]['source_units'][0]['source_assertion']='Changed source'
                    elif kind=='omit-own-unit':changed[key]['source_units'].pop(0)
                    else:changed[key]['source_context_scope']='selected_units_only'
                refresh(changed)
                with self.assertRaises(ValueError):reconstruct_question_patch(changed)

    def test_rationale_free_author_projection_preserves_all_original_locked_metadata(self):
        evidence = proof_fixture()
        for key in ('input_packet', 'patch'):
            evidence[key] = source_only_projection(evidence[key])
            evidence[key]['locked_author_metadata_omission_fields'] = list(AUTHOR_RATIONALE_FIELDS)
        refresh(evidence)
        merged = reconstruct_question_patch(evidence);base = evidence['base_authored']
        self.assertNotIn('rationale', evidence['input_packet']['selected_acquisition_records'][0])
        self.assertEqual(merged['groups'], base['groups'])
        self.assertEqual(merged['source_units'], base['source_units'])
        for before, after in zip(base['acquisition_records'], merged['acquisition_records']):
            self.assertEqual({k:v for k,v in before.items() if k != 'questions'},
                             {k:v for k,v in after.items() if k != 'questions'})

    def test_rationale_projection_cannot_hide_gold_source_or_protected_metadata_changes(self):
        for kind in ('arbitrary-omission', 'changed-gold', 'reintroduced-rationale', 'source', 'omit-evidence'):
            with self.subTest(kind=kind):
                evidence = proof_fixture()
                for key in ('input_packet', 'patch'):
                    evidence[key] = source_only_projection(evidence[key])
                    evidence[key]['locked_author_metadata_omission_fields'] = list(AUTHOR_RATIONALE_FIELDS)
                if kind == 'arbitrary-omission':
                    for key in ('input_packet', 'patch'):evidence[key]['locked_author_metadata_omission_fields'].append('answer')
                elif kind == 'changed-gold':
                    for key in ('input_packet', 'patch'):evidence[key]['selected_acquisition_records'][0]['answer'] = 'different gold'
                elif kind == 'reintroduced-rationale':evidence['patch']['selected_acquisition_records'][0]['rationale'] = 'replacement opinion'
                elif kind == 'source':
                    for key in ('input_packet', 'patch'):evidence[key]['groups'][0]['statement'] = 'different assertion'
                else:
                    for key in ('input_packet', 'patch'):evidence[key]['selected_acquisition_records'][0].pop('evidence')
                refresh(evidence)
                with self.assertRaises(ValueError):reconstruct_question_patch(evidence)

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
