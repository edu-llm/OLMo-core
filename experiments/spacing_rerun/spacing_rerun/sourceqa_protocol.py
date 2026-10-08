"""Exact prospective source-QA protocol coverage, never semantic approval creation.

Coverage packages retain genuine original receipts and task bytes. Adapters name
JSON pointers into those bytes; they cannot substitute a parent's pass flag.
"""
from __future__ import annotations

import copy
import ast
import datetime
import hashlib
import json
import math
import os
import random
import re
import shlex
import stat
from pathlib import Path as _NativePath, PurePosixPath
from urllib.parse import unquote

from .common import digest, require
from .confirmation import (author_context, checked, decisions_for, identity_channels, identity_names, packet_binding,
                           raw_artifact, receipt_identity_names, source_preservation_context, source_review_content, strict_json_equal)

SCHEMA = 'spacing-sourceqa-protocol-coverage-v1'
ADOPTED_PROTOCOL_SHA256 = 'dc5f4728b1756fd9490024d4591c7b497e910f1ec9d007e74cd3037dbde89561'
DEPENDENCY_KEYS = {'membership_sha256', 'exception_lanes_sha256', 'required_source_conditions_sha256',
                   'lane_reporting_policy_sha256', 'algorithm_sha256'}


def json_pointer(value, pointer):
    require(isinstance(pointer, str) and (pointer == '' or pointer.startswith('/')), 'Invalid actual artifact JSON pointer')
    for part in pointer.split('/')[1:]:
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def completed_artifact(package):
    """Bind genuine final bytes and latest completed original task, including direct status snapshots."""
    aliases=[checked(package[k]) for k in ('artifact','receipt') if k in package]
    require(aliases and all(_strict_equal(v,aliases[0]) for v in aliases),
            'Completed package artifact/receipt aliases are absent or contradictory')
    artifact=aliases[0]
    evidence = package['evidence'];raw_artifact(artifact, evidence['receipt_file'])
    require('receipt_sha256' not in evidence or evidence['receipt_sha256'] == artifact['sha256'],
            'Actual completion evidence declares another receipt seal')
    info = evidence['task_status_file'];text = info.get('utf8')
    require(isinstance(text, str) and hashlib.sha256(text.encode()).hexdigest() == info.get('file_sha256'),
            'Protocol artifact lacks retained exact actual task-status bytes')
    snapshot = json.loads(text);status = evidence['task_status'];client = evidence['client_request_id']
    if 'status_json_pointer' in info:
        candidates = [json_pointer(snapshot, info['status_json_pointer'])]
    else:
        candidates = ([r['result'] for r in snapshot['task_statuses'] if r.get('clientRequestId') == client]
                      if 'task_statuses' in snapshot else snapshot.get('records', [snapshot]))
    require(sum(_strict_equal(s,status) for s in candidates) == 1 and client and
            status.get('status') == status.get('latestTerminalStatus') == 'completed' and
            status.get('hasPendingChildRuns') is False and status.get('taskId') and status.get('childThreadId') and
            status.get('childRunId') and status.get('model') and
            status.get('latestTerminalRunId') == status['childRunId'] and
            completed_output_binding(artifact, evidence, status),
            'Protocol artifact lacks genuine latest completed exact-output task evidence')
    require(unquote(status['taskId']).endswith(':delegate-task:' + client) or
            'task_statuses' in snapshot and any(r.get('clientRequestId') == client and _strict_equal(r.get('result'),status)
                                              for r in snapshot['task_statuses']),
            'Protocol artifact task label is not bound by the actual status snapshot')
    declared=set();roles=set()
    if 'reviewer_actor_identity' in artifact:
        actor=artifact['reviewer_actor_identity']
        require(isinstance(actor,str) and actor and actor==actor.strip() and actor not in ('/root','root'),
                'Completed review stable actor identity is malformed or generic')
        declared.add(actor)
    for key in ('reviewer','reviewer_identity'):
        if key not in artifact:continue
        person=artifact[key]
        require(isinstance(person,dict), 'Completed review lacks its declared provider/model identity')
        models=[person[k] for k in ('model','model_id') if k in person]
        require(models and all(model==status['model'] for model in models),
                'Declared reviewer model differs from actual task completion')
        declared.update(identity_channels(person))
        for role_key in ('role','reviewer_role'):
            if role_key not in person:continue
            role=person[role_key]
            require(isinstance(role,str) and role.strip() and role==role.strip(), 'Declared reviewer role is malformed')
            require(not re.search(r'(?:\bauthor\b|_author(?:_|$))',role.casefold()),
                    'Declared reviewer role is an author role')
            roles.add(role)
        actual_channels={client,status['taskId'],status['childRunId'],status['childThreadId']}
        if 'reviewer_actor_identity' in artifact:actual_channels.add(artifact['reviewer_actor_identity'])
        actual_channels.update(('/root','root'))
        for key in ('agent_id','identity','review_task_id','review_name','name','actual_identity','client_label'):
            if key in person:require(person[key] in actual_channels,
                                    'Declared reviewer identity alias is not bound to actual actor/task completion')
        typed={'review_task_id':{client,status['taskId']},'client_label':{client},'task_id':{status['taskId']},
               'client_request_id':{client},'child_run_id':{status['childRunId']},'run_id':{status['childRunId']},
               'child_thread_id':{status['childThreadId']},'thread_id':{status['childThreadId']},
               'runtime_thread_id':{status['childThreadId']},'runtime_session_id':{status['childThreadId']},
               'codex_thread_id':{status['childThreadId']}}
        for key,expected in typed.items():
            if key in person:require(person[key] in expected,
                                    'Declared reviewer execution channel differs from actual task completion')
        for key in ('provider_instance_id','providerInstanceId'):
            if key in person:require(status.get('providerInstanceId') and person[key]==status['providerInstanceId'],
                                    'Declared reviewer provider instance differs from actual provenance')
        if status.get('providerFamily') and 'provider' in person:
            require(person['provider'] in (status['providerFamily'],status.get('providerInstanceId')),
                    'Declared reviewer provider family differs from actual orchestration provenance')
    return artifact, {'task_id': status['taskId'], 'client_request_id': client,
                      'child_run_id': status['childRunId'], 'child_thread_id':status['childThreadId'],
                      'model':status['model'],'provider_instance_id':status.get('providerInstanceId'),
                      'declared_reviewer_identities':sorted(declared),'declared_reviewer_roles':sorted(roles),
                      'task_status_sha256': digest(status)}


def _python_trace_source(command):
    """Decode only an isolated Python stdin heredoc, with no trailing shell action."""
    require(isinstance(command,str) and command.strip(),'Original command transport is absent')
    command=command.removeprefix('$ ').strip()
    if command.startswith(('/bin/zsh ','/bin/bash ','/bin/sh ')):
        require('$(' not in command and '`' not in command,'Unsupported command substitution in original transport')
        words=shlex.split(command)
        require(len(words)==3 and words[1]=='-c','Original shell transport is not one complete command')
        command=words[2]
    first,sep,rest=command.partition('\n')
    match=re.fullmatch(r"(.+?)\s+<<(['\"])([A-Za-z_][A-Za-z0-9_]*)\2",first)
    require(sep and match,'Original command is not a quoted Python heredoc')
    args=shlex.split(match[1]);delimiter=match[3]
    require(args and re.fullmatch(r'python(?:[0-9]+(?:\.[0-9]+)*)?',args[0].rsplit('/',1)[-1]) and
            args[-1]=='-' and set(args[1:-1])=={'-I','-S','-B'} and len(args)==5 and
            rest.endswith('\n'+delimiter) and '\n'+delimiter+'\n' not in rest,
            'Original Python transport is not isolated or has additional shell commands')
    return rest[:-(len(delimiter)+1)]


class _TraceUnrelatedTransport(ValueError):
    """Consistent unsupported transport that declares no bound target report."""


def _trace_report_targets(report,path,*,canonical_sha256=None,file_sha256=None):
    if not path or not isinstance(report,dict):return False
    def hints(row):
        if not isinstance(row,dict):return False
        markers={'sha256','file_sha256','mode','canonical_verified','readonly_mode',
                 'canonical_seal_independently_verified','raw_written_bytes_independently_verified',
                 'chmod_444_after_final_write_verified'}
        return (row.get('path')==path or any(row.get(k)==path for k in ('target_path','output_path')) or
                canonical_sha256 is not None and row.get('sha256')==canonical_sha256 or
                file_sha256 is not None and row.get('file_sha256')==file_sha256 or
                not isinstance(row.get('path'),str) and bool(markers & set(row)))
    return hints(report) or isinstance(report.get('artifacts'),list) and any(hints(row) for row in report['artifacts'])


def _trace_command_stdout(item, *, target_path=None,canonical_sha256=None,file_sha256=None):
    """Separate authenticated source and stdout; never search a heredoc for output."""
    commands=[item[k] for k in ('command','cmd') if k in item]
    outputs=[item[k] for k in ('stdout','standard_output') if k in item]
    require(all(isinstance(c,str) and c.strip() for c in commands) and all(isinstance(o,str) for o in outputs),
            'Original command/stdout declarations are malformed')
    require(not outputs or all(_strict_equal(o,outputs[0]) for o in outputs),
            'Original stdout aliases contradict')
    sources=[]
    for command in commands:
        try:sources.append(_python_trace_source(command))
        except (ValueError,TypeError):sources.append(None)
    require(not commands or (all(s is not None for s in sources) and all(_strict_equal(s,sources[0]) for s in sources)) or
            (all(s is None for s in sources) and all(_strict_equal(c,commands[0]) for c in commands)),
            'Original command aliases contradict')
    explicit_report=None
    if outputs:
        try:explicit_report=json.loads(outputs[0])
        except (ValueError,TypeError):
            require(not outputs[0].lstrip().startswith('{'),'Original declared JSON stdout is malformed')
    declared_target=_trace_report_targets(explicit_report,target_path,canonical_sha256=canonical_sha256,file_sha256=file_sha256)
    candidates=[]
    if 'text' in item:
        text=item['text'];require(isinstance(text,str),'Original command text is malformed')
        # T3 activity displays one shell-quoted command followed by stdout. A
        # candidate is accepted only when the whole command decodes, including
        # the final heredoc delimiter, and all remaining stdout is one JSON value.
        for boundary in (m.start() for m in re.finditer('\n',text)):
            command=text[:boundary];stdout=text[boundary+1:]
            if not stdout.lstrip().startswith('{'):continue
            try:source=_python_trace_source(command)
            except (ValueError,TypeError):continue
            report=json.loads(stdout)
            if isinstance(report,dict):candidates.append((source,stdout,report))
        require(len(candidates)<=1,'Original combined transport has repeated command/stdout boundaries')
    if bool(commands)!=bool(outputs):
        require(not candidates and not declared_target and not any(s is not None for s in sources),
                'Original explicit command/stdout transport is incomplete')
        raise _TraceUnrelatedTransport('Unrelated unsupported incomplete transport')
    if commands:
        if any(s is None for s in sources) or not isinstance(explicit_report,dict):
            require(not candidates and not declared_target,'Declared original target lacks complete supported command/stdout transport')
            raise _TraceUnrelatedTransport('Unrelated consistent unsupported command/stdout transport')
        report=explicit_report
        require('text' not in item or candidates,'Original combined text contradicts supported explicit transport')
        if candidates:require(_strict_equal(candidates[0][0],sources[0]) and _strict_equal(candidates[0][1],outputs[0]),
                              'Original text and command/stdout aliases contradict')
        else:candidates.append((sources[0],outputs[0],report))
    if not candidates:raise _TraceUnrelatedTransport('Unrelated unsupported transport lacks authenticated JSON stdout')
    return candidates[0][0],candidates[0][2]


def _trace_ast(source, *, creation=False):
    """Small auditable source language: no dynamic execution or external mutation in verification."""
    tree=ast.parse(source)
    function_returns={node for function in tree.body if isinstance(function,ast.FunctionDef)
                      for node in ast.walk(function) if isinstance(node,ast.Return)}
    grammar={
        ast.Module:{'body','type_ignores'},ast.Import:{'names'},ast.ImportFrom:{'module','names','level'},
        ast.alias:{'name','asname'},ast.FunctionDef:{'name','args','body','decorator_list','returns','type_comment','type_params'},
        ast.arguments:{'posonlyargs','args','vararg','kwonlyargs','kw_defaults','kwarg','defaults'},
        ast.arg:{'arg','annotation','type_comment'},ast.Assign:{'targets','value','type_comment'},
        ast.Expr:{'value'},ast.Assert:{'test','msg'},ast.If:{'test','body','orelse'},
        ast.For:{'target','iter','body','orelse','type_comment'},ast.With:{'items','body','type_comment'},
        ast.withitem:{'context_expr','optional_vars'},ast.Return:{'value'},
        ast.Constant:{'value','kind'},ast.Name:{'id','ctx'},ast.List:{'elts','ctx'},ast.Tuple:{'elts','ctx'},
        ast.Dict:{'keys','values'},ast.Subscript:{'value','slice','ctx'},ast.Attribute:{'value','attr','ctx'},
        ast.Call:{'func','args','keywords'},ast.keyword:{'arg','value'},
        ast.DictComp:{'key','value','generators'},ast.ListComp:{'elt','generators'},
        ast.GeneratorExp:{'elt','generators'},ast.SetComp:{'elt','generators'},
        ast.comprehension:{'target','iter','ifs','is_async'},ast.Compare:{'left','ops','comparators'},
        ast.BoolOp:{'op','values'},ast.UnaryOp:{'op','operand'},ast.BinOp:{'left','op','right'},
        **{node:set() for node in (ast.Load,ast.Store,ast.Eq,ast.NotEq,ast.Is,ast.IsNot,ast.In,ast.NotIn,
                                  ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.And,ast.Or,ast.Not,ast.Add,ast.Div)}}
    for node in ast.walk(tree):
        require(type(node) in grammar and set(node._fields)<=grammar[type(node)],
                'Trace source has unsupported Python grammar or fields')
        require(not getattr(node,'type_comment',None),'Trace source has unsupported type metadata')
        if isinstance(node,ast.Module):require(not node.type_ignores,'Trace source has unsupported type-ignore metadata')
        if isinstance(node,ast.Constant):require(type(node.value) in (str,bytes,int,float,bool,type(None)),
                                                'Trace source has unsupported constant type')
        if isinstance(node,ast.FunctionDef):
            require(node.returns is None and not getattr(node,'type_params',[]),
                    'Trace source has unsupported return annotation or type parameters')
        if isinstance(node,ast.arguments):
            require(not node.posonlyargs and not node.defaults and node.vararg is None and node.kwarg is None and
                    not node.kwonlyargs and not node.kw_defaults,
                    'Trace source has unsupported function argument grammar')
            require(len({a.arg for a in node.args})==len(node.args),'Trace source repeats function parameter names')
        if isinstance(node,ast.arg):require(node.annotation is None,'Trace source has unsupported parameter annotation')
        if isinstance(node,ast.Return):require(node in function_returns,'Trace source returns outside a supported function')
        if isinstance(node,ast.Call):
            keys=[k.arg for k in node.keywords]
            require(None not in keys and len(set(keys))==len(keys),'Trace source expands or repeats call keywords')
        if isinstance(node,ast.comprehension):require(node.is_async==0,'Trace source has unsupported asynchronous comprehension')
    modules={'copy','hashlib','json','stat'}|({'os'} if creation else set())
    builtins={'str','len','set','all','any','sorted','enumerate','range','isinstance','print'}
    names=set(builtins)
    imported_names=set()
    functions=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            require(all(a.name in modules and (a.asname is None or a.asname.isidentifier()) for a in node.names),
                    'Trace source imports unsupported execution or filesystem capability')
            imported_names.update(a.asname or a.name for a in node.names)
        if isinstance(node,ast.ImportFrom):
            require(node.module=='pathlib' and node.level==0 and len(node.names)==1 and
                    node.names[0].name=='Path' and (node.names[0].asname is None or node.names[0].asname.isidentifier()),
                    'Trace source imports unsupported capability')
            imported_names.add(node.names[0].asname or 'Path')
        if isinstance(node,(ast.ClassDef,ast.AsyncFunctionDef,ast.Lambda,ast.Try,ast.While,ast.Delete,ast.Global,ast.Nonlocal)):
            require(False,'Trace source has unsupported dynamic execution structure')
        if isinstance(node,ast.FunctionDef):
            require(creation and node in tree.body and node.name=='seal' and not node.decorator_list and not functions,
                    'Trace source has unsupported nested or repeated deferred execution')
            functions.add(node.name)
        if isinstance(node,ast.Attribute):
            require(not node.attr.startswith('_') and not isinstance(node.ctx,ast.Store),
                    'Trace source uses hidden or mutable object capabilities')
    allowed_methods={'read_bytes','stat','hexdigest','encode','strip','casefold','split','rstrip','join','append','pop','items','keys','values','deepcopy',
                     'loads','dumps','sha256','S_IMODE'}|({'open','write','flush','fileno','fsync','chmod','lexists'} if creation else set())
    allowed_module_calls={'copy':{'deepcopy'},'hashlib':{'sha256'},'json':{'loads','dumps'},'stat':{'S_IMODE'},
                          'os':{'fsync','chmod'}}
    require(not imported_names & (set(builtins)|functions),'Trace source imports over an execution capability')
    names.update(imported_names)
    for node in ast.walk(tree):
        if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store):
            require(node.id not in names|functions,'Trace source rebinds an execution capability')
        if not isinstance(node,ast.Call):continue
        if isinstance(node.func,ast.Name):require(node.func.id in builtins|imported_names|{'Path'}|functions,'Trace source calls an unverified alias')
        elif isinstance(node.func,ast.Attribute):
            require(node.func.attr in allowed_methods,'Trace source performs unsupported or write-capable operation')
            if isinstance(node.func.value,ast.Name) and node.func.value.id in modules:
                require(node.func.attr in allowed_module_calls[node.func.value.id],'Trace source calls unsupported imported capability')
        else:require(False,'Trace source calls a dynamic capability')
    return tree


class _TraceBytes(bytes):
    def __new__(cls,value,path=None,kind=None):
        obj=super().__new__(cls,value);obj.path=path;obj.kind=kind;return obj


class _TraceDigest(str):
    def __new__(cls,value,path=None,kind=None):
        obj=super().__new__(cls,value);obj.path=path;obj.kind=kind;return obj


class _TraceSeal(str):
    """A file's own seal is not an independently declared expected seal."""


class _TraceLiteralString(str):
    """Only an actual source constant can anchor an expected hash."""


class _TraceLiteralInt(int):
    """Only an actual source constant can anchor the expected target mode."""


class _TraceReturn(Exception):
    def __init__(self,value):self.value=value


class _TraceDump(str):
    def __new__(cls,value,path=None):
        obj=super().__new__(cls,value);obj.path=path;return obj


class _TraceDict(dict):
    def __init__(self,value=(),path=None):super().__init__(value);self.path=path
    def __getitem__(self,key):
        value=super().__getitem__(key)
        return _TraceSeal(value) if key=='sha256' and isinstance(value,str) else value


class _TraceDictView:
    """Retain a native live view, including its iteration and comparison rules."""
    def __init__(self,value,path=None):self.value=value;self.path=path
    def __iter__(self):
        require(len(self.value)<=2000,'Trace dictionary view iteration exceeds its finite bound')
        return iter(self.value)
    def __len__(self):return len(self.value)
    def __contains__(self,key):return key in self.value
    @staticmethod
    def _native(other):return other.value if isinstance(other,_TraceDictView) else other
    def __eq__(self,other):return self.value==self._native(other)
    def __ne__(self,other):return self.value!=self._native(other)
    def __lt__(self,other):return self.value<self._native(other)
    def __le__(self,other):return self.value<=self._native(other)
    def __gt__(self,other):return self.value>self._native(other)
    def __ge__(self,other):return self.value>=self._native(other)


class _TracePath:
    def __init__(self,path):
        require(isinstance(path,(str,_TracePath)),'Trace Path input is not concretely bound')
        self.path=path.path if isinstance(path,_TracePath) else str(PurePosixPath(path))
        require('..' not in PurePosixPath(self.path).parts,'Trace Path contains unsupported traversal')
    def __eq__(self,other):return isinstance(other,_TracePath) and self.path==other.path
    def __hash__(self):return hash(self.path)


class _TraceMode(int):
    def __new__(cls,value,path):obj=super().__new__(cls,value);obj.path=path;return obj


class _TraceUnknownMode:
    def __init__(self,path):self.path=path


_TRACE_EXECUTION_VALUES = {
    'str':str, 'len':len, 'set':set, 'all':all, 'any':any, 'sorted':sorted,
    'enumerate':enumerate, 'range':range, 'isinstance':isinstance, 'print':print,
    'Path':_NativePath, 'copy':copy, 'hashlib':hashlib, 'json':json, 'stat':stat, 'os':os,
    'copy.deepcopy':copy.deepcopy, 'hashlib.sha256':hashlib.sha256,
    'json.loads':json.loads, 'json.dumps':json.dumps, 'stat.S_IMODE':stat.S_IMODE,
    'os.path':os.path, 'os.path.lexists':os.path.lexists, 'os.chmod':os.chmod, 'os.fsync':os.fsync,
}


class _TraceCapability:
    """Static execution values compare as originals; opaque capabilities fail closed."""
    def __init__(self,name,value=None):self.name=name;self.value=value
    def execution_value(self):
        require(self.value is None and self.name in _TRACE_EXECUTION_VALUES,
                'Unsupported opaque capability value operation: '+self.name)
        return _TRACE_EXECUTION_VALUES[self.name]
    def __eq__(self,other):
        return self.execution_value()==(other.execution_value() if isinstance(other,_TraceCapability) else other)
    def __ne__(self,other):return not self.__eq__(other)
    def __hash__(self):return hash(self.execution_value())
    def __bool__(self):return bool(self.execution_value())
    def __deepcopy__(self,memo):
        native=self.execution_value()
        require(copy.deepcopy(native) is native,'Unsupported mutable capability deepcopy')
        return self


class _TraceFunction:
    """Each supported function is an atomic identity value, including on copy."""
    def __init__(self,node):self.node=node
    def __deepcopy__(self,memo):return self


class _TraceIterator:
    """One deferred finite iterator over interpreter operations, never source code."""
    def __init__(self,iterator,execution_type=None):self.iterator=iterator;self.execution_type=execution_type
    def __iter__(self):return self
    def __next__(self):return next(self.iterator)


class _TraceScope(dict):
    """Locals shadow a referenced outer scope; globals are never copied values."""
    def __init__(self,parent):super().__init__();self.parent=parent
    def __contains__(self,name):return dict.__contains__(self,name) or name in self.parent
    def __getitem__(self,name):
        return dict.__getitem__(self,name) if dict.__contains__(self,name) else self.parent[name]


def _trace_local_names(nodes):
    """Lexical bindings in the finite grammar; comprehensions own their targets."""
    names=set()
    def visit(node):
        if isinstance(node,(ast.DictComp,ast.ListComp,ast.GeneratorExp,ast.SetComp)):return
        if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store):names.add(node.id)
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names.update(alias.asname or alias.name for alias in node.names)
        if isinstance(node,ast.FunctionDef):names.add(node.name);return
        for child in ast.iter_child_nodes(node):visit(child)
    for node in nodes:visit(node)
    return names


def _trace_virtual_inputs(artifact,evidence):
    """Only retained artifact bytes and separately bound original contract/helper bytes."""
    files={};originals={}
    def retain(info):
        require(isinstance(info,dict) and isinstance(info.get('path'),str) and
                PurePosixPath(info['path']).is_absolute() and '..' not in PurePosixPath(info['path']).parts and
                isinstance(info.get('utf8'),str) and hashlib.sha256(info['utf8'].encode()).hexdigest()==info.get('file_sha256'),
                'Trace virtual input lacks its exact bound original path/bytes')
        path=str(PurePosixPath(info['path']));data=info['utf8'].encode()
        require(path not in files or files[path]==data,'Trace input aliases bind contradictory original bytes')
        files[path]=data
    retain(evidence['receipt_file'])
    contexts=evidence.get('trace_input_artifacts',{})
    require(isinstance(contexts,dict) and set(contexts)<={'input_packet','base_authored','response'},
            'Trace original artifact context is unsupported')
    for key,pair in contexts.items():
        original=checked(pair['artifact']);raw_artifact(original,pair['artifact_file']);retain(pair['artifact_file']);originals[key]=original
    external=evidence.get('original_trace_input_files',{})
    require(isinstance(external,dict) and set(external)<={'author_output_contract','packing_helper'},
            'Trace caller custody adapter has unsupported inputs')
    receipt=originals.get('response',artifact)
    if external:
        require('author_output_contract' in external and isinstance(receipt.get('author_output_contract'),dict),
                'Trace caller inputs are not bound by the original response contract declaration')
        expected=receipt['author_output_contract'];info=external['author_output_contract']
        require(isinstance(info,dict) and set(info)<={'path','utf8','file_sha256','canonical_sha256','sha256'} and
                info.get('path')==expected.get('path') and info.get('file_sha256')==expected.get('file_sha256'),
                'Original trace contract path/raw hash differs from the original response')
        contract=checked(json.loads(info['utf8']))
        require(contract['sha256']==expected.get('sha256') and all(expected[k]==contract['sha256'] for k in
                ('canonical_sha256','sha256') if k in expected) and all(info[k]==contract['sha256'] for k in
                ('canonical_sha256','sha256') if k in info),
                'Original trace contract canonical declarations differ from the response')
        retain(info)
        if 'packing_helper' in external:
            helper=external['packing_helper']
            require(isinstance(helper,dict) and set(helper)<={'path','utf8','file_sha256'} and
                    helper.get('path')==contract.get('authorized_offline_packing_script_path') and
                    helper.get('file_sha256')==contract.get('authorized_offline_packing_script_file_sha256') and
                    not any(k in helper for k in ('canonical_sha256','sha256')),
                    'Original trace helper path/raw hash differs from its bound original contract')
            validation=receipt.get('packing_validation',{})
            require('helper_file_sha256_verified_before_execution' not in validation or
                    validation['helper_file_sha256_verified_before_execution']==helper['file_sha256'],
                    'Original response/helper hash declarations contradict')
            retain(helper)
    return files


def _trace_dataflow(tree,artifact,file_info,files,report,*,creation,future_output_paths=()):
    """Interpret a bounded AST over virtual bytes only; never compile/exec source.

    This establishes the target slice, not unrelated file modes or total I/O.
    All filesystem operations are represented in memory. Unsupported values or
    control flow fail closed; unrelated unknown mode assertions confer nothing.
    """
    target=file_info['path'];expected=file_info['utf8'].encode();virtual=dict(files)
    if creation:
        for path in (target,*future_output_paths):virtual.pop(path,None)
    env={};printed=[];checks=set();writes=[];chmods=[];open_handles=set();budget=40000
    builtins={'str','len','set','all','any','sorted','enumerate','range','isinstance','print'}
    uninitialized=object()
    def plain(value):
        if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [plain(v) for v in value] if isinstance(value,list) else tuple(plain(v) for v in value)
        if isinstance(value,str):return str(value)
        if isinstance(value,int):return int(value) if type(value) is not bool else value
        return value
    def require_text_value(value):
        require(not isinstance(value,(_TraceCapability,_TraceFunction,_TraceIterator,_TraceDictView,_TracePath,_TraceUnknownMode)),
                'Trace string conversion depends on unsupported execution-value representation')
        if isinstance(value,dict):
            for key,item in value.items():require_text_value(key);require_text_value(item)
        elif isinstance(value,(list,tuple,set)):
            for item in value:require_text_value(item)
    def tick():
        nonlocal budget
        budget-=1;require(budget>=0,'Trace dataflow exceeds its finite validation budget')
    def bind(node,value,scope):
        if isinstance(node,ast.Name):scope[node.id]=value
        elif isinstance(node,(ast.Tuple,ast.List)):
            require(isinstance(value,(list,tuple)) and len(node.elts)==len(value),'Trace unpacking is not concretely bound')
            for child,item in zip(node.elts,value):bind(child,item,scope)
        elif isinstance(node,ast.Subscript):expr(node.value,scope)[expr(node.slice,scope)]=value
        else:require(False,'Unsupported trace assignment')
    def anchor(left,right):
        if isinstance(left,(tuple,list)) and isinstance(right,(tuple,list)) and len(left)==len(right):
            for a,b in zip(left,right):anchor(a,b)
        if isinstance(left,_TraceDigest) and left.path==target and isinstance(right,_TraceLiteralString):
            if left.kind=='canonical' and right==artifact['sha256']:checks.add('canonical')
            if left.kind=='raw' and right==file_info['file_sha256']:checks.add('raw')
        if isinstance(left,_TraceMode) and left.path==target and isinstance(right,_TraceLiteralInt) and right==0o444:checks.add('mode')
    def compare(node,scope,record=False):
        values=[expr(node.left,scope)]
        for op,right_node in zip(node.ops,node.comparators):
            left=values[-1];right=expr(right_node,scope);values.append(right)
            if any(isinstance(v,_TraceUnknownMode) for v in (left,right)):
                return _TraceUnknownMode(next(v.path for v in (left,right) if isinstance(v,_TraceUnknownMode)))
            if isinstance(op,ast.Eq):ok=left==right
            elif isinstance(op,ast.NotEq):ok=left!=right
            elif isinstance(op,(ast.Is,ast.IsNot)):
                require(left is None or right is None or type(left) is bool or type(right) is bool,
                        'Trace identity comparison depends on unsupported implementation identity')
                ok=left is right if isinstance(op,ast.Is) else left is not right
            elif isinstance(op,ast.In):ok=left in right
            elif isinstance(op,ast.NotIn):ok=left not in right
            elif isinstance(op,ast.Lt):ok=left<right
            elif isinstance(op,ast.LtE):ok=left<=right
            elif isinstance(op,ast.Gt):ok=left>right
            elif isinstance(op,ast.GtE):ok=left>=right
            else:require(False,'Unsupported original trace comparison')
            if not ok:return False
        if record:
            # Transitive chained equality also binds the independently declared
            # final literal to its preceding computed digest.
            if all(isinstance(op,ast.Eq) for op in node.ops):
                for i,left in enumerate(values):
                    for right in values[i+1:]:anchor(left,right);anchor(right,left)
        return True
    def truth(value):
        require(not isinstance(value,_TraceUnknownMode),'Unknown unrelated mode cannot control target execution')
        return bool(value)
    def iterator(value):
        require(isinstance(value,(list,tuple,dict,set,str,range,_TraceIterator,_TraceDictView)), 'Trace iteration is not finitely bound')
        if not isinstance(value,_TraceIterator):require(len(value)<=2000,'Trace iteration exceeds its finite bound')
        source_iterator=iter(value)
        def bounded():
            for index,item in enumerate(source_iterator):
                tick();require(index<2000,'Trace iterator exceeds its finite bound');yield item
        return _TraceIterator(bounded())
    def comprehension(node,scope):
        origins=set();local_names=_trace_local_names([gen.target for gen in node.generators])
        first_value=expr(node.generators[0].iter,scope)
        first_iterator=iterator(first_value)
        if isinstance(first_value,_TraceDictView) and first_value.path:origins.add(first_value.path)
        def walk(index,current):
            if index==len(node.generators):
                yield (expr(node.key,current),expr(node.value,current)) if isinstance(node,ast.DictComp) else expr(node.elt,current)
                return
            gen=node.generators[index];require(not gen.is_async,'Unsupported asynchronous trace comprehension')
            iterable=first_value if index==0 else expr(gen.iter,current)
            if isinstance(iterable,_TraceDictView) and iterable.path:origins.add(iterable.path)
            for value in first_iterator if index==0 else iterator(iterable):
                bind(gen.target,value,current)
                if all(truth(expr(c,current)) for c in gen.ifs):yield from walk(index+1,current)
        def deferred():
            local=_TraceScope(scope);local.update({name:uninitialized for name in local_names})
            yield from walk(0,local)
        if isinstance(node,ast.GeneratorExp):return _TraceIterator(deferred())
        values=_TraceDict() if isinstance(node,ast.DictComp) else set() if isinstance(node,ast.SetComp) else []
        for value in iterator(_TraceIterator(deferred())):
            if isinstance(node,ast.DictComp):values[value[0]]=value[1]
            elif isinstance(node,ast.SetComp):values.add(value)
            else:values.append(value)
        if isinstance(node,ast.DictComp):values.path=next(iter(origins)) if len(origins)==1 else None
        return values
    def call(cap,args,kwargs,scope):
        if isinstance(cap,_TraceFunction):
            fn=cap.node;require(not kwargs and len(args)==len(fn.args.args) and not fn.args.defaults and
                not fn.args.vararg and not fn.args.kwarg and not fn.args.kwonlyargs,'Unsupported trace function arguments')
            local=_TraceScope(env)
            local.update({name:uninitialized for name in _trace_local_names(fn.body)})
            for parameter,value in zip(fn.args.args,args):local[parameter.arg]=value
            try:statements(fn.body,local)
            except _TraceReturn as result:return result.value
            return None
        require(isinstance(cap,_TraceCapability),'Trace calls an unbound execution capability')
        name=cap.name;value=cap.value
        if name=='Path':require(len(args)==1 and not kwargs,'Unsupported Path arguments');return _TracePath(args[0])
        if name=='str':
            require(len(args)<=1 and not kwargs,'Unsupported trace str arguments')
            if args and not isinstance(args[0],_TracePath):require_text_value(args[0])
            return args[0].path if args and isinstance(args[0],_TracePath) else str(args[0]) if args else ''
        if name in ('len','set','all','any','sorted','enumerate','range','isinstance'):
            require(not kwargs,'Unsupported trace builtin keyword arguments')
            require(len(args)==1 if name in ('len','set','all','any','sorted') else
                    len(args) in (1,2) if name=='enumerate' else len(args) in (1,2,3) if name=='range' else len(args)==2,
                    'Unsupported trace builtin positional arguments')
            if name=='len':return len(args[0])
            if name=='set':return set(iterator(args[0]))
            if name=='all':return all(truth(v) for v in iterator(args[0]))
            if name=='any':return any(truth(v) for v in iterator(args[0]))
            if name=='sorted':return sorted(iterator(args[0]))
            if name=='enumerate':
                return _TraceIterator(enumerate(iterator(args[0]),args[1] if len(args)==2 else 0),enumerate)
            if name=='range':result=range(*args);require(len(result)<=2000,'Trace range exceeds its bound');return result
            if name=='isinstance':
                def classinfo(value):
                    if isinstance(value,tuple):return tuple(classinfo(v) for v in value)
                    require(isinstance(value,_TraceCapability),'Trace isinstance class is not a supported type')
                    native=value.execution_value()
                    require(isinstance(native,type),'Trace isinstance class is not a type')
                    return _TracePath if native is _NativePath else native
                value=args[0].execution_value() if isinstance(args[0],_TraceCapability) else args[0]
                expected_type=classinfo(args[1])
                if isinstance(value,_TraceIterator) and value.execution_type is not None:
                    return issubclass(value.execution_type,expected_type)
                return isinstance(value,expected_type)
        if name=='print':
            require(len(args)==1 and not kwargs,'Unsupported original trace output')
            require_text_value(args[0]);printed.append(str(args[0]));return None
        if name=='json.loads':
            require(len(args)==1 and not kwargs,'Unsupported original JSON parsing')
            parsed=json.loads(args[0]);return _TraceDict(parsed,args[0].path) if isinstance(parsed,dict) and isinstance(args[0],_TraceBytes) else parsed
        if name=='json.dumps':
            require(len(args)==1 and set(kwargs)<={'sort_keys','separators','ensure_ascii','indent'},'Unsupported JSON serialization arguments')
            result=json.dumps(plain(args[0]),**kwargs);origin=getattr(args[0],'path',None)
            canonical=None
            if origin in virtual and isinstance(args[0],dict):
                original=json.loads(virtual[origin]);projection={k:v for k,v in original.items() if k!='sha256'}
                if (_strict_equal(plain(args[0]),projection) and kwargs.get('sort_keys') is True and
                        kwargs.get('separators')==(',',':') and kwargs.get('ensure_ascii',True) is False):canonical=origin
            return _TraceDump(result,canonical)
        if name=='hashlib.sha256':
            require(len(args)==1 and not kwargs and isinstance(args[0],bytes),'Unsupported digest input')
            return _TraceCapability('digest',args[0])
        if name=='digest.hexdigest':
            require(not args and not kwargs,'Unsupported digest output')
            return _TraceDigest(hashlib.sha256(value).hexdigest(),getattr(value,'path',None),getattr(value,'kind',None))
        if name=='copy.deepcopy':require(len(args)==1 and not kwargs,'Unsupported deepcopy arguments');return copy.deepcopy(args[0])
        if name=='stat.S_IMODE':
            require(len(args)==1 and not kwargs and isinstance(args[0],(_TraceMode,_TraceUnknownMode)),'Unsupported stat mode arguments')
            return args[0] if isinstance(args[0],_TraceUnknownMode) else _TraceMode(int(args[0])&0o7777,args[0].path)
        if name=='os.path.lexists':
            require(creation and len(args)==1 and not kwargs and isinstance(args[0],_TracePath),'Unsupported lexists arguments')
            return args[0].path in virtual
        if name=='os.chmod':
            require(creation and len(args)==2 and isinstance(args[0],_TracePath) and args[0].path==target and
                    type(args[1]) in (int,_TraceLiteralInt) and args[1]==0o444 and not kwargs and target in virtual and virtual[target]==expected and
                    target not in open_handles and len(chmods)==0,
                    'Trace chmod is not bound to exact retained bytes after final target write')
            chmods.append(target);return None
        if name=='os.fsync':require(creation and len(args)==1 and not kwargs and args[0]==('virtual-fd',target),'Unsupported virtual fsync');return None
        if name.startswith('path.'):
            require(isinstance(value,_TracePath),'Unbound original filesystem target');path=value.path
            if name=='path.read_bytes':
                require(not args and not kwargs and path in virtual,'Trace read path lacks exact original custody bytes')
                return _TraceBytes(virtual[path],path,'raw')
            if name=='path.stat':
                require(not args and not kwargs and path in virtual,'Trace stat path is not bound')
                return _TraceCapability('stat-result',_TraceMode(0o444,path) if path==target and (not creation or chmods) else _TraceUnknownMode(path))
            if name=='path.open':
                require(creation and path==target and args==['xb'] and not kwargs and path not in virtual and not chmods,
                        'Trace creation is not one exclusive open of the exact retained target')
                virtual[path]=b'';open_handles.add(path);return _TraceCapability('handle',path)
        if name.startswith('handle.'):
            require(creation and value==target and value in open_handles,'Unbound virtual write handle')
            if name=='handle.write':
                require(len(args)==1 and isinstance(args[0],bytes) and not kwargs and not chmods,'Unsupported or late trace write')
                candidate=virtual[target]+bytes(args[0])
                require(expected.startswith(candidate),'Trace write payload differs from the exact retained artifact bytes')
                virtual[target]=candidate;writes.append(bytes(args[0]));return len(args[0])
            if name=='handle.flush':require(not args and not kwargs,'Unsupported flush');return None
            if name=='handle.fileno':require(not args and not kwargs,'Unsupported fileno arguments');return ('virtual-fd',target)
        if name.startswith('memory.'):
            method=name.split('.',1)[1]
            def signature(parameters,required,*,positional_only=False):
                require(len(args)<=len(parameters) and set(kwargs)<=set(parameters) and
                        not set(parameters[:len(args)])&set(kwargs) and
                        (not positional_only or not kwargs) and
                        all(i<len(args) or parameter in kwargs for i,parameter in enumerate(parameters[:required])),
                        'Unsupported in-memory trace argument shape: '+method)
            if method in ('items','keys','values') and isinstance(value,dict):
                require(not args and not kwargs,'Unsupported dictionary view arguments')
                return _TraceDictView(getattr(value,method)(),getattr(value,'path',None))
            if method=='encode' and isinstance(value,str):
                signature(('encoding','errors'),0)
                data=str(value).encode(*args,**kwargs);return _TraceBytes(data,value.path,'canonical') if isinstance(value,_TraceDump) and value.path else data
            require(method in ('strip','casefold','split','rstrip','join','append','pop','keys','values') and
                    isinstance(value,(str,list,dict)),'Unsupported in-memory trace capability')
            if isinstance(value,str):
                require(method in ('strip','casefold','split','rstrip','join'),'Unsupported string method')
                if method in ('strip','rstrip'):signature(('chars',),0,positional_only=True)
                elif method=='casefold':signature((),0,positional_only=True)
                elif method=='split':signature(('sep','maxsplit'),0)
                else:signature(('iterable',),1,positional_only=True)
            elif isinstance(value,list):
                require(method in ('append','pop'),'Unsupported list method')
                signature(('object',) if method=='append' else ('index',),1 if method=='append' else 0,positional_only=True)
            else:
                require(method in ('pop','keys','values'),'Unsupported dictionary method')
                signature(('key','default') if method=='pop' else (),1 if method=='pop' else 0,positional_only=True)
            return getattr(value,method)(*args,**kwargs)
        require(False,'Unsupported bound trace capability: '+name)
    def expr(node,scope,record=False):
        tick()
        if isinstance(node,ast.Constant):
            if type(node.value) is str:return _TraceLiteralString(node.value)
            if type(node.value) is int:return _TraceLiteralInt(node.value)
            return node.value
        if isinstance(node,ast.Name):
            require(node.id in scope or node.id in builtins,'Trace expression depends on an unbound value: '+node.id)
            value=scope[node.id] if node.id in scope else _TraceCapability(node.id)
            require(value is not uninitialized,'Trace expression reads an uninitialized lexical local: '+node.id)
            return value
        if isinstance(node,ast.List):return [expr(n,scope) for n in node.elts]
        if isinstance(node,ast.Tuple):return tuple(expr(n,scope) for n in node.elts)
        if isinstance(node,ast.Dict):
            require(all(k is not None for k in node.keys),'Unsupported dictionary expansion')
            return {expr(k,scope):expr(v,scope) for k,v in zip(node.keys,node.values)}
        if isinstance(node,ast.Subscript):return expr(node.value,scope)[expr(node.slice,scope)]
        if isinstance(node,ast.Attribute):
            value=expr(node.value,scope)
            if isinstance(value,_TraceCapability):
                if value.name=='stat-result' and node.attr=='st_mode':return value.value
                if value.name=='digest' and node.attr=='hexdigest':return _TraceCapability('digest.hexdigest',value.value)
                if value.name=='handle':
                    require(node.attr in ('write','flush','fileno'),'Unsupported virtual handle attribute')
                    return _TraceCapability('handle.'+node.attr,value.value)
                name=value.name+'.'+node.attr
                require(value.value is None and name in _TRACE_EXECUTION_VALUES,'Unsupported execution-value attribute')
                return _TraceCapability(name)
            if isinstance(value,_TracePath):
                require(node.attr in ('read_bytes','stat','open'),'Unsupported virtual Path attribute')
                return _TraceCapability('path.'+node.attr,value)
            methods={'encode','strip','casefold','split','rstrip','join'} if isinstance(value,str) else \
                    {'append','pop'} if isinstance(value,list) else {'pop','items','keys','values'} if isinstance(value,dict) else set()
            require(node.attr in methods,'Unsupported in-memory execution-value attribute')
            return _TraceCapability('memory.'+node.attr,value)
        if isinstance(node,ast.Call):
            require(all(k.arg is not None for k in node.keywords),'Unsupported keyword expansion')
            cap=expr(node.func,scope);args=[expr(a,scope) for a in node.args];kwargs={k.arg:expr(k.value,scope) for k in node.keywords}
            return call(cap,args,kwargs,scope)
        if isinstance(node,(ast.DictComp,ast.ListComp,ast.GeneratorExp,ast.SetComp)):
            return comprehension(node,scope)
        if isinstance(node,ast.Compare):return compare(node,scope,record)
        if isinstance(node,ast.BoolOp):
            result=True if isinstance(node.op,ast.And) else False
            for value in node.values:
                # An OR alternative can bypass a false expected-value check;
                # its comparisons cannot establish enforced assertion anchors.
                result=expr(value,scope,record and isinstance(node.op,ast.And))
                if isinstance(node.op,ast.And) and not truth(result) or isinstance(node.op,ast.Or) and truth(result):break
            return result
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not):return not truth(expr(node.operand,scope))
        if isinstance(node,ast.BinOp):
            left=expr(node.left,scope);right=expr(node.right,scope)
            if isinstance(node.op,ast.Add):
                if isinstance(left,(str,bytes,list,tuple)):
                    require(isinstance(right,type(left)) or isinstance(left,str) and isinstance(right,str) or
                        isinstance(left,bytes) and isinstance(right,bytes),'Unsupported mixed trace addition')
                    require(len(left)+len(right)<=(16*1024*1024 if isinstance(left,(str,bytes)) else 2000),
                            'Trace addition exceeds its finite bound')
                else:require(type(left) in (int,_TraceLiteralInt) and type(right) in (int,_TraceLiteralInt),'Unsupported numeric trace addition')
                return left+right
            if isinstance(node.op,ast.Div) and isinstance(left,_TracePath) and isinstance(right,str):return _TracePath(str(PurePosixPath(left.path)/right))
        require(False,'Unsupported original trace expression: '+type(node).__name__)
    def statements(nodes,scope):
        for node in nodes:
            tick()
            if isinstance(node,ast.Import):
                for alias in node.names:scope[alias.asname or alias.name]=_TraceCapability(alias.name)
            elif isinstance(node,ast.ImportFrom):scope[node.names[0].asname or 'Path']=_TraceCapability('Path')
            elif isinstance(node,ast.FunctionDef):scope[node.name]=_TraceFunction(node)
            elif isinstance(node,ast.Assign):
                value=expr(node.value,scope)
                for target_node in node.targets:bind(target_node,value,scope)
            elif isinstance(node,ast.Expr):expr(node.value,scope)
            elif isinstance(node,ast.Assert):
                result=expr(node.test,scope,True)
                if isinstance(result,_TraceUnknownMode):
                    require(result.path!=target,'Target mode is not mechanically established')
                else:require(truth(result),'Original trace assertion does not hold over exact retained bytes')
            elif isinstance(node,ast.If):statements(node.body if truth(expr(node.test,scope)) else node.orelse,scope)
            elif isinstance(node,ast.For):
                iterable=iterator(expr(node.iter,scope))
                for item in iterable:bind(node.target,item,scope);statements(node.body,scope)
                require(not node.orelse,'Unsupported original loop else')
            elif isinstance(node,ast.With):
                require(creation and len(node.items)==1,'Unsupported original trace context manager')
                context=node.items[0];handle=expr(context.context_expr,scope)
                require(isinstance(handle,_TraceCapability) and handle.name=='handle','Unbound original write context')
                if context.optional_vars:bind(context.optional_vars,handle,scope)
                try:statements(node.body,scope)
                finally:open_handles.remove(handle.value)
            elif isinstance(node,ast.Return):raise _TraceReturn(expr(node.value,scope) if node.value else None)
            else:require(False,'Unsupported original trace statement: '+type(node).__name__)
        return None
    statements(tree.body,env)
    require(len(printed)==1 and _strict_equal(json.loads(printed[0]),report),'Trace stdout differs from its bound source/dataflow')
    if creation:require(writes and virtual.get(target)==expected and chmods==[target] and not open_handles,
                        'Trace creation does not write exact retained bytes to the bound readonly target')
    else:require(checks=={'canonical','raw','mode'},'Trace verification lacks executed target/expected canonical/raw/0444 comparisons')


def completed_output_binding(artifact, evidence, status):
    """Use full terminal seals or exact original ordered creation/verification transport."""
    full_terminal=artifact['sha256'] in status.get('latestTerminalSummary','')
    info=evidence.get('completion_trace_file')
    if info is None:return full_terminal
    text=info.get('utf8')
    require(isinstance(text,str) and hashlib.sha256(text.encode()).hexdigest()==info.get('file_sha256'),
            'Authenticated child write trace lacks its exact original bytes')
    raw_trace=json.loads(text);trace=json_pointer(raw_trace,info['trace_json_pointer'])
    blocks=[b for b in raw_trace.get('content',[]) if b.get('type')=='text']
    require(raw_trace.get('isError') is False and blocks and all(_strict_equal(json.loads(b['text']),trace) for b in blocks) and
            ('structuredContent' not in raw_trace or _strict_equal(raw_trace['structuredContent'],trace)),
            'Authenticated trace read envelope has contradictory original result aliases')
    thread=trace['thread'];run_id=status['childRunId']
    current_runs=[r for r in trace['recentRuns'] if r.get('runId')==run_id]
    require(thread['threadId']==status['childThreadId'] and thread['status']=='completed' and thread['latestRunId']==run_id and
            thread.get('activeRunId') is None and type(thread.get('pendingRequestCount')) is int and thread['pendingRequestCount']==0 and
            thread.get('model')==status['model'] and len(current_runs)==1 and current_runs[0].get('status')=='completed' and
            current_runs[0].get('model')==status['model'],
            'Authenticated write trace belongs to another or nonterminal child run')
    file_info=evidence['receipt_file'];path=file_info.get('path')
    require(path and path in status.get('latestTerminalSummary',''),'Original completion does not identify trace-bound output path')
    virtual_inputs=_trace_virtual_inputs(artifact,evidence)
    future_outputs=[pair['artifact_file']['path'] for key,pair in evidence.get('trace_input_artifacts',{}).items() if key=='response']
    creations=[];verifications=[];positions=set()
    for item in trace['items']:
        if (item.get('sourceThreadId')!=status['childThreadId'] or item.get('runId')!=run_id or
                item.get('type')!='command_execution' or item.get('visibility')!='local'):continue
        require(type(item.get('position')) is int and item['position']>=0 and item['position'] not in positions,
                'Original command trace positions are malformed or repeated');positions.add(item['position'])
        require(item.get('status')=='completed' and item.get('textTruncated') is False and item.get('nextTextOffset') is None and
                all(type(item[k]) is int and item[k]==0 for k in ('exit_code','exitCode','return_code','returncode') if k in item) and
                all(item[k] in ('completed','success','succeeded') for k in ('execution_status','executionStatus') if k in item) and
                all(item[k] is False for k in ('isError','failed') if k in item),
                'Original command transport is incomplete or explicitly failed')
        # Some same-run commands are unrelated to this output (e.g. packing).
        # They remain retained; only a strict complete JSON transport can qualify.
        try:source,report=_trace_command_stdout(item,target_path=path,canonical_sha256=artifact['sha256'],file_sha256=file_info['file_sha256'])
        except _TraceUnrelatedTransport:continue
        exact=lambda row:isinstance(row,dict) and row.get('path')==path and row.get('sha256')==artifact['sha256'] and row.get('file_sha256')==file_info['file_sha256']
        def declared_target(row,*,artifact_row=False):
            require(isinstance(row,dict),'Original artifact report row is malformed')
            hints=(row.get('sha256')==artifact['sha256'] or row.get('file_sha256')==file_info['file_sha256'])
            require(not any(k in row for k in ('target_path','output_path')),
                    'Original target report uses an unsupported path declaration')
            declares=(artifact_row or any(k in row for k in ('path','sha256','file_sha256','mode','canonical_verified',
                      'readonly_mode','canonical_seal_independently_verified','raw_written_bytes_independently_verified',
                      'chmod_444_after_final_write_verified')))
            if declares:
                require(isinstance(row.get('path'),str) and bool(row['path']),
                        'Original target-like report lacks its declared path')
                require(not hints or row['path']==path,'Original report declares target pins at another path')
            return row.get('path')==path
        packing='packing_output' in report
        if packing:
            # The retained offline packing summary declares artifact bindings,
            # not a creation or independent-verification execution receipt.
            require(isinstance(report['packing_output'],dict) and isinstance(report.get('patch'),dict) and
                    isinstance(report.get('response'),dict) and report.get('canonical_verified') is True and
                    type(report.get('packing_import_profile_event_count')) is int and
                    isinstance(report.get('packing_stderr_non_import_lines'),list) and
                    not any(k in report for k in ('path','sha256','file_sha256','mode','target_path','output_path','artifacts')),
                    'Original offline packing summary is incomplete or contradicts its report role')
            for row in (report['patch'],report['response']):
                require(set(row)<={'path','sha256','file_sha256','mode'} and isinstance(row.get('path'),str) and bool(row['path']) and
                        all(isinstance(row.get(k),str) and re.fullmatch('[0-9a-f]{64}',row[k]) for k in ('sha256','file_sha256')) and
                        ('mode' not in row or row['mode']=='0444'), 'Original packing artifact binding is incomplete')
            response_context=evidence.get('trace_input_artifacts',{}).get('response')
            require(isinstance(response_context,dict),'Original packing summary lacks its original response context')
            response_artifact=response_context['artifact'];response_file=response_context['artifact_file']
            require(exact(report['patch']) and report['response']['path']==response_file['path'] and
                    report['response']['sha256']==response_artifact['sha256'] and
                    report['response']['file_sha256']==response_file['file_sha256'],
                    'Original packing summary contradicts its corresponding patch or response binding')
        creation_target=False if packing else declared_target(report)
        if creation_target:
            require(exact(report) and report.get('mode')=='0444' and report.get('canonical_verified') is True,
                    'Original bound-target creation report has mismatched pins or incomplete verification')
            tree=_trace_ast(source,creation=True)
            _trace_dataflow(tree,artifact,file_info,virtual_inputs,report,creation=True,future_output_paths=future_outputs)
            creations.append(item)
        rows=report.get('artifacts',[])
        require(isinstance(rows,list),'Original artifact report inventory is malformed')
        for row in rows:
            if declared_target(row,artifact_row=True):
                require(exact(row) and row.get('canonical_seal_independently_verified') is True and
                    row.get('raw_written_bytes_independently_verified') is True and row.get('readonly_mode')=='0444' and
                    row.get('chmod_444_after_final_write_verified') is True and
                    isinstance(report.get('independent_verification'),dict) and
                    report['independent_verification'].get('verification_performed_no_writes') is True,
                    'Original bound-target artifact report has mismatched pins or incomplete verification')
                _trace_dataflow(_trace_ast(source),artifact,file_info,virtual_inputs,report,creation=False)
                verifications.append(item)
    require(len(creations)==len(verifications)==1 and creations[0]['position']<verifications[0]['position'],
            'Original child trace lacks unique ordered exact creation and read-only verification')
    return True


def _protocol(protocol):
    from .confirmation_lane import verify_exception_protocol
    verify_exception_protocol(protocol)
    require(protocol['sha256'] == ADOPTED_PROTOCOL_SHA256 and protocol['blind_audit_design_proposal']['seed'] == 2026100813,
            'Protocol text or fixed audit design differs from the actual adopted version')


def _event_context(packet, event, *, kind, record=None, authored=None):
    preservation = packet.get('source_preservation_protocol')
    if preservation is None:
        require(authored is not None and authored['sha256'] == packet['authored_chunk_sha256'],
                'Historical group context lacks its exact authored preservation source')
        preservation = source_preservation_context(authored)
    units = sorted((u for u in packet['source_units'] if u['event'] == event), key=lambda u: u['unit_id'])
    owned = set(record['source_unit_ids']) if kind == 'groups' else {u['unit_id'] for u in units}
    value = {'event': event, 'factsheet': packet['factsheets'][event], 'original_source_units': units,
             'source_dispositions': sorted((d for d in preservation['source_dispositions'] if d['unit_id'] in owned),
                                           key=lambda d: d['unit_id'])}
    if kind == 'groups':
        # A declaration's own proposition/membership/disposition is unchanged
        # by a neighboring acquisition paraphrase or a different group merger.
        value['own_group'] = record
    else:
        value.update(groups=sorted((r for r in packet['groups'] if r['event'] == event), key=lambda r: r['id']),
                     acquisition_records=sorted((r for r in packet['acquisition_records'] if r['event'] == event), key=lambda r: r['id']))
    return value


def protocol_source_snapshot(chunks, catalog):
    """Recompute exact source-only contexts; no outcomes or author rationales."""
    checked(catalog)
    require(len(chunks) == 8 and sorted(c['review_packet']['chunk_index'] for c in chunks) == list(range(1, 9)),
            'Protocol coverage needs all eight exact current source chunks')
    from .confirmation_dependency import source_binding_snapshot
    source_binding_snapshot(chunks, catalog)
    rows = []
    for chunk in sorted(chunks, key=lambda c: c['review_packet']['chunk_index']):
        packet = checked(chunk['review_packet']);authored = checked(chunk['authored'])
        require(packet['source_catalog_sha256'] == catalog['sha256'] and packet['authored_chunk_sha256'] == authored['sha256'] and
                packet.get('author_rationales_included') is False, 'Current protocol packet provenance changed')
        require(all(packet[field] == [source_review_content(r, field) for r in authored[field]]
                    for field in ('groups', 'acquisition_records')), 'Protocol snapshot changed the authored source projection')
        for kind in ('groups', 'acquisition_records'):
            for record in packet[kind]:
                context = _event_context(packet, record['event'], kind=kind, record=record, authored=authored)
                rows.append({'chunk_index': packet['chunk_index'], 'field': kind, 'id': record['id'], 'event': record['event'],
                             'content_sha256': digest(record), 'context_sha256': digest(context),
                             'packet_sha256': packet['sha256']})
    require(len(rows) == len({(r['field'], r['id']) for r in rows}), 'Duplicate current protocol coverage identity')
    return sorted(rows, key=lambda r: (r['chunk_index'], r['field'], r['id']))


def seeded_audit_plan(eligible_by_chunk, failure_record_ids=()):
    """CF5 fixed selection/escalation, also retaining a true empty population."""
    require(set(eligible_by_chunk) == set(range(1, 9)), 'Audit eligibility must include every chunk, including empty chunks')
    rng = random.Random(2026100813);eligible = {};sample = {}
    for chunk in range(1, 9):
        ids = sorted(eligible_by_chunk[chunk]);require(len(ids) == len(set(ids)), 'Repeated carryover audit identity')
        eligible[str(chunk)] = ids
        count = min(len(ids), max(30, math.ceil(.10 * len(ids))))
        sample[str(chunk)] = sorted(rng.sample(ids, count))
    failures = sorted(failure_record_ids)
    require(len(failures) == len(set(failures)) and set(failures) <= {rid for ids in sample.values() for rid in ids},
            'Audit failure is outside the frozen blind seeded sample')
    failing_chunks = [c for c, ids in sample.items() if set(ids) & set(failures)]
    escalation = sorted({rid for c, ids in eligible.items() if len(failing_chunks) >= 2 or c in failing_chunks for rid in ids})
    return {'seed': 2026100813, 'eligible_by_chunk': eligible, 'sample_by_chunk': sample,
            'failure_record_ids': failures, 'escalated_record_ids': escalation,
            'empty_population': not any(eligible.values()), 'sampling_without_replacement': True}


def _same_binding(receipt, key, artifact, info, aliases=()):
    """Truthful first-pass aliases, validating every declaration that is present."""
    found = False
    declared_aliases = {'first_pass': ('b1_receipt', 'first_pass_receipt', 'b1_first_pass'),
                        'prior_review_supplement': ('prior_review_evidence', 'prior_history', 'history_supplement',
                                                   'prior_review_history')}
    keys = tuple(dict.fromkeys((key, *declared_aliases.get(key, ()), *aliases)))
    require('bindings' not in receipt or isinstance(receipt['bindings'],dict), 'Actual B2 binding container is malformed')
    containers = [receipt] + ([receipt['bindings']] if isinstance(receipt.get('bindings'), dict) else [])
    seal_names = ('sha256', 'canonical_sha256', 'content_sha256', 'payload_sha256', 'payload_seal',
                  'payload_seal_sha256', 'seal', 'seal_sha256', 'self_seal')
    for container in containers:
        for name in (name for k in keys for name in (k, k + '_binding')):
            if name not in container:continue
            value = container[name]
            require(isinstance(value, dict) and 'path' in value,
                    'Actual B2 ' + key + ' declaration is malformed or incomplete')
            seals = [value[name] for name in seal_names if name in value]
            require(seals and all(seal == artifact['sha256'] for seal in seals) and
                    value.get('file_sha256') == info['file_sha256'] and value['path'] == info['path'],
                    'Actual B2 ' + key + ' declaration changed');found = True
        for prefix in (prefix for k in keys for prefix in (k, k + '_receipt')):
            if not any(prefix + suffix in container for suffix in ('_sha256', '_file_sha256', '_path')):continue
            require(all(prefix+suffix in container for suffix in ('_sha256','_file_sha256','_path')) and
                    container[prefix + '_sha256'] == artifact['sha256'] and
                    container.get(prefix + '_file_sha256') == info['file_sha256'] and
                    container.get(prefix + '_path') == info['path'], 'Actual flat B2 ' + key + ' binding changed');found = True
    require(found, 'Final B2 receipt does not bind its retained actual ' + key + ' bytes')


def _checklist(decision, *, conditional=False):
    candidates = [decision[k] for k in ('retained_conditions', 'retained_checklist', 'checklist', 'R_checks') if k in decision]
    require(candidates and all(_strict_equal(v,candidates[0]) for v in candidates), 'Actual row lacks an unambiguous retained-condition checklist')
    checks = candidates[0];require(set(checks) >= {f'R{i}' for i in range(1, 10)}, 'Actual row omitted adopted retained conditions')
    for key in (f'R{i}' for i in range(1, 10)):
        value = checks[key];states = ('pass', 'passed', 'approved') + (('conditional',) if conditional and key == 'R7' else ())
        if isinstance(value, dict):
            statuses = [value[name] for name in ('status', 'result') if name in value]
            notes = [value[name] for name in ('notes', 'note', 'reason') if name in value]
            require(statuses and all(status == statuses[0] for status in statuses) and
                    statuses[0] in states and notes and all(notes), 'Actual adopted rule remains failed or unexplained: ' + key)
        else:
            require(isinstance(value, str) and re.match(r'^(?:' + '|'.join(states) + r')\b', value.casefold()) and
                    (len(value.split()) > 1 or any(decision.get(name) for name in
                        ('notes', 'note', 'evidence_note', 'evidence_notes', 'evidence'))),
                    'Actual adopted rule remains failed or unexplained: ' + key)


def _original_review_value(package, stage, receipt, key, *, required=True):
    """Read declared pointers into untouched original review formats."""
    pointers = package.get('original_review_value_pointers', {})
    allowed = {'packet_sha256', 'protocol_sha256', 'stage', 'acquisition_records', 'author_rationales_seen',
               'model_outcomes_seen', 'evaluator_wording_seen', 'other_reviewer_judgments_seen', 'prior_review_history_seen'}
    require(set(pointers) <= {'first_pass', 'final'} and all(isinstance(value, dict) and set(value) <= allowed
            for value in pointers.values()), 'Review adapter contains invented or undeclared semantic fields')
    values=[]
    if key in pointers.get(stage, {}):
        values.append(json_pointer(receipt,pointers[stage][key]))
    if key == 'packet_sha256':
        values.extend(receipt[name] for name in ('packet_sha256','review_packet_sha256','packet_content_sha256') if name in receipt)
        require(all(isinstance(receipt[k],dict) for k in ('bindings','inputs') if k in receipt),
                'Original packet binding container is malformed')
        containers=[receipt]+[receipt[k] for k in ('bindings','inputs') if isinstance(receipt.get(k),dict)]
        for container in containers:
            for name in ('primary_packet','packet'):
                if name not in container:continue
                binding=container[name]
                require(isinstance(binding,dict), 'Original packet binding is malformed')
                seals=[binding[k] for k in ('sha256','content_sha256','payload_seal','payload_seal_sha256','canonical_sha256','seal_sha256','seal') if k in binding]
                require(seals, 'Original packet binding lacks a declared seal')
                values.extend(seals)
    elif key == 'protocol_sha256':
        values.extend(receipt[k] for k in ('protocol_sha256', 'adopted_protocol_sha256', 'review_protocol_sha256') if k in receipt)
    elif key == 'acquisition_records':
        aliases=('acquisition_records','qa_reviews','acquisition_reviews','acquisition_decisions')
        if receipt.get('schema')=='spacing-source-semantic-review-first-pass-v1':aliases+=('rows',)
        values.extend(receipt[k] for k in aliases if k in receipt)
        require(all(isinstance(v,list) for v in values), 'Original QA decision array is malformed')
    else:
        aliases={'stage':('stage','phase'),
                 'evaluator_wording_seen':('evaluator_wording_seen','evaluation_wording_seen'),
                 'author_rationales_seen':('author_rationales_seen','author_rationales_used'),
                 'model_outcomes_seen':('model_outcomes_seen','model_outcomes_used')}.get(key,(key,))
        values.extend(receipt[k] for k in aliases if k in receipt)
        if key.endswith('_seen') and 'disclosures' in receipt:
            require(isinstance(receipt['disclosures'],dict), 'Original review visibility container is malformed')
            values.extend(receipt['disclosures'][k] for k in aliases if k in receipt['disclosures'])
    require(not values or all(_strict_equal(v,values[0]) for v in values),
            'Original review aliases or pointer contradict an explicit declaration: '+key)
    if key in ('packet_sha256','protocol_sha256'):
        require((values or not required) and all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in values),
                'Original review binding seal is absent or malformed: '+key)
    return values[0] if values else ([] if key=='acquisition_records' else None)


def _strict_equal(value, expected):
    """JSON equality without Python's bool/int coercion, including nested values."""
    return strict_json_equal(value,expected)


def _require_final(receipt, *, stage=None):
    stage = _original_review_value({},'final',receipt,'stage') if stage is None else stage
    require(not _nonfinal_stage(stage) and all(type(receipt[k]) is bool and receipt[k] is True
            for k in ('counts_as_final_approval','first_pass_counts_as_final_approval') if k in receipt),
            'Provisional or explicitly nonfinal review cannot confer final approval')


def _require_nonfinal_declarations(receipt):
    require(all(type(receipt[k]) is bool and receipt[k] is False for k in
                ('counts_as_final_approval','first_pass_counts_as_final_approval') if k in receipt),
            'Original B1 finality declarations must be exact false booleans')


def _first_pass_stage(value):
    return isinstance(value, str) and ('first_pass' in value.casefold() or re.match(r'^b1(?:\b|_)', value.casefold()))


def _nonfinal_stage(value):
    return _first_pass_stage(value) or isinstance(value,str) and bool(re.search(
        r'(?:^|[\s_-])(?:provisional|nonfinal|non_final|draft|preliminary)(?:$|[\s_-])',value.casefold()))


def _source_author_channels(chunks, *, original_trace_input_files_by_proof=None):
    """Collect verified author/revision channels, including retained ancestors."""
    channels=set()
    for chunk in chunks:
        for version in list(chunk.get('history',[]))+[chunk]:
            context=author_context(version,original_trace_input_files_by_proof=original_trace_input_files_by_proof)
            for person in context['people'].values():channels.update(identity_channels(person)-{'/root','root'})
            for row in context['task_provenance']:
                task=row.get('task_record',{})
                channels.update(v for v in (task.get('client_request_id'),task.get('task_id'),task.get('child_thread_id'),
                    task.get('child_run_id'),task.get('canonical_task_name')) if v)
    return channels


def _review_channels(binding):
    return (set(binding['declared_reviewer_identities'])-{'/root','root'}) | {
        binding[k] for k in ('task_id','client_request_id','child_run_id','child_thread_id')}


def _review_content_hash(decision):
    values = [decision[k] for k in ('reviewed_content_sha256', 'content_sha256') if k in decision]
    require(values and all(value == values[0] for value in values), 'Original review content aliases are absent or contradictory')
    return values[0]


def _census_source_unit(row):
    """Projection of original assertions/labels, excluding evaluator IDs and author text."""
    return {'unit_id': row['unit_id'], 'event': row.get('event', row.get('event_id')),
            'source_assertion': row.get('source_assertion', row.get('source_statement')),
            'canonical_answer_labels': row.get('canonical_answer_labels', row.get('canonical_answers')),
            'evidence': row.get('evidence', [])}


def development_census_snapshot(sources):
    """Derive all 157 normal groups and both scale chunks from original local bytes.

    The raw originals are custodian evidence. The census receives their safe
    projections, never their role labels, author rationales or probe inventories.
    """
    require(set(sources) == {'normal_grounded', 'normal_qa_versions', 'scale_packets'},
            'Development census needs complete normal and scale original sources')
    def original(package):
        value = checked(package['artifact']);raw_artifact(value, package['artifact_file']);return value
    normal = original(sources['normal_grounded']);versions = [original(p) for p in sources['normal_qa_versions']]
    require(normal.get('schema') == 'spacing-grounded-declarations-v1' and normal.get('mode') == 'development' and
            len(normal['groups']) == 157 and len(normal['source_units']) == 188 and len(versions) == 3,
            'Normal development census is missing the full original declarations or QA overlay chain')
    qa = {}
    for index, version in enumerate(versions):
        require(version.get('evaluation_question_text_included', False) is False and
                version.get('model_outcomes_used', False) is False,
                'Normal development QA provenance contains evaluation wording/outcomes')
        if index:
            require(version['parent_review_packet_sha256'] == versions[index-1]['sha256'] and
                    {q['id'] for q in version['records']} <= set(qa),
                    'Normal QA overlay does not bind its exact original parent/identities')
        records = version['records'];require(len(records) == len({q['id'] for q in records}), 'Duplicate normal QA overlay identity')
        qa.update((q['id'], source_review_content(q, 'acquisition_records')) for q in records)
    require(len(qa) == 78, 'Normal development census does not contain the reviewed 78 QA')
    normal_groups = [{k:g[k] for k in ('id','event','source_unit_ids','statement')} for g in normal['groups']]
    packets = [('normal', {'groups':normal_groups, 'acquisition_records':list(qa.values()),
        'source_units':[_census_source_unit(u) for u in normal['source_units']], 'factsheets':normal['factsheets']},
        digest([normal['sha256'], *[v['sha256'] for v in versions]]))]
    scale = [original(p) for p in sources['scale_packets']]
    require(len(scale) == 2 and sorted(p['chunk_index'] for p in scale) == [1,2] and
            sum(len(p['groups']) for p in scale) == 257 and sum(len(p['acquisition_records']) for p in scale) == 286,
            'Development census omitted reviewed scale groups/targets or substituted the deployed subset')
    for packet in sorted(scale, key=lambda p:p['chunk_index']):
        require(packet.get('author_rationales_included') is False, 'Scale census input contains author rationales')
        projected = {k:[source_review_content(r,k) for r in packet[k]] for k in ('groups','acquisition_records')}
        require(all(_strict_equal(projected[k],packet[k]) for k in projected), 'Scale census input changed its source-only projection')
        projected.update(source_units=[_census_source_unit(u) for u in packet['source_units']], factsheets=packet['factsheets'])
        packets.append(('scale', projected, packet['sha256']))
    rows = []
    for scope, packet, packet_seal in packets:
        units = {u['unit_id']:u for u in packet['source_units']};groups = {g['id']:g for g in packet['groups']}
        require(len(units) == len(packet['source_units']) and len(groups) == len(packet['groups']), 'Repeated development census source/group')
        for field in ('groups','acquisition_records'):
            for record in packet[field]:
                event = record['event'];owned = record['source_unit_ids'] if field == 'groups' else groups[record['unit_id']]['source_unit_ids']
                require(set(owned) <= set(units), 'Development census target lost its exact original assertions')
                context = {'factsheet':packet['factsheets'][event],
                    'original_source_units':sorted((u for u in units.values() if u['event']==event), key=lambda u:u['unit_id']),
                    'groups':sorted((g for g in groups.values() if g['event']==event),key=lambda g:g['id']),
                    'acquisition_records':sorted((q for q in packet['acquisition_records'] if q['event']==event),key=lambda q:q['id'])}
                rows.append({'domain':'development', 'scope':scope, 'field':field, 'id':record['id'], 'event':event,
                    'content_sha256':digest(record), 'context_sha256':digest(context), 'packet_sha256':packet_seal})
    require(len(rows) == len({(r['scope'],r['field'],r['id']) for r in rows}), 'Duplicate development census population identity')
    return sorted(rows,key=lambda r:(r['scope'],r['field'],r['id']))


def _census_identity(row):
    return tuple(row[k] for k in ('domain','scope','field','id'))


def _utc(value):
    require(isinstance(value,str), 'Census chronology lacks an original UTC timestamp')
    parsed = datetime.datetime.fromisoformat(value.replace('Z','+00:00'))
    require(parsed.utcoffset() == datetime.timedelta(0), 'Census chronology timestamp is not UTC')
    return parsed


def _census_prior_approvals(chunks, sources, *, original_trace_input_files_by_proof=None):
    """Original approved QA references, including changed historical versions."""
    versions = [(v['review_packet'],v.get('reviews',[]),v.get('artifact_files',{}).get('reviews',[]),v.get('original_review_packages',[]),
                 v.get('retained_history_inventory'),
                 'confirmation',str(c['review_packet']['chunk_index']),v is not c)
                for c in chunks for v in list(c.get('history',[]))+[c]]
    for source in sources:
        packet = checked(source['review_packet']);raw_artifact(packet,source['review_packet_file'])
        versions.append((packet,[p['artifact'] for p in source['reviews']],
            [p['evidence']['receipt_file'] for p in source['reviews']],source['reviews'],None,'development',source['scope'],False))
    refs = []
    source_author_channels=_source_author_channels(chunks,original_trace_input_files_by_proof=original_trace_input_files_by_proof)
    for packet, reviews, infos, packages, retained_inventory, domain, scope, historical in versions:
        checked(packet)
        if not reviews and domain=='confirmation' and historical:
            require(not infos and not packages and retained_inventory is not None,
                    'Unreviewed retained history lacks its exact complete inventory and exclusion')
            inventory,_=completed_artifact(retained_inventory)
            require(inventory.get('history_status')=='unreviewed_retained' and inventory.get('packet_sha256')==packet['sha256'] and
                    inventory.get('original_review_receipt_sha256s')==[] and inventory.get('final_approval_qualified') is False and
                    inventory.get('review_decisions_created') is False,
                    'Retained unreviewed history is not explicitly excluded from final approval')
            continue
        require(len(reviews)==len(infos)==len(packages) and len({p['artifact']['sha256'] for p in packages})==len(packages) and
                {p['artifact']['sha256'] for p in packages}=={r['sha256'] for r in reviews},
                'Original census reviews lack complete exact original completion packages')
        completed={};consumed_infos=set()
        for p in packages:
            artifact,binding=completed_artifact(p)
            require(not _review_channels(binding)&source_author_channels,
                    'Original confirmation/development reviewer is a source author identity/task')
            completed[artifact['sha256']]=p
        records={r['id']:r for r in packet.get('acquisition_records',packet.get('records',[]))}
        require(len(records)==len(packet.get('acquisition_records',packet.get('records',[]))),
                'Original census packet repeats QA identities')
        final_decided=set();has_final=False
        for review in reviews:
            require(_original_review_value({},'final',review,'packet_sha256')==packet['sha256'],
                    'Original prior approval binds another source packet')
            checked(review);matches=[i for i in infos if json.loads(i.get('utf8','{}')).get('sha256')==review['sha256']]
            require(len(matches)==1, 'Prior census approval lacks its exact original receipt bytes')
            raw_artifact(review,matches[0])
            index=next(i for i,info in enumerate(infos) if info is matches[0])
            require(index not in consumed_infos,'Original census raw receipt was multiply consumed');consumed_infos.add(index)
            require(_strict_equal(completed[review['sha256']]['evidence']['receipt_file'],matches[0]),
                    'Original census completion retained another receipt file')
            field='acquisition_records' if 'acquisition_records' in packet else 'records'
            decisions=(_original_review_value({},'final',review,'acquisition_records')
                       if field=='acquisition_records' else decisions_for(review,field))
            provisional=(_nonfinal_stage(_original_review_value({},'final',review,'stage')) or
                    any(review.get(k) is False for k in ('counts_as_final_approval','first_pass_counts_as_final_approval')))
            require(len(decisions)==len({d['id'] for d in decisions}) and all(d['id'] in records and
                    (not any(k in d for k in ('reviewed_content_sha256','content_sha256')) or
                     _review_content_hash(d)==digest(records[d['id']])) for d in decisions),
                    'Original decision history is repeated or contradicts its declared exact source content')
            if provisional:
                require(all(review[k] is False for k in ('counts_as_final_approval','first_pass_counts_as_final_approval') if k in review),
                        'Original nonfinal history declares contradictory final approval qualification')
                continue
            has_final=True
            _require_final(review)
            require(len(decisions)==len({d['id'] for d in decisions}) and all(d['id'] in records and
                    d.get('status') in ('approved','approve','approved_development','needs_revision','reject','rejected','failed',
                                      'revise','needs_revision_development') and
                    _review_content_hash(d)==digest(records[d['id']]) for d in decisions),
                    'Original final decision coverage is repeated, invalid, or not bound to exact source content')
            final_decided.update(d['id'] for d in decisions)
            for decision in decisions:
                if decision.get('status') not in ('approved','approve','approved_development'):continue
                require(decision['id'] in records and _review_content_hash(decision)==digest(records[decision['id']]),
                        'Prior approved census row does not bind exact original source content')
                refs.append({'domain':domain,'scope':scope,'field':'acquisition_records','id':decision['id'],
                    'content_sha256':_review_content_hash(decision),'packet_sha256':packet['sha256'],
                    'receipt_sha256':review['sha256']})
        require(consumed_infos==set(range(len(infos))),'Original census raw receipt inventory is not exactly consumed')
        if historical and not has_final and retained_inventory is not None:
            inventory,_=completed_artifact(retained_inventory)
            require(inventory.get('history_status')=='reviewed_nonfinal_retained' and
                    inventory.get('packet_sha256')==packet['sha256'] and
                    sorted(inventory.get('original_review_receipt_sha256s',[]))==sorted(r['sha256'] for r in reviews) and
                    inventory.get('final_approval_qualified') is False and inventory.get('review_decisions_created') is False,
                    'Reviewed nonfinal history was erased or routed as unreviewed/final history')
        # A genuinely completed historical B1-only version retains its reviewed
        # nonfinal state. Current/development and any final-reviewed version keep
        # complete final decision coverage, including negatives.
        require(historical and not has_final or final_decided==set(records),
                'Original final review decisions omit source QA, including negative or unapproved rows')
    unique={digest(r):r for r in refs}
    return sorted(unique.values(),key=lambda r:tuple(str(r[k]) for k in sorted(r)))


def legacy_claim_unverified_envelope(package, *, custodian=None):
    """Separate provenance proposal. Never accepted as a completed review/launch approval."""
    packet=checked(package['review_packet']);receipt=checked(package['original_receipt'])
    raw_artifact(packet,package['review_packet_file']);raw_artifact(receipt,package['original_receipt_file'])
    require(_original_review_value({},'final',receipt,'packet_sha256')==packet['sha256'] and
        package.get('original_task_completion_verified') is False and
        package.get('completion_evidence_unresolved')=='missing_original_task_transport',
        'Legacy provenance proposal must retain exact original bindings and unresolved completion')
    field='acquisition_records' if 'acquisition_records' in packet else 'records'
    records={r['id']:r for r in packet[field]};decisions=(_original_review_value({},'final',receipt,'acquisition_records')
        if field=='acquisition_records' else decisions_for(receipt,field))
    claims=[]
    for row in decisions:
        declared=any(k in row for k in ('reviewed_content_sha256','content_sha256'))
        require(row['id'] in records and (not declared or _review_content_hash(row)==digest(records[row['id']])),
                'Legacy claim does not bind exact original source content')
        claims.append({'id':row['id'],'original_status':row.get('status'),
            'content_sha256':digest(records[row['id']]),
            'content_binding_kind':'original_declared_row_hash' if declared else 'derived_from_original_packet_seal_and_row_id',
            'original_row_hash_declaration_present':declared,'original_receipt_sha256':receipt['sha256'],
            'source_packet_sha256':packet['sha256'],'provenance_kind':'legacy_claim_unverified',
            'approval_kind':'claimed_approval_unverified' if row.get('status') in ('approved','approve','approved_development')
                else 'original_nonapproval_retained', 'qualified_source_or_carryover_approval':False})
    value={'schema':'spacing-sourceqa-legacy-historical-claim-envelope-proposal-v1',
        'original_receipt_binding':{k:package['original_receipt_file'][k] for k in ('path','file_sha256')},
        'original_receipt_sha256':receipt['sha256'],'source_packet_sha256':packet['sha256'],'claims':claims,
        'completion_state':'legacy_claim_unverified','original_task_completion_verified':False,
        'completion_metadata_invented':False,'qualified_source_approval_created':False,
        'accepted_by_protocol_coverage':False,'proposal_only':True}
    value['sha256']=digest(value)
    if custodian is not None:
        artifact,binding=_attested_package(custodian,{'legacy_claim_envelope_sha256':value['sha256'],
            'exact_original_bytes_verified':True,'original_task_completion_verified':False,
            'qualified_source_approval_created':False,'claims_reused_as_launch_approval':False})
        return {'envelope':value,'actual_provenance_custodian_sha256':artifact['sha256'],
                'actual_provenance_custodian_task':binding,'accepted_by_protocol_coverage':False}
    return value


def verify_census_contract(package, census, protocol, confirmation_snapshot, chunks, *, original_trace_input_files_by_proof=None):
    """K6 full property results and K7 genuine freeze/count chronology, tagging only."""
    pointers=package['contract_pointers']
    required={'population_snapshot','property_rows','lane_sets','newly_lane_eligible_sets',
        'prior_approval_results','previously_approved_failing_C6_rows','per_sub_kind_counts',
        'rule_freeze_sha256','prior_approval_inventory_sha256','retained_prior_census_versions',
        'retained_prior_rule_versions','chronology'}
    require(set(pointers)==required,'Census lacks complete property/freeze/result adapters')
    values={k:json_pointer(census,p) for k,p in pointers.items()}
    development=development_census_snapshot(package['development_population_sources'])
    confirmation=[{'domain':'confirmation','scope':str(r['chunk_index']),**{k:v for k,v in r.items() if k!='chunk_index'}}
                  for r in confirmation_snapshot]
    population=sorted(confirmation+development,key=_census_identity)
    require(_strict_equal(values['population_snapshot'],population),'K6 census omitted groups/targets, changed source contexts or used a development subset')
    freeze,freeze_task=completed_artifact(package['rule_freeze'])
    require(freeze.get('schema')=='spacing-sourceqa-census-rule-freeze-v1' and freeze['protocol_sha256']==protocol['sha256'] and
        values['rule_freeze_sha256']==freeze['sha256'] and _strict_equal(freeze['rule_text'],protocol['exception_rules']) and
        _strict_equal(freeze['census_constants'],protocol['census_constants']) and
        _strict_equal(freeze['normalization_reference'],{'path':'experiments/spacing_rerun/spacing_rerun/data.py',
            'file_sha256':protocol['frozen_mechanical_files']['experiments/spacing_rerun/spacing_rerun/data.py'],
            'function_text':'return " ".join(str(text).strip().casefold().split()).rstrip(".!?,;:")'}),
        'K7 freeze lacks exact adopted rules, definitions, categories or normalization reference')
    disclosure=freeze['prior_count_view_disclosure']
    require(type(disclosure.get('prior_counts_seen')) is bool and isinstance(disclosure.get('report_sha256s'),list) and
        isinstance(disclosure.get('known_row_motivation'),str) and disclosure['known_row_motivation'].strip(),
        'K7 freeze omits previous count views or known-row motivation')
    properties=freeze['property_definitions'];subkinds=freeze['eligibility_sub_kinds']
    require(isinstance(properties,dict) and {'LE1a','LE1b','LE1c','C6_masked','ordinary_retained_defect'}<=set(properties) and
        all(isinstance(v,dict) and isinstance(v.get('operational_definition'),str) and v['operational_definition'].strip()
            for v in properties.values()) and subkinds=={'LE1a':'conflict','LE1b':'alias','LE1c':'original_label_nonuniqueness'},
        'K6 property categories are not the frozen eligibility and retained-condition tests')
    results=values['property_rows'];require(len(results)==len(population) and
        [_census_identity(r) for r in results]==[_census_identity(r) for r in population], 'K6 per-property row coverage is incomplete/repeated')
    lookup={_census_identity(r):r for r in population}
    for row in results:
        require(row['content_sha256']==lookup[_census_identity(row)]['content_sha256'] and
            row['context_sha256']==lookup[_census_identity(row)]['context_sha256'] and
            set(row['properties'])==set(properties) and all(type(v) is bool for v in row['properties'].values()) and
            isinstance(row.get('source_only_evidence'),str) and row['source_only_evidence'].strip(),
            'K6 result lacks exact source bindings, every frozen property or substantive source evidence')
    result_lookup={_census_identity(r):r for r in results};sets=values['lane_sets'];seen=set();counts={kind:0 for kind in subkinds.values()}
    newly=[]
    previous_rule_seals={v['sha256'] for v in values['retained_prior_rule_versions']}
    for item in sets:
        require(item['set_id'] not in seen and item['property_key'] in subkinds and
            item['sub_kind']==subkinds[item['property_key']] and type(item['previously_lane_eligible']) is bool,
            'K6 set repeats an identity or changes a frozen sub-kind')
        seen.add(item['set_id']);members=[tuple(v[k] for k in ('domain','scope','field','id')) for v in item['members']]
        require(len(members)>=2 and len(members)==len(set(members)) and set(members)<=set(lookup) and
            len({(lookup[k]['domain'],lookup[k]['scope'],lookup[k]['event']) for k in members})==1 and
            all(result_lookup[k]['properties'][item['property_key']] for k in members),
            'K6 set is outside exact co-event population or contradicts property tags')
        require(isinstance(item.get('source_only_evidence'),str) and item['source_only_evidence'].strip(), 'K6 set lacks property evidence')
        previous=item['prior_eligibility_assessment']
        require(previous['rule_version_sha256'] in previous_rule_seals and
            previous['eligible']==item['previously_lane_eligible'] and type(previous['eligible']) is bool and
            isinstance(previous.get('source_only_evidence'),str) and previous['source_only_evidence'].strip(),
            'K6 newly eligible set lacks its source-only recheck under an exact retained prior rule')
        counts[item['sub_kind']]+=1
        if not item['previously_lane_eligible']:newly.append(item['set_id'])
    for property_key in subkinds:
        tagged={k for k,r in result_lookup.items() if r['properties'][property_key]}
        members={_census_identity(r) for item in sets if item['property_key']==property_key for r in item['members']}
        require(tagged==members, 'K6 '+property_key+' property tags disagree with matching lane-set membership')
    require(values['newly_lane_eligible_sets']==sorted(newly) and _strict_equal(values['per_sub_kind_counts'],counts),
            'K6 lane sets/newly eligible list/sub-kind counts disagree with full property tags')
    prior_sources=package['development_prior_approval_sources']
    prior_ids={(source['scope'],q['id']) for source in prior_sources
        for q in source['review_packet'].get('acquisition_records',source['review_packet'].get('records',[]))}
    require(prior_ids=={(r['scope'],r['id']) for r in development if r['field']=='acquisition_records'} and
        all(source['reviews'] for source in prior_sources), 'K6 original development review history omitted normal/scale targets')
    prior=_census_prior_approvals(chunks,prior_sources,original_trace_input_files_by_proof=original_trace_input_files_by_proof)
    require(values['prior_approval_inventory_sha256']==digest(prior),'K6 prior-approval inventory is incomplete or stale')
    prior_results=values['prior_approval_results']
    require(_strict_equal([r['original_approval_reference'] for r in prior_results],prior) and all(
        type(r.get('masked_C6_passed')) is bool and isinstance(r.get('source_only_evidence'),str) and r['source_only_evidence'].strip()
        for r in prior_results),'K6 lacks the masked C6 recheck for every actual original approval')
    failing=[r['original_approval_reference'] for r in prior_results if not r['masked_C6_passed']]
    require(_strict_equal(values['previously_approved_failing_C6_rows'],failing),'K6 failing prior approvals were dropped or invented')
    chronology,task=_attested_package(package['freeze_order_attestation'],{
        'rule_freeze_sha256':freeze['sha256'],'census_artifact_sha256':census['sha256'],
        'all_versions_retained':True,'chronology_verified_from_original_traces':True})
    timeline=json_pointer(chronology,package['freeze_order_attestation']['chronology_pointer'])
    require(_strict_equal(timeline,values['chronology']) and timeline['rule_freeze_sha256']==freeze['sha256'],
            'K7 original chronology binds another rule or count-view record')
    trace_hashes=[]
    for info in package['freeze_order_attestation']['chronology_trace_files']:
        _original_json_file(info);trace_hashes.append(info['file_sha256'])
    require(trace_hashes and timeline['original_trace_file_sha256s']==trace_hashes,'K7 count/freeze chronology lacks exact original traces')
    frozen=_utc(timeline['frozen_utc']);started=_utc(timeline['census_started_utc']);viewed=_utc(timeline['first_count_view_utc'])
    require(frozen<=started<=viewed,'K7 current rule/categories were frozen after census counts were viewed')
    ancestors=values['retained_prior_census_versions'];require(_strict_equal(ancestors,timeline['retained_prior_census_versions']),
        'K7 census rerun ancestry differs from actual chronology')
    require(disclosure['prior_counts_seen']==timeline['prior_counts_viewed_before_current_freeze'] and
        len(disclosure['report_sha256s'])==len(set(disclosure['report_sha256s'])) and
        set(disclosure['report_sha256s'])<={a['sha256'] for a in ancestors} and
        (not disclosure['prior_counts_seen'] or disclosure['report_sha256s']),
        'K7 previous count views were omitted from the retained rerun history')
    originals=package['prior_census_versions'];require(len(originals)==len(ancestors),'K7 lost a prior census version')
    for ancestor,original in zip(ancestors,originals):
        value=checked(original['artifact']);raw_artifact(value,original['artifact_file'])
        require(ancestor['sha256']==value['sha256'] and ancestor['file_sha256']==original['artifact_file']['file_sha256'] and
            isinstance(ancestor.get('superseded_reason'),str) and ancestor['superseded_reason'].strip(), 'K7 rerun ancestor lacks exact retained original bytes/reason')
    old_rules=values['retained_prior_rule_versions']
    require(_strict_equal(old_rules,timeline['retained_prior_rule_versions']) and len(old_rules)==len(package['prior_rule_versions']),
            'K7 earlier rule/definition/category versions were not retained')
    for reference,original in zip(old_rules,package['prior_rule_versions']):
        value=checked(original['artifact']);raw_artifact(value,original['artifact_file'])
        require(reference['sha256']==value['sha256'] and reference['file_sha256']==original['artifact_file']['file_sha256'],
                'K7 earlier rule version differs from the exact retained original bytes')
    require(timeline.get('prior_counts_viewed_before_current_freeze') is False or
        timeline.get('prior_counts_viewed_before_current_freeze') is True and ancestors and
        timeline.get('R2_rerun_completed') is True,'K7 earlier counts require retained versions and actual R2 rerun')
    require(task['task_id'] not in {freeze_task['task_id'],package['evidence']['task_status']['taskId']},
            'K7 independent chronology custodian shares freeze/census task')
    return {'full_population_sha256':digest(population),'full_population_count':len(population),
        'development_population_sha256':digest(development),'rule_freeze_sha256':freeze['sha256'],
        'property_result_sha256':digest(results),'per_sub_kind_counts':counts,
        'prior_approval_inventory_sha256':digest(prior),'previously_approved_failing_C6_count':len(failing),
        'freeze_order_attestation_sha256':chronology['sha256'],'semantic_decisions_created':False}


def _fresh_qa(package, packet, protocol, conditional_ids):
    receipt, binding = completed_artifact(package)
    require(_original_review_value(package, 'final', receipt, 'packet_sha256') == packet['sha256'] and
            _original_review_value(package, 'final', receipt, 'protocol_sha256') == protocol['sha256'],
            'Fresh QA review does not bind exact current packet and adopted protocol')
    first = checked(package['first_pass_receipt']);first_info = package['first_pass_file'];raw_artifact(first, first_info)
    require('first_pass_evidence' in package, 'B1 lacks genuine original completed task evidence')
    first_actual,first_binding=completed_artifact({'artifact':first,'evidence':package['first_pass_evidence']})
    require(_strict_equal(first_actual,first) and _strict_equal(package['first_pass_evidence']['receipt_file'],first_info),
            'B1 completion binds different original receipt bytes')
    first_protocol=_original_review_value(package,'first_pass',first,'protocol_sha256',required=False)
    require(first_protocol is None or first_protocol==protocol['sha256'],
            'Original B1 protocol declaration differs from adopted review protocol')
    _require_nonfinal_declarations(first)
    def actors(r):
        result={r['reviewer_actor_identity']} if 'reviewer_actor_identity' in r else set()
        if result:return result
        for k in ('reviewer','reviewer_identity'):
            if k not in r:continue
            person=r[k]
            if not any(name in person for name in ('agent_id','identity','review_task_id','review_name','name','actual_identity','client_label')):continue
            result.update(identity_names(person)-{person[n] for n in ('review_task_id','client_label') if n in person})
        return result-{'/root','root'}
    first_actors=actors(first);final_actors=actors(receipt)
    same_original_task=(first_binding['task_id']==binding['task_id'] and first_binding['child_thread_id']==binding['child_thread_id'])
    require(first_binding['model']==binding['model'] and
            first_binding['provider_instance_id']==binding['provider_instance_id'] and
            first_binding['declared_reviewer_roles']==binding['declared_reviewer_roles'] and
            (first_actors and final_actors and first_actors==final_actors or not first_actors and not final_actors and same_original_task),
            'B1 and B2 do not bind the same declared independent reviewer actor/model')
    _same_binding(receipt, 'first_pass', first, first_info)
    history = checked(package['prior_review_supplement']);history_info = package['prior_review_supplement_file']
    raw_artifact(history, history_info)
    _same_binding(receipt, 'prior_review_supplement', history, history_info, aliases=('prior_review_evidence',))
    require(history.get('current_review_packet_sha256') == packet['sha256'] and
            history.get('review_decisions_created') is False and history.get('source_only') is True and
            history.get('author_rationales_included') is False and history.get('model_outcomes_included') is False,
            'B2 history bytes do not bind exact source-only current review context')
    require(_original_review_value(package, 'first_pass', first, 'packet_sha256') == packet['sha256'] and
            _first_pass_stage(_original_review_value(package, 'first_pass', first, 'stage')) and
            not _nonfinal_stage(_original_review_value(package, 'final', receipt, 'stage')) and
            _original_review_value(package, 'final', receipt, 'stage'), 'Fresh review lacks genuine separate B1/B2 stages')
    _require_final(receipt,stage=_original_review_value(package,'final',receipt,'stage'))
    for stage, value in (('first_pass', first), ('final', receipt)):
        require(all(_original_review_value(package, stage, value, key) is False for key in
                ('author_rationales_seen', 'model_outcomes_seen', 'evaluator_wording_seen')),
                'Fresh B1/B2 review lost source/evaluation/author custody')
    require(_original_review_value(package, 'first_pass', first, 'other_reviewer_judgments_seen') is False and
            _original_review_value(package, 'first_pass', first, 'prior_review_history_seen') is False,
            'Blind B1 reviewer saw historical judgments')
    require(_original_review_value(package, 'final', receipt, 'prior_review_history_seen') is True, 'B2 lacks actual history reconciliation')
    first_rows = _original_review_value(package, 'first_pass', first, 'acquisition_records')
    before = {d['id']: d for d in first_rows};final = _original_review_value(package, 'final', receipt, 'acquisition_records')
    current = {r['id']: digest(r) for r in packet['acquisition_records']}
    require(len(first_rows) == len(before) and len(final) == len({d['id'] for d in final}) and set(before) == {d['id'] for d in final} <= set(current),
            'B1/B2 QA identities disappeared or repeated')
    for decision in final:
        rid = decision['id'];require(decision.get('status') in ('approved', 'approve') and _review_content_hash(decision) == current[rid] and
                                    _review_content_hash(before[rid]) == current[rid], 'Fresh review is negative, stale or partial on a required row')
        _checklist(decision, conditional=rid in conditional_ids)
        if before[rid].get('status') in ('approved', 'approve'):
            _checklist(before[rid], conditional=rid in conditional_ids)
        if before[rid].get('status') not in ('approved', 'approve'):
            require(decision.get('b1_error_finding') or decision.get('first_pass_error_finding'),
                    'B1 fail changed to B2 approval without an actual written error finding')
    binding['first_pass_task']=first_binding
    return {d['id'] for d in final}, binding


def _attested_package(package, required):
    artifact, binding = completed_artifact(package)
    pointers = package['attestation_pointers']
    for key, expected in required.items():
        require(key in pointers and _strict_equal(json_pointer(artifact, pointers[key]),expected),
                'Actual source protocol obligation absent or changed: ' + key)
    return artifact, binding


def development_source_runtime_snapshot(package):
    """Bind the isolated development review's own inputs, separately from confirmation."""
    artifact, _ = completed_artifact(package)
    if 'runtime_observation_interface' in package:
        return _separate_development_source_runtime_snapshot(package, artifact)
    pointers = package['development_source_runtime_snapshot_pointers']
    require(set(pointers) == {'normal', 'scale'} and all(set(v) == {'source', 'runtime'} for v in pointers.values()),
            'Development reclassification lacks both own source/runtime snapshot domains')
    snapshot = {'domain': 'development_source_runtime_v1', 'actual_review_artifact_sha256': artifact['sha256']}
    for scope in ('normal', 'scale'):
        snapshot[scope] = {kind: json_pointer(artifact, pointer) for kind, pointer in pointers[scope].items()}
        require(all(isinstance(v, (dict, list)) and v for v in snapshot[scope].values()),
                'Development source/runtime snapshot must bind substantive actual input sections')
    return snapshot


def _original_json_file(info):
    text = info.get('utf8')
    require(isinstance(text, str) and hashlib.sha256(text.encode()).hexdigest() == info.get('file_sha256') and
            isinstance(info.get('path'), str) and info['path'], 'Original metadata/tool bytes are missing or changed')
    return json.loads(text)


def _separate_development_source_runtime_snapshot(package, artifact):
    """Recompute reviewed membership from exact source rows and genuine remote metadata.

    The remote custodian keeps manifests and its full observation remote. Its
    original successful execution stdout contains only identities and hashes.
    No runtime statements are inserted into the independent source receipt.
    """
    interface = package['runtime_observation_interface']
    metadata = _original_json_file(interface['metadata_stdout_file'])
    execution = _original_json_file(interface['actual_execution_file'])
    require(type(execution.get('exit_code')) is int and execution['exit_code'] == 0 and
            execution.get('output') == interface['metadata_stdout_file']['utf8'] and
            execution.get('chunk_id') and not execution.get('session_id'),
            'Development runtime metadata is not bound to its original completed successful execution')
    require(interface['validator_script_file_sha256'] ==
            'cf81ef91d8599e19e5f66f916cbff5523f0bfea433f85b0ef68eb12a8d610374',
            'Development metadata used another runtime/source validator')
    info = package['evidence']['receipt_file'];review_binding = metadata['actual_source_review_binding']
    require(review_binding['sha256'] == artifact['sha256'] and review_binding['file_sha256'] == info['file_sha256'] and
            type(review_binding['bytes']) is int and review_binding['bytes'] == len(info['utf8'].encode()),
            'Runtime validator read a different source-review artifact')
    require(metadata['domain'] == 'development_source_runtime_v1' and metadata['semantic_decisions_created'] is False and
            metadata['runtime_or_source_edited'] is False and metadata['models_run'] is False and
            metadata['evaluation_wording_read'] is False,
            'Development runtime observation crossed source/evaluation or execution custody')
    require(set(metadata['attestations_not_established']) ==
            {'development_retrained', 'E_reselected', 'gate_changed', 'prospective_snapshot_unchanged'},
            'Original narrow runtime observation limitations were dropped')
    rows = json_pointer(artifact, package['development_source_record_rows_pointer'])
    require(len(rows) == 364 and len({(r['set'], r['id']) for r in rows}) == 364 and
            {r['set'] for r in rows} == {'normal78', 'scale286'}, 'Development source reclassification is asymmetric or partial')
    scopes = metadata['scopes'];require(set(scopes) == {'normal78', 'scale286'}, 'Runtime observation omitted a development scope')
    result = {'domain': 'development_source_runtime_v1', 'actual_review_artifact_sha256': artifact['sha256'],
              'actual_runtime_custody_binding': copy.deepcopy(metadata['actual_runtime_custody_binding']),
              'actual_remote_observation_binding': copy.deepcopy(metadata['actual_observation_binding']),
              'actual_execution_file_sha256': interface['actual_execution_file']['file_sha256'],
              'metadata_stdout_file_sha256': interface['metadata_stdout_file']['file_sha256'],
              'validator_script_file_sha256': interface['validator_script_file_sha256']}
    for name, source_scope, source_count, runtime_count in (('normal', 'normal78', 78, 78), ('scale', 'scale286', 286, 218)):
        source_rows = sorted(({k: r[k] for k in ('id', 'event', 'content_sha256')} for r in rows if r['set'] == source_scope),
                             key=lambda r: r['id'])
        observed = scopes[source_scope];excluded = observed['excluded_reviewed_record_ids']
        require(len(source_rows) == source_count and all(re.fullmatch('[0-9a-f]{64}', r['content_sha256']) for r in source_rows) and
                excluded == sorted(set(excluded)) and set(excluded) <= {r['id'] for r in source_rows} and
                len(excluded) == source_count - runtime_count, 'Runtime subset membership is incomplete or changed')
        runtime_rows = [r for r in source_rows if r['id'] not in set(excluded)]
        require(_strict_equal(observed['reviewed_record_count'],source_count) and
                _strict_equal(observed['runtime_record_count'],runtime_count) and
                observed['reviewed_source_projection_sha256'] == digest(source_rows) and
                observed['runtime_source_projection_sha256'] == digest(runtime_rows) and
                observed['all_runtime_records_exact_reviewed_source_content'] is True and
                observed['membership_kind'] == ('exact_equality' if name == 'normal' else 'strict_subset'),
                'Actual deployed source pool differs from the independently reviewed content')
        result[name] = {'source': source_rows, 'runtime': runtime_rows, 'original_observed_scope': copy.deepcopy(observed)}
    trials = metadata['trials'];expected_trials = {f'normal-{i:02}': 'normal' for i in range(1, 7)}
    expected_trials.update({f'scaled-{i:02}': 'scale' for i in range(1, 4)})
    require(len(trials) == len(expected_trials) and {r['trial_id'] for r in trials} == set(expected_trials),
            'Development runtime grid observation is partial or duplicated')
    for trial in trials:
        runtime = result[expected_trials[trial['trial_id']]]['runtime']
        require(_strict_equal(trial['source_qa_record_count'],len(runtime)) and
                _strict_equal(trial['source_event_count'],len({r['event'] for r in runtime})) and
                trial['source_review_content_projection_sha256'] == digest(runtime) and
                trial['all_runtime_records_exact_reviewed_source_content'] is True,
                'A deployed development trial uses a different or unreviewed source pool')
    result['actual_trial_bindings'] = copy.deepcopy(trials)
    return result


def _verify_development_obligations(package, protocol, snapshot):
    if 'runtime_observation_interface' not in package:
        return _attested_package(package, {'protocol_sha256': protocol['sha256'], 'development_content_changed': False,
            'development_retrained': False, 'E_reselected': False, 'gate_changed': False, 'symmetry_complete': True})
    artifact, binding = _attested_package(package, {'protocol_sha256': protocol['sha256'], 'development_content_changed': False})
    preservation = package['runtime_preservation_attestation']
    _, runtime_binding = _attested_package(preservation, {'source_review_sha256': artifact['sha256'],
        'runtime_custody_sha256': snapshot['actual_runtime_custody_binding']['sha256'],
        'runtime_source_observation_sha256': snapshot['actual_remote_observation_binding']['sha256'],
        'development_retrained': False, 'E_reselected': False, 'gate_changed': False})
    require(not {runtime_binding['client_request_id'], runtime_binding['task_id']} &
            {binding['client_request_id'], binding['task_id']},
            'Development runtime custodian shares the isolated scientific source-review task')
    return artifact, binding


def _trusted_evaluator_metadata(expected, catalog, snapshot):
    """Caller-provided current custodian metadata. No evaluator wording is accepted."""
    require(expected is not None, 'Trusted external current evaluator input metadata is required')
    checked(expected,'spacing-sourceqa-current-evaluator-input-metadata-v1')
    require(set(expected)=={'schema','sha256','scope','configuration_sha256','catalog_binding','snapshot_binding',
                           'provenance_binding','required_surface_correlate_ids_by_record'} and
            expected['scope']=={'mode':'confirmation','source_catalog_sha256':catalog['sha256'],
                               'source_snapshot_sha256':digest(snapshot)},
            'Trusted evaluator metadata does not bind exact current scope/configuration')
    require(isinstance(expected['configuration_sha256'],str) and
            re.fullmatch('[0-9a-f]{64}',expected['configuration_sha256']), 'Trusted evaluator configuration seal is malformed')
    for key in ('catalog_binding','snapshot_binding','provenance_binding'):
        reference=expected[key]
        require(isinstance(reference,dict) and set(reference)=={'path','file_sha256','content_sha256'} and
                isinstance(reference['path'],str) and reference['path'].startswith('/') and
                all(isinstance(reference[k],str) and re.fullmatch('[0-9a-f]{64}',reference[k])
                    for k in ('file_sha256','content_sha256')), 'Trusted current evaluator reference is malformed: '+key)
    inventory=expected['required_surface_correlate_ids_by_record']
    qa={r['id'] for r in snapshot if r['field']=='acquisition_records'}
    require(isinstance(inventory,dict) and set(inventory)==qa and all(isinstance(ids,list) and
            len(ids)==len(set(ids)) and all(isinstance(v,str) and v and v==v.strip() for v in ids)
            for ids in inventory.values()), 'Trusted original custodian correlate inventory is incomplete or malformed')
    return expected


def verify_protocol_coverage(proof, protocol, chunks, catalog, *, dependency_context=None, expected_evaluator_metadata=None,
                             original_trace_input_files_by_proof=None):
    """Fail closed on coverage/identity/CF/custody; semantic decisions remain original receipts."""
    _protocol(protocol);checked(proof, SCHEMA);snapshot = protocol_source_snapshot(chunks, catalog)
    expected_evaluator_metadata=_trusted_evaluator_metadata(expected_evaluator_metadata,catalog,snapshot)
    require(proof.get('protocol_sha256') == protocol['sha256'] and proof.get('source_catalog_sha256') == catalog['sha256'] and
            _strict_equal(proof.get('source_snapshot'),snapshot) and proof.get('source_snapshot_sha256') == digest(snapshot) and
            proof.get('semantic_approvals_created') is False, 'Protocol coverage no longer binds exact current source contexts')
    require(set(proof['dependency_context']) == DEPENDENCY_KEYS and all(re.fullmatch('[0-9a-f]{64}', value)
            for value in proof['dependency_context'].values()), 'Coverage dependency/lane context is incomplete')
    if dependency_context is not None:require(_strict_equal(proof['dependency_context'],dependency_context), 'Coverage dependency/lane proof changed')
    packets = {c['review_packet']['chunk_index']: c['review_packet'] for c in chunks}
    conditional = set(proof['conditional_acquisition_record_ids']);qa = {r['id']: r for r in snapshot if r['field'] == 'acquisition_records'}
    require(conditional <= set(qa), 'Conditional routing references absent QA')
    covered, reviewers = set(), set()
    source_author_channels = _source_author_channels(chunks,original_trace_input_files_by_proof=original_trace_input_files_by_proof)
    actual_conditional = set()
    for package in proof['fresh_qa_reviews']:
        receipt = package.get('artifact', package.get('receipt'))
        for decision in _original_review_value(package, 'final', receipt, 'acquisition_records'):
            if any(decision.get(k) for k in ('conditions', 'approval_conditions', 'source_conditions', 'exception_lane')):
                actual_conditional.add(decision['id'])
    require(actual_conditional <= conditional, 'Actual conditional review was routed as ordinary fresh QA')
    for package in proof['fresh_qa_reviews']:
        index = package['chunk_index'];ids, task = _fresh_qa(package, packets[index], protocol, conditional)
        stage_channels=set()
        for stage_task in (task,task['first_pass_task']):
            stage_channels.update(stage_task[k] for k in ('client_request_id','task_id','child_run_id','child_thread_id'))
            stage_channels.update(set(stage_task['declared_reviewer_identities'])-{'/root','root'})
        require(not stage_channels & source_author_channels, 'Fresh protocol reviewer is a source author task or declared author identity')
        require(not covered & ids, 'Current QA has duplicated fresh review coverage')
        covered.update(ids);reviewers.update(stage_channels)
    require(covered == set(qa), 'Full current fresh QA B1/B2 coverage remains incomplete')
    expected_routes = [{'id': rid, 'route': 'fresh_conditional' if rid in conditional else 'fresh',
                        'content_sha256': row['content_sha256'], 'context_sha256': row['context_sha256']}
                       for rid, row in sorted(qa.items())]
    require(_strict_equal(proof['qa_routing'],expected_routes), 'Changed/conditional/carryover routing differs from actual full-fresh coverage')
    expected_audit = seeded_audit_plan({i: [] for i in range(1, 9)})
    require(_strict_equal(proof['carryover_audit'],expected_audit), 'CF5 audit may be empty only with proven all-current fresh QA coverage')
    # CF1/CF2 for declarations: exact original approval and its own source context,
    # never QA paraphrase changes being treated as a changed declaration claim.
    group_lookup = {r['id']: r for r in snapshot if r['field'] == 'groups'};group_covered = set()
    authored_versions = {v['authored']['sha256']: v['authored'] for c in chunks for v in list(c.get('history', [])) + [c]}
    for package in proof['group_reviews']:
        receipt, group_task = completed_artifact(package);old = checked(package['review_packet']);raw_artifact(old, package['review_packet_file'])
        _require_final(receipt)
        require(not ((set(group_task['declared_reviewer_identities'])-{'/root','root'}) |
                {group_task[k] for k in ('task_id','client_request_id','child_run_id','child_thread_id')}) & source_author_channels,
                'Group carryover reviewer is a source author identity/task')
        require(_original_review_value(package,'final',receipt,'packet_sha256') == old['sha256'], 'Group carryover receipt binds another packet')
        rows = {r['id']: r for r in old['groups']}
        for decision in decisions_for(receipt, 'groups'):
            rid = decision['id']
            if rid not in group_lookup or decision.get('status') != 'approved':continue
            old_context = _event_context(old, rows[rid]['event'], kind='groups', record=rows[rid],
                                         authored=authored_versions.get(old['authored_chunk_sha256']))
            if decision['reviewed_content_sha256'] == group_lookup[rid]['content_sha256'] == digest(rows[rid]) and digest(old_context) == group_lookup[rid]['context_sha256']:
                group_covered.add(rid)
    require(group_covered == set(group_lookup), 'Genuine exact unchanged group approval/context coverage remains incomplete')
    common = {'protocol_sha256': protocol['sha256'], 'source_snapshot_sha256': digest(snapshot)}
    census, task = _attested_package(proof['census'], {**common, 'model_outcomes_seen': False,
        'evaluation_wording_seen': False, 'author_rationales_seen': False, 'no_semantic_decisions_created': True})
    require(not {task['client_request_id'], task['task_id']} & (reviewers | source_author_channels),
            'Census agent is a source author or same-row B1/B2 reviewer')
    census_rows = []
    for projection in proof['census']['record_binding_pointers']:
        if isinstance(projection, str):census_rows.append(json_pointer(census, projection))
        else:
            require(set(projection) == {'field', 'pointers'} and set(projection['pointers']) ==
                    {'chunk_index', 'id', 'event', 'content_sha256', 'context_sha256', 'packet_sha256'},
                    'Census adapter cannot invent current row/content/context fields')
            census_rows.append({'field': projection['field'], **{k: json_pointer(census, p)
                               for k, p in projection['pointers'].items()}})
    require(_strict_equal(census_rows,snapshot), 'Census omitted or changed exact current records/contexts')
    census_contract = verify_census_contract(proof['census'], census, protocol, snapshot, chunks,
        original_trace_input_files_by_proof=original_trace_input_files_by_proof)
    development_snapshot = development_source_runtime_snapshot(proof['development_reclassification'])
    _verify_development_obligations(proof['development_reclassification'], protocol, development_snapshot)
    require(_strict_equal(proof.get('development_source_runtime_snapshot'),development_snapshot) and
            proof.get('development_source_runtime_snapshot_sha256') == digest(development_snapshot),
            'Development reclassification own source/runtime snapshot changed or crossed confirmation domains')
    custody, custodian = _attested_package(proof['evaluation_custody'], {**common, 'boolean_only_output': True,
        'evaluation_text_exposed': False, 'filtering_permitted': False, 'missing_flags_imputed_false': False})
    custody_package=proof['evaluation_custody']
    require(_strict_equal(json_pointer(custody,custody_package['evaluator_input_metadata_pointer']),expected_evaluator_metadata),
            'Current evaluator custody inputs differ from trusted external raw/content/provenance pins')
    for prefix,key in (('evaluation_catalog','catalog_binding'),('evaluation_snapshot','snapshot_binding')):
        for suffix,field in (('_sha256','content_sha256'),('_file_sha256','file_sha256'),('_path','path')):
            if prefix+suffix in custody:
                require(custody[prefix+suffix]==expected_evaluator_metadata[key][field],
                        'Original evaluator input alias differs from trusted current input: '+prefix+suffix)
    require(custodian['client_request_id'] not in reviewers and custodian['client_request_id'] != task['client_request_id'],
            'Evaluation custody shares the source-review/census task channel')
    require(not {custodian['client_request_id'], custodian['task_id']} & (source_author_channels | reviewers),
            'Evaluation custodian shares a source author/reviewer task channel')
    flags = json_pointer(custody, proof['evaluation_custody']['flag_rows_pointer'])
    require(len(flags) == len({r['id'] for r in flags}) and {r['id'] for r in flags} == set(qa) and all(
        type(r.get('canonical_or_paraphrase_overlap_flag')) is bool and r.get('content_sha256') == qa[r['id']]['content_sha256']
        for r in flags), 'Evaluation custody flag coverage is incomplete, stale or imputed')
    for row in flags:
        require(isinstance(row.get('surface_correlate_flags'), dict) and all(type(v) is bool for v in row['surface_correlate_flags'].values()),
                'L5 custody flags must remain explicit booleans, never exposed wording')
    counts = json_pointer(custody, proof['evaluation_custody']['canonical_root_counts_pointer'])
    expected_counts = {u['unit_id']: len({p['opaque_probe_id'] for p in u['canonical_probes']}) for u in catalog['units']}
    require(_strict_equal(counts,expected_counts), 'Evaluation-custody K1 original canonical-root counts changed or are incomplete')
    correlate_ids = proof['evaluation_custody']['required_surface_correlate_ids_by_record']
    original_correlates=json_pointer(custody,custody_package['required_surface_correlate_ids_pointer'])
    require(_strict_equal(correlate_ids,original_correlates) and
            _strict_equal(original_correlates,expected_evaluator_metadata['required_surface_correlate_ids_by_record']),
            'L5 correlate inventory is not bound to original custodian output and trusted current evaluator inputs')
    require(set(correlate_ids) == set(qa) and all(isinstance(ids, list) and len(ids) == len(set(ids))
            for ids in correlate_ids.values()) and all(set(row['surface_correlate_flags']) == set(correlate_ids[row['id']])
            for row in flags), 'Evaluation custody omitted a required L5 correlate flag')
    require(_strict_equal(proof['evaluation_custody']['dependency_context'],proof['dependency_context']), 'Custody/L5 flags bind another lane policy')
    require(_strict_equal(proof.get('claims_forfeited'),protocol['claims_forfeited']) and _strict_equal(proof.get('timing_disclosure'),protocol['review_rule_change_disclosure']),
            'Protocol disclosure or forfeited claims were dropped')
    return {'protocol_sha256': protocol['sha256'], 'coverage_sha256': proof['sha256'],
            'source_snapshot_sha256': digest(snapshot), 'fresh_qa_count': len(qa), 'group_count': len(group_lookup),
            'carryover_qa_count': 0, 'census_contract': census_contract, 'semantic_approvals_created': False}


def build_protocol_coverage(protocol, chunks, catalog, *, fresh_qa_reviews, group_reviews, census,
                            development_reclassification, evaluation_custody, dependency_context,
                            conditional_acquisition_record_ids=(), expected_evaluator_metadata=None,
                            original_trace_input_files_by_proof=None):
    """Mechanically assemble only actual completed evidence; never invent a receipt."""
    snapshot = protocol_source_snapshot(chunks, catalog);conditional = sorted(conditional_acquisition_record_ids)
    development_snapshot = development_source_runtime_snapshot(development_reclassification)
    proof = {'schema': SCHEMA, 'protocol_sha256': protocol['sha256'], 'source_catalog_sha256': catalog['sha256'],
             'source_snapshot': snapshot, 'source_snapshot_sha256': digest(snapshot),
             'fresh_qa_reviews': copy.deepcopy(fresh_qa_reviews), 'group_reviews': copy.deepcopy(group_reviews),
             'census': copy.deepcopy(census), 'development_reclassification': copy.deepcopy(development_reclassification),
             'development_source_runtime_snapshot': development_snapshot,
             'development_source_runtime_snapshot_sha256': digest(development_snapshot),
             'evaluation_custody': copy.deepcopy(evaluation_custody), 'dependency_context': copy.deepcopy(dependency_context),
             'conditional_acquisition_record_ids': conditional,
             'qa_routing': [{'id': r['id'], 'route': 'fresh_conditional' if r['id'] in conditional else 'fresh',
                             'content_sha256': r['content_sha256'], 'context_sha256': r['context_sha256']}
                            for r in sorted(snapshot, key=lambda r: r['id']) if r['field'] == 'acquisition_records'],
             'carryover_audit': seeded_audit_plan({i: [] for i in range(1, 9)}),
             'claims_forfeited': protocol['claims_forfeited'], 'timing_disclosure': protocol['review_rule_change_disclosure'],
             'semantic_approvals_created': False, 'confirmation_ready': False}
    proof['sha256'] = digest(proof);verify_protocol_coverage(proof, protocol, chunks, catalog,
        dependency_context=dependency_context,expected_evaluator_metadata=expected_evaluator_metadata,
        original_trace_input_files_by_proof=original_trace_input_files_by_proof)
    return proof
