import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from spacing_rerun.common import digest
from spacing_rerun.final_span import cost_report
from spacing_rerun.schedule import ARMS

COMMIT='1b6f6e99d99b9a2d1c3ced6bb9c39fbffc4f9ec7'
FIELDS='JobIDRaw,JobID,State%40,ElapsedRaw,AllocTRES%300,AllocCPUS,Submit,Start,End'
HEADER='JobIDRaw|JobID|State|ElapsedRaw|AllocTRES|AllocCPUS|Submit|Start|End\n'
# Explicitly synthetic query provenance: no scheduler ran, no production proof is restamped.
QUERIED='2026-10-08T11:00:00+00:00'
CAPTURED='2026-10-08T11:00:01+00:00'
def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value,sealed=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    value=dict(value)
    if sealed:value['sha256']=digest(value)
    path.write_text(json.dumps(value));return value
def row(path):return {'path':str(path.resolve()),'file_sha256':sha(path)}
def clock(seconds):
    hour,rest=divmod(8*3600+seconds,3600);return f'2026-10-08T{hour:02}:{rest//60:02}:{rest%60:02}'

class OuterCostTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'normal-01'
        self.bundle={'bundle_path':str(self.root),'trial_id':'normal-01','manifest_sha256':'manifest','attempts':[]}
        self.account={'schema':'spacing-slurm-allocation-accounting-v2','outer_arm_infrastructure_accounting_required':True,
                      'observed_outer_arm_launch_inputs':[],'observed_arm_attempt_inputs':[],'allocations':[]}
        self.raw_overrides={}
        for i,arm in enumerate(ARMS,201):
            if arm=='UNI':self.start(arm,'101',1,seconds=600)
            self.start(arm,str(i),2 if arm=='UNI' else 1)
            p=self.root/arm/'attempt-01.json';attempt=write(p,{'slurm_job_id':str(i),'wall_seconds':100.,'gpu_count':1,'progress':{'arm':arm}})
            self.bundle['attempts'].append(attempt);self.account['observed_arm_attempt_inputs'].append(dict(trial='normal-01',arm=arm,slurm_job_id=str(i),**row(p)))
        self.path=self.root/'allocation.json'

    def start(self,arm,job,number,seconds=120,restart=0):
        directory=self.root/'infrastructure-launches'/arm
        prior=sorted(directory.glob('launch-*.json'))
        p=directory/f'launch-{job}-restart-{restart}.json'
        write(p,{'schema':'p4-final-span-actual-arm-infrastructure-launch-v1','maximum_infrastructure_starts':3,'session_attempt_counter_unchanged':True,
                 'numerical_commit':COMMIT,'manifest_sha256':'manifest','trial_index':1,'arm':arm,'slurm_job_id':job,'scheduler_restarts':restart,
                 'infrastructure_start':number,'prior_actual_starts':[row(x) for x in prior],
                 'actual_scontrol_command':['scontrol','show','job','--oneliner',job],'actual_scontrol_stdout':f'JobId={job} Restarts={restart}',
                 'actual_utc':'2026-10-08T08:00:00+00:00'},True)
        self.account['observed_outer_arm_launch_inputs'].append(dict(trial='normal-01',arm=arm,slurm_job_id=job,**row(p)))
        if job not in {r['job_id'] for r in self.account['allocations']}:
            self.account['allocations'].append({'job_id':job,'elapsed_seconds':seconds,'allocated_gpu_count':1,'source':'sacct','state':'FAILED' if job=='101' else 'COMPLETED'})
        return p

    def raw_fields(self,value):
        fields={'JobIDRaw':str(value['job_id']),'JobID':str(value['job_id']),'State':value['state'],'ElapsedRaw':str(value['elapsed_seconds']),
                'AllocTRES':f"cpu=4,gres/gpu={value['allocated_gpu_count']}",'AllocCPUS':'4',
                'Submit':'2026-10-08T07:59:00','Start':clock(0),'End':clock(value['elapsed_seconds'])}
        fields.update(self.raw_overrides.get(value['job_id'],{}));return fields

    def rebind(self,text=None,jobs=None,proof=None,**account):
        """Write raw bytes and an explicitly synthetic v2 provenance binding, then the accounting file."""
        raw=self.root/'raw-sacct.txt'
        if text is None:
            text=HEADER+''.join('|'.join(self.raw_fields(value)[k] for k in HEADER[:-1].split('|'))+'\n' for value in self.account['allocations'])
        raw.write_text(text)
        jobs=jobs if jobs is not None else [str(value['job_id']) for value in self.account['allocations']]
        command=['sacct','-X','-P','-j',','.join(jobs),'-o',FIELDS]
        self.account['raw_sacct']=dict(row(raw),bytes=raw.stat().st_size,actual_command=command,actual_environment={'TZ':'UTC'},
            unset_environment=['SLURM_TIME_FORMAT'],scheduler_timezone='UTC',queried_utc=QUERIED,synthetic_fixture=True)
        self.account['raw_sacct'].update(proof or {})
        provenance=dict(self.account['raw_sacct'],schema='p4-actual-sacct-query-provenance-v1',returncode=0,
                        stderr='',stderr_sha256=hashlib.sha256(b'').hexdigest())
        provenance_path=self.root/'synthetic-sacct-query-provenance.json'
        provenance=write(provenance_path,provenance,True)
        self.account['raw_sacct_provenance']=dict(row(provenance_path),payload_sha256=provenance['sha256'])
        self.account.update(captured_utc=CAPTURED,scheduler_timezone='UTC',actual_sacct_arguments=command)
        self.account.update(account)
        write(self.path,self.account);return raw

    def cost(self):
        for value in self.account['allocations']:value.setdefault('allocated_cpu_count',4)
        self.rebind();return cost_report([self.bundle],self.path,chosen_n=8)

    def test_pre_session_failure_in_denominator_and_largest_bundle_projection(self):
        self.account['allocations'].append({'job_id':'999','elapsed_seconds':10000,'allocated_gpu_count':1,'source':'sacct','state':'COMPLETED'})
        packet=self.cost();self.assertAlmostEqual(packet['allocation_gpu_hours'],1200/3600)
        self.assertAlmostEqual(packet['largest_observed_five_arm_bundle_gpu_hours'],1200/3600)
        self.assertAlmostEqual(packet['confirmation_cost_projection']['arm_gpu_hours_at_largest_observed_bundle'],8*1200/3600)
        self.assertTrue(packet['pre_session_arm_failures_included']);self.assertTrue(packet['outer_infrastructure_accounting_complete'])
        self.assertFalse(packet['confirmation_cost_projection']['full_experiment_budget_ready'])

    def test_missing_failed_allocation_and_missing_outer_proof_rejected(self):
        self.account['allocations']=[r for r in self.account['allocations'] if r['job_id']!='101']
        with self.assertRaisesRegex(ValueError,'omits'):self.cost()
        self.account['allocations'].append({'job_id':'101','elapsed_seconds':600,'allocated_gpu_count':1,'source':'sacct','state':'FAILED'})
        self.account['observed_outer_arm_launch_inputs'].pop(0)
        with self.assertRaisesRegex(ValueError,'missing from accounting'):self.cost()

    def test_forged_stream_scheduler_and_changed_bytes_rejected(self):
        path=Path(self.account['observed_outer_arm_launch_inputs'][0]['path']);original=json.loads(path.read_text())
        for key,value in (('manifest_sha256','forged'),('arm','GEN'),('actual_scontrol_stdout','JobId=999 Restarts=0')):
            changed=dict(original);changed.pop('sha256');changed[key]=value;write(path,changed,True)
            self.account['observed_outer_arm_launch_inputs'][0].update(row(path))
            with self.subTest(key=key),self.assertRaises(ValueError):self.cost()
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.cost()

    def test_session_job_outside_own_outer_stream_and_extra_citation_rejected(self):
        p=Path(self.account['observed_arm_attempt_inputs'][0]['path']);value=json.loads(p.read_text());value['slurm_job_id']='202';write(p,value)
        self.account['observed_arm_attempt_inputs'][0].update(row(p),slurm_job_id='202')
        with self.assertRaisesRegex(ValueError,'own actual outer'):self.cost()

    def test_same_native_id_restarts_count_once_and_duplicate_rows_fail(self):
        self.start('UNI','202',3,restart=1)
        self.assertAlmostEqual(self.cost()['allocation_gpu_hours'],1200/3600)
        self.account['allocations'].append(dict(self.account['allocations'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate Slurm'):self.cost()

    def test_extra_outer_citation_or_native_job_reuse_across_arms_rejected(self):
        self.account['observed_outer_arm_launch_inputs'].append(dict(self.account['observed_outer_arm_launch_inputs'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate outer'):self.cost()
        self.account['observed_outer_arm_launch_inputs'].pop()
        self.start('NONE','202',2)
        with self.assertRaisesRegex(ValueError,'reused across'):self.cost()

    def test_over_budget_native_rejection_cannot_disappear_from_eligibility(self):
        write(self.root/'infrastructure-launches/UNI/rejected-999-restart-0.json',{'slurm_job_id':'999','numerical_child_started':False},True)
        with self.assertRaisesRegex(ValueError,'Over-budget native'):self.cost()

    def test_optional_outer_omission_and_normalized_duplicate_fail(self):
        self.account['outer_arm_infrastructure_accounting_required']=False
        with self.assertRaisesRegex(ValueError,'complete outer'):self.cost()
        self.account['outer_arm_infrastructure_accounting_required']=True
        self.account['allocations'].append(dict(self.account['allocations'][0],job_id=int(self.account['allocations'][0]['job_id'])))
        with self.assertRaisesRegex(ValueError,'Duplicate Slurm'):self.cost()

    def test_reverse_same_native_restart_order_fails(self):
        self.start('NONE','201',2,restart=1)
        path=self.root/'infrastructure-launches/NONE/launch-201-restart-0.json';record=json.loads(path.read_text());record.pop('sha256');record['scheduler_restarts']=2;record['actual_scontrol_stdout']='JobId=201 Restarts=2'
        replacement=path.with_name('launch-201-restart-2.json');path.unlink();write(replacement,record,True)
        for value in self.account['observed_outer_arm_launch_inputs']:
            if value['path']==str(path.resolve()):value.update(row(replacement))
        with self.assertRaisesRegex(ValueError,'restart sequence'):self.cost()

    def test_resealed_reverse_chronology_rejected(self):
        first=self.root/'infrastructure-launches/UNI/launch-101-restart-0.json'
        second=self.root/'infrastructure-launches/UNI/launch-202-restart-0.json'
        value=json.loads(first.read_text());value.pop('sha256');value['actual_utc']='2026-10-08T09:00:00+00:00';write(first,value,True)
        value=json.loads(second.read_text());value.pop('sha256');value['prior_actual_starts']=[row(first)];write(second,value,True)
        for proof in self.account['observed_outer_arm_launch_inputs']:
            proof.update(row(Path(proof['path'])))
        with self.assertRaisesRegex(ValueError,'chronology'):self.cost()

    def test_native_aliases_at_outer_and_session_boundaries_rejected(self):
        first=self.root/'infrastructure-launches/UNI/launch-101-restart-0.json'
        value=json.loads(first.read_text());value.pop('sha256');value['slurm_job_id']='0101';write(first,value,True)
        next(proof for proof in self.account['observed_outer_arm_launch_inputs'] if proof['path']==str(first.resolve())).update(row(first))
        with self.assertRaisesRegex(ValueError,'canonical'):self.cost()
        value['slurm_job_id']='101';write(first,value,True)
        next(proof for proof in self.account['observed_outer_arm_launch_inputs'] if proof['path']==str(first.resolve())).update(row(first))
        session=Path(self.account['observed_arm_attempt_inputs'][0]['path']);value=json.loads(session.read_text());value['slurm_job_id']='0201';write(session,value)
        self.account['observed_arm_attempt_inputs'][0].update(row(session))
        with self.assertRaisesRegex(ValueError,'canonical'):self.cost()

    def test_raw_cpu_tres_contradictions_deny_complete_cost(self):
        self.cost();raw=Path(self.account['raw_sacct']['path']);baseline=raw.read_text()
        for cpu in ('cpu=1','cpu=4,cpu=4','cpu=00','cpu=-1',''):
            changed=baseline.replace('cpu=4,',cpu+',' if cpu else '',1)
            self.rebind(changed)
            with self.subTest(cpu=cpu),self.assertRaises(ValueError):cost_report([self.bundle],self.path)

    def test_raw_scheduler_hash_native_coverage_and_cost_must_match(self):
        self.cost();raw=Path(self.account['raw_sacct']['path'])
        raw.write_text(raw.read_text().replace('gres/gpu=1','gres/gpu=2',1))
        with self.assertRaisesRegex(ValueError,'raw sacct bytes'):cost_report([self.bundle],self.path)
        self.rebind(raw.read_text())
        with self.assertRaisesRegex(ValueError,'declared allocation'):cost_report([self.bundle],self.path)
        changed=raw.read_text().replace('gres/gpu=2','gres/gpu=1',1).replace('101|101','9999|9999',1)
        self.rebind(changed)
        with self.assertRaisesRegex(ValueError,'exact retained query IDs'):cost_report([self.bundle],self.path)
        jobs=['9999' if value['job_id']=='101' else value['job_id'] for value in self.account['allocations']]
        self.rebind(changed,jobs=jobs)
        with self.assertRaisesRegex(ValueError,'allocation coverage'):cost_report([self.bundle],self.path)

    # R4 F18: raw chronology and query provenance at the core complete-accounting boundary.

    def test_query_sidecar_bytes_schema_seal_and_success_required(self):
        self.assertTrue(self.cost()['accounting_complete'])
        baseline=json.loads(json.dumps(self.account))
        binding=baseline['raw_sacct_provenance'];path=Path(binding['path']);original=path.read_bytes()
        for case in ('missing_binding','missing_file','changed_bytes','schema','seal','payload_binding','returncode','bool_returncode'):
            account=json.loads(json.dumps(baseline));path.write_bytes(original)
            provenance=json.loads(original)
            if case=='missing_binding':account.pop('raw_sacct_provenance')
            elif case=='missing_file':account['raw_sacct_provenance']['path']=str(path.with_name('absent.json'))
            elif case=='changed_bytes':path.write_bytes(original+b'\n')
            else:
                if case=='schema':provenance['schema']='unsupported-provenance'
                elif case=='seal':provenance['sha256']='forged'
                elif case=='payload_binding':account['raw_sacct_provenance']['payload_sha256']='forged'
                elif case=='returncode':provenance['returncode']=1
                elif case=='bool_returncode':provenance['returncode']=False
                if case not in ('seal','payload_binding'):
                    provenance.pop('sha256');provenance=write(path,provenance,True)
                else:write(path,provenance)
                account['raw_sacct_provenance'].update(row(path))
                if case!='payload_binding':account['raw_sacct_provenance']['payload_sha256']=provenance['sha256']
            write(self.path,account)
            with self.subTest(case=case),self.assertRaisesRegex(ValueError,'provenance sidecar|query did not succeed'):
                cost_report([self.bundle],self.path)

    def test_resealed_packet_cannot_contradict_unchanged_query_sidecar(self):
        self.cost();baseline=json.loads(json.dumps(self.account));path=Path(baseline['raw_sacct_provenance']['path']);original=path.read_bytes()
        # Each replacement is internally valid; only the unchanged sidecar exposes the conflict.
        raw=Path(baseline['raw_sacct']['path']);alternative=raw.with_name('same-bytes-different-path.txt');alternative.write_bytes(raw.read_bytes())
        cases=[{'path':str(alternative)}, {'file_sha256':'f'*64}, {'bytes':raw.stat().st_size+1},
               {'actual_command':list(baseline['raw_sacct']['actual_command'][:-1])+[FIELDS.replace('State%40','State%80')]},
               {'actual_environment':{'TZ':'Etc/UTC'},'scheduler_timezone':'Etc/UTC'},
               {'unset_environment':[]}, {'queried_utc':'2026-10-08T11:00:00.500000+00:00'}]
        for changes in cases:
            account=json.loads(json.dumps(baseline));account['raw_sacct'].update(changes)
            account['actual_sacct_arguments']=account['raw_sacct']['actual_command'];account['scheduler_timezone']=account['raw_sacct']['scheduler_timezone']
            write(self.path,account,True)
            with self.subTest(changes=changes),self.assertRaises(ValueError):cost_report([self.bundle],self.path)
            self.assertEqual(path.read_bytes(),original)

    def test_resealed_sidecar_cannot_contradict_accounting_proof(self):
        self.cost();baseline=json.loads(json.dumps(self.account));path=Path(baseline['raw_sacct_provenance']['path']);original=path.read_bytes()
        for field,value in [('path','/synthetic/another-raw.txt'),('file_sha256','f'*64),('bytes',1),
                            ('actual_command',['sacct','-X','-P','-j','101','-o',FIELDS]),
                            ('actual_environment',{'TZ':'Etc/UTC'}),('scheduler_timezone','Etc/UTC'),
                            ('unset_environment',[]),('queried_utc','2026-10-08T11:00:00.500000+00:00')]:
            provenance=json.loads(original);provenance.pop('sha256');provenance[field]=value
            provenance=write(path,provenance,True);account=json.loads(json.dumps(baseline))
            account['raw_sacct_provenance']=dict(row(path),payload_sha256=provenance['sha256']);write(self.path,account,True)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'differs from the retained query provenance sidecar'):
                cost_report([self.bundle],self.path)

    def test_r4_exact_reversed_and_non_time_exploits_rejected(self):
        self.assertTrue(self.cost()['accounting_complete'])
        # R4 corrected reproduction: native101 End precedes Start, raw SHA and accounting rebound.
        self.raw_overrides['101']={'Submit':'2026-10-08T00:59:00','Start':'2026-10-08T03:00:00','End':'2026-10-08T01:00:00'}
        self.rebind()
        with self.assertRaisesRegex(ValueError,'chronology is impossible'):cost_report([self.bundle],self.path,chosen_n=8)
        for start,end,message in (('gibberish','gibberish','YYYY-MM-DDTHH:MM:SS'),('2026-02-30T00:00:00','2026-10-08T08:10:00','valid calendar'),
                                  ('2026-10-08T08:00:00Z','2026-10-08T08:10:00','YYYY'),('2026-10-08 08:00:00','2026-10-08T08:10:00','YYYY'),
                                  ('','2026-10-08T08:10:00','YYYY')):
            self.raw_overrides['101']={'Start':start,'End':end};self.rebind()
            with self.subTest(start=start),self.assertRaisesRegex(ValueError,message):cost_report([self.bundle],self.path)
        # R4 exact legacy form: Start/End appended to a provenance-free raw proof under schema v1.
        self.raw_overrides.clear();self.cost()
        lines=HEADER.replace('|Submit|Start|End','').strip()+'|Start|End\n'
        for value in self.account['allocations']:
            fields=self.raw_fields(value);times='|2026-10-08T03:00:00|2026-10-08T01:00:00' if value['job_id']=='101' else '|2026-10-08T01:00:00|2026-10-08T01:10:00'
            lines+='|'.join(fields[k] for k in ('JobIDRaw','JobID','State','ElapsedRaw','AllocTRES','AllocCPUS'))+times+'\n'
        raw=self.root/'raw-sacct.txt';raw.write_text(lines);legacy=dict(self.account,raw_sacct=row(raw))
        for key in ('captured_utc','scheduler_timezone','actual_sacct_arguments'):legacy.pop(key)
        write(self.path,legacy)
        with self.assertRaisesRegex(ValueError,'provenance'):cost_report([self.bundle],self.path)
        write(self.path,dict(legacy,schema='spacing-slurm-allocation-accounting-v1'))
        with self.assertRaisesRegex(ValueError,'v2 raw query provenance'):cost_report([self.bundle],self.path)

    def test_missing_or_false_query_provenance_and_capture_rejected(self):
        self.assertTrue(self.cost()['outer_infrastructure_accounting_complete'])
        future='2099-01-01T00:00:00+00:00'
        cases=[({'actual_environment':{'TZ':'America/Los_Angeles'}},{},'actual_environment'),
               ({'actual_environment':{}},{},'actual_environment'),
               ({'actual_environment':{'TZ':'UTC','SLURM_TIME_FORMAT':'relative'}},{},'actual_environment'),
               ({'scheduler_timezone':'PST'},{},'IANA'),({'scheduler_timezone':'Local'},{},'IANA'),({'scheduler_timezone':None},{},'IANA'),
               ({'unset_environment':[]},{},'SLURM_TIME_FORMAT'),
               ({'queried_utc':'2026-10-08T11:00:00'},{},'queried_utc'),({'queried_utc':'2026-10-08T04:00:00-07:00'},{},'queried_utc'),
               ({'queried_utc':'2026-10-08T11:00:00Z'},{},'queried_utc'),({'queried_utc':future},{'captured_utc':future},'future'),
               ({'queried_utc':'2026-10-08T08:05:00+00:00'},{},'after its own query time'),
               ({},{'captured_utc':None},'captured_utc'),({},{'captured_utc':'2026-10-08T10:59:59+00:00'},'after the accounting capture'),
               ({},{'captured_utc':future},'future'),({},{'captured_utc':'2026-10-08T11:00:01'},'captured_utc'),
               ({},{'scheduler_timezone':'America/Los_Angeles'},'different scheduler query'),
               ({},{'actual_sacct_arguments':['sacct','-X','-P','-j','101','-o',FIELDS]},'different scheduler query'),
               ({'actual_command':['sacct','-P','-j','101','-o',FIELDS]},{},'exact allocation-level'),
               ({'actual_command':['sacct','-X','-P','--starttime','2026-10-08T00:00:00','-o',FIELDS]},{},'exact allocation-level'),
               ({'bytes':1},{},'byte count')]
        for proof,account,message in cases:
            with self.subTest(proof=proof,account=account):
                self.rebind(proof=proof,**account)
                if 'actual_command' in proof:self.account['actual_sacct_arguments']=proof['actual_command'];write(self.path,self.account)
                with self.assertRaisesRegex(ValueError,message):cost_report([self.bundle],self.path)
        for key in ('actual_command','actual_environment','unset_environment','scheduler_timezone','queried_utc'):
            self.rebind();self.account['raw_sacct'].pop(key);write(self.path,self.account)
            with self.subTest(missing=key),self.assertRaisesRegex(ValueError,'provenance required'):cost_report([self.bundle],self.path)
        self.rebind();self.account.pop('captured_utc');write(self.path,self.account)
        with self.assertRaisesRegex(ValueError,'captured_utc'):cost_report([self.bundle],self.path)
        # Raw header must equal the retained queried fields.
        self.rebind(HEADER.replace('Submit','Eligible')+'101|101|FAILED|600|cpu=4,gres/gpu=1|4|2026-10-08T07:59:00|2026-10-08T08:00:00|2026-10-08T08:10:00\n')
        with self.assertRaisesRegex(ValueError,'header differs'):cost_report([self.bundle],self.path)

    def test_ordering_elapsed_unknown_and_terminal_rows_rejected(self):
        self.cost()
        cases=[({'Submit':'2026-10-08T08:00:01'},'chronology is impossible'),
               ({'Start':'2026-10-08T08:20:00','End':'2026-10-08T08:10:00'},'chronology is impossible'),
               ({'Submit':'Unknown'},'Submit is Unknown'),({'Submit':'None'},'Submit is Unknown'),
               ({'End':'2026-10-08T08:09:59'},'ElapsedRaw exceeds End-Start'),
               ({'Start':'Unknown'},'Start is Unknown/None'),({'Start':'None','End':'None'},'Start is Unknown/None'),
               ({'End':'Unknown'},'End is Unknown/None'),({'End':'None'},'End is Unknown/None')]
        for override,message in cases:
            self.raw_overrides['101']=override;self.rebind()
            with self.subTest(override=override),self.assertRaisesRegex(ValueError,message):cost_report([self.bundle],self.path)
        self.raw_overrides.clear()
        # Every raw/declared row must be terminal, including additional non-arm rows.
        self.account['allocations'].append({'job_id':'997','elapsed_seconds':60,'allocated_gpu_count':0,'allocated_cpu_count':4,'source':'sacct','state':'RUNNING'})
        self.raw_overrides['997']={'State':'RUNNING','AllocTRES':'cpu=4'}
        self.rebind()
        with self.assertRaisesRegex(ValueError,'not terminal'):cost_report([self.bundle],self.path)

    def test_dst_ambiguous_and_nonexistent_wall_times_rejected_explicit_zone_accepted(self):
        self.cost();zone='America/Los_Angeles'
        local={value['job_id']:{'Submit':'2026-10-08T00:59:00','Start':'2026-10-08T01:00:00',
                                'End':f"2026-10-08T01:{value['elapsed_seconds']//60:02}:00"} for value in self.account['allocations']}
        def bind(**overrides):
            self.raw_overrides=json.loads(json.dumps(local));self.raw_overrides['101'].update(overrides)
            self.rebind(proof={'actual_environment':{'TZ':zone},'scheduler_timezone':zone},scheduler_timezone=zone)
        bind();self.assertTrue(cost_report([self.bundle],self.path)['accounting_complete'])
        for override,message in (({'Submit':'2025-11-02T01:30:00'},'ambiguous or nonexistent'),({'Submit':'2026-03-08T02:30:00'},'ambiguous or nonexistent')):
            bind(**override)
            with self.subTest(override=override),self.assertRaisesRegex(ValueError,message):cost_report([self.bundle],self.path)
        # Wall times read in the wrong zone would end after the query; never reinterpreted.
        bind(Start='2026-10-08T04:00:00',End='2026-10-08T04:10:00')
        with self.assertRaisesRegex(ValueError,'after its own query time'):cost_report([self.bundle],self.path)

    def test_genuinely_unstarted_cancelled_and_compressed_records_accepted_without_dates(self):
        for job,display,start,end in (('996','996','Unknown','Unknown'),('997','997_[1-4]','None','None'),('998','998','Unknown','2026-10-08T08:30:00')):
            self.account['allocations'].append({'job_id':job,'elapsed_seconds':0,'allocated_gpu_count':0,'allocated_cpu_count':0,'source':'sacct','state':'CANCELLED',
                                                'scheduler_submit':'2026-10-08T07:59:00','scheduler_start':start,'scheduler_end':end})
            self.raw_overrides[job]={'JobID':display,'AllocTRES':'','AllocCPUS':'0','Start':start,'End':end,'State':'CANCELLED by 1'}
        packet=self.cost();self.assertTrue(packet['accounting_complete']);self.assertAlmostEqual(packet['allocation_gpu_hours'],1200/3600)
        raw=Path(self.account['raw_sacct']['path']).read_text();self.assertIn('997|997_[1-4]|CANCELLED by 1|0||0|2026-10-08T07:59:00|None|None',raw)
        # A started or costed cancellation cannot use the unknown-time exception.
        self.raw_overrides['996'].update(AllocTRES='cpu=4',AllocCPUS='4');self.account['allocations'][-3]['allocated_cpu_count']=4
        self.rebind()
        with self.assertRaisesRegex(ValueError,'Start is Unknown/None'):cost_report([self.bundle],self.path)

    def test_declared_timestamps_must_match_actual_raw_values(self):
        for value in self.account['allocations']:
            value.update(scheduler_submit='2026-10-08T07:59:00',scheduler_start=clock(0),scheduler_end=clock(value['elapsed_seconds']))
        self.assertTrue(self.cost()['accounting_complete'])
        self.account['allocations'][0]['scheduler_end']='2026-10-08T09:00:00'
        self.rebind()
        with self.assertRaisesRegex(ValueError,'Declared scheduler timestamps differ'):cost_report([self.bundle],self.path)
        self.account['allocations'][0]['scheduler_end']=clock(self.account['allocations'][0]['elapsed_seconds'])
        self.account['allocations'][0]['scheduler_start_utc']='2026-10-08T08:00:00+00:00'
        self.rebind()
        with self.assertRaisesRegex(ValueError,'Undeclared scheduler timestamp'):cost_report([self.bundle],self.path)

if __name__=='__main__':unittest.main()
