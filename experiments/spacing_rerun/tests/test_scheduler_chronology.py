import datetime
import unittest

from spacing_rerun import scheduler_chronology as sc

FIELDS='JobIDRaw,JobID,State%40,ElapsedRaw,AllocTRES%300,AllocCPUS,Submit,Start,End'
UTC=sc.retained_zone('UTC')


class ChronologyHelperTest(unittest.TestCase):
    def test_producer_environment_forces_retained_zone_and_removes_time_format(self):
        base={'TZ':'America/Los_Angeles','SLURM_TIME_FORMAT':'relative','PATH':'/bin'}
        environment=sc.query_environment(base,'UTC')
        self.assertEqual(environment,{'TZ':'UTC','PATH':'/bin'});self.assertEqual(base['SLURM_TIME_FORMAT'],'relative')
        self.assertEqual(sc.environment_provenance(environment),
                         {'actual_environment':{'TZ':'UTC'},'scheduler_timezone':'UTC','unset_environment':['SLURM_TIME_FORMAT']})
        self.assertEqual(sc.environment_provenance(base)['unset_environment'],[])
        with self.assertRaisesRegex(ValueError,'IANA'):sc.query_environment(base,'PST')

    def test_exact_command_and_parsable_output_only(self):
        self.assertEqual(sc.sacct_command(['sacct','-X','-P','-j','7,12','-o',FIELDS])[0],['7','12'])
        for command in (['sacct','-P','-X','-j','7','-o',FIELDS],['sacct','-X','-P','-j','07','-o',FIELDS],['sacct','-X','-P','-j','7,7','-o',FIELDS],
                        ['sacct','-X','-P','-j','7','-o','JobIDRaw,State,ElapsedRaw,AllocTRES,AllocCPUS,Start,End'],
                        ['sacct','-X','-P','-j','7','-o',FIELDS,'--units=K'],'sacct -X -P -j 7'):
            with self.subTest(command=command),self.assertRaises(ValueError):sc.sacct_command(command)
        columns=sc.sacct_command(['sacct','-X','-P','-j','7','-o',FIELDS])[1]
        header='|'.join(columns)
        self.assertEqual(sc.parse_sacct(header+'\n7|7|COMPLETED|1|cpu=1|1|a|b|c\n',columns)[0]['End'],'c')
        for text in (header+'\n7|7|COMPLETED|1|cpu=1|1|a|b|c',header+'\r\n',header+'\n7|7|COMPLETED|1|cpu=1|1|a|b|c|extra\n',
                     header.replace('Submit','Eligible')+'\n'):
            with self.subTest(text=text),self.assertRaises(ValueError):sc.parse_sacct(text,columns)

    def test_wall_clock_dst_and_explicit_utc(self):
        zone=sc.retained_zone('America/Los_Angeles')
        self.assertEqual(sc.wall('2026-10-08T01:00:00','t',zone),datetime.datetime(2026,10,8,8,tzinfo=datetime.timezone.utc))
        for text in ('2025-11-02T01:30:00','2026-03-08T02:30:00'):
            with self.subTest(text=text),self.assertRaisesRegex(ValueError,'ambiguous or nonexistent'):sc.wall(text,'t',zone)
        for text in ('2026-10-08T08:00:00','2026-10-08T08:00:00Z','2026-10-08T01:00:00-07:00','2099-01-01T00:00:00+00:00',None):
            with self.subTest(text=text),self.assertRaises(ValueError):sc.explicit_utc(text,'t')

    def test_capture_is_required_and_unstarted_cancelled_exception_is_narrow(self):
        proof={'actual_command':['sacct','-X','-P','-j','7','-o',FIELDS],'actual_environment':{'TZ':'UTC'},
               'unset_environment':['SLURM_TIME_FORMAT'],'scheduler_timezone':'UTC','queried_utc':'2026-10-08T04:00:00+00:00'}
        with self.assertRaisesRegex(ValueError,'captured_utc'):sc.query_provenance(proof,None)
        queried=sc.query_provenance(proof,'2026-10-08T04:00:00+00:00')['queried']
        row={'Submit':'2026-10-08T02:00:00','Start':'Unknown','End':'Unknown'}
        self.assertTrue(sc.allocation_chronology(row,'7',UTC,queried,state='CANCELLED',elapsed=0,cpu=0,gpu=0)['unstarted_cancelled'])
        for state,elapsed,cpu,gpu in (('FAILED',0,0,0),('CANCELLED',1,0,0),('CANCELLED',0,1,0),('CANCELLED',0,0,1)):
            with self.subTest(state=state,elapsed=elapsed,cpu=cpu,gpu=gpu),self.assertRaisesRegex(ValueError,'Start is Unknown'):
                sc.allocation_chronology(row,'7',UTC,queried,state=state,elapsed=elapsed,cpu=cpu,gpu=gpu)
        started={'Submit':'2026-10-08T02:00:00','Start':'2026-10-08T03:00:00','End':'2026-10-08T03:00:10'}
        self.assertFalse(sc.allocation_chronology(started,'7',UTC,queried,state='COMPLETED',elapsed=10,cpu=1,gpu=1)['unstarted_cancelled'])
        with self.assertRaisesRegex(ValueError,'exceeds'):sc.allocation_chronology(started,'7',UTC,queried,state='COMPLETED',elapsed=11,cpu=1,gpu=1)


if __name__=='__main__':unittest.main()
