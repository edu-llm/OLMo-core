"""Strict raw sacct query provenance and allocation chronology (R4 F18); stdlib only, fails closed.

Shared by the development accounting producer (Report operator) and the core raw-accounting
verifier. It reuses the reviewed MainV9 full-study ledger chronology rules (R3 F16) without
importing or changing that ledger:

  * raw Submit/Start/End are sacct local wall-clock seconds in one explicit retained IANA
    scheduler_timezone, the zone sacct actually ran under: actual_environment is exactly
    {"TZ": scheduler_timezone} and SLURM_TIME_FORMAT was removed (unset_environment);
  * DST-ambiguous or skipped wall times are rejected, never guessed;
  * queried_utc is explicit UTC, not in the future, at or after every raw timestamp, and at or
    before the caller's required explicit UTC captured_utc (never waived);
  * every row needs a known Submit and possible Submit<=Start<=End with ElapsedRaw<=End-Start;
  * Unknown/None Start/End is accepted only for a genuinely unstarted zero-allocation CANCELLED
    record (Start Unknown/None, ElapsedRaw, CPU and GPU all zero) and is never filled in.

No cutoff is invented for this development interface. The query must be the exact
allocation-level command sacct -X -P -j <canonical native IDs> -o <fields>; raw output must be
complete newline-terminated parsable text whose header equals the queried field names.

Free of package-relative imports so identical bytes serve the core package and flat operator
packages. Raw CPU/GPU counts remain the caller's slurm_identity.raw_allocation_counts result.
"""
import datetime
import re
import zoneinfo

UTC = datetime.timezone.utc
RAW_TIME = '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}'
UNKNOWN_TIME = frozenset({'Unknown', 'None'})
TIME_COLUMNS = ('Submit', 'Start', 'End')
REQUIRED_COLUMNS = ('JobIDRaw', 'State', 'ElapsedRaw', 'AllocTRES', 'AllocCPUS') + TIME_COLUMNS
UNSET_ENVIRONMENT = ('SLURM_TIME_FORMAT',)
PROVENANCE_KEYS = frozenset({'actual_command', 'actual_environment', 'unset_environment',
                             'scheduler_timezone', 'queried_utc'})
DECLARED_TIME_FIELDS = {'scheduler_submit': 'Submit', 'scheduler_start': 'Start', 'scheduler_end': 'End'}


def _require(value, message):
    if not value:
        raise ValueError(message)


def retained_zone(name, what='scheduler_timezone'):
    _require(type(name) is str and name in zoneinfo.available_timezones(),
             what + ' must be an explicit retained IANA timezone name: ' + repr(name))
    return zoneinfo.ZoneInfo(name)


def wall(text, what, zone):
    """One raw sacct local wall-clock second in the retained zone, returned as UTC; never guessed across DST."""
    _require(type(text) is str and re.fullmatch(RAW_TIME, text),
             what + ' must be a raw sacct YYYY-MM-DDTHH:MM:SS timestamp: ' + repr(text))
    try:
        naive = datetime.datetime.strptime(text, '%Y-%m-%dT%H:%M:%S')
    except ValueError:
        raise ValueError(what + ' is not a valid calendar timestamp: ' + repr(text)) from None
    early, late = naive.replace(tzinfo=zone, fold=0), naive.replace(tzinfo=zone, fold=1)
    _require(early.utcoffset() == late.utcoffset() and early.astimezone(UTC).astimezone(zone).replace(tzinfo=None) == naive,
             what + ' is ambiguous or nonexistent in ' + zone.key + ' (DST transition); never guessed: ' + text)
    return early.astimezone(UTC)


def explicit_utc(text, what):
    """An explicit +00:00 ISO instant that is not in the future."""
    _require(type(text) is str and re.fullmatch(RAW_TIME + '([.][0-9]{1,6})?[+]00:00', text),
             what + ' must be explicit UTC ISO seconds with +00:00: ' + repr(text))
    value = datetime.datetime.fromisoformat(text)
    _require(value <= datetime.datetime.now(UTC), what + ' is in the future: ' + text)
    return value


def query_environment(base, zone='UTC'):
    """Producer environment: base copy with TZ forced to the retained zone and SLURM_TIME_FORMAT removed."""
    retained_zone(zone)
    environment = dict(base)
    environment['TZ'] = zone
    for name in UNSET_ENVIRONMENT:
        environment.pop(name, None)
    return environment


def environment_provenance(environment):
    """Provenance fields derived from the exact environment actually passed to sacct."""
    return {'actual_environment': {'TZ': environment.get('TZ')}, 'scheduler_timezone': environment.get('TZ'),
            'unset_environment': [name for name in UNSET_ENVIRONMENT if name not in environment]}


def sacct_command(command):
    """Exact sacct -X -P -j <ids> -o <fields>; return (canonical native IDs, header column names)."""
    _require(type(command) is list and len(command) == 7 and all(type(v) is str for v in command) and
             command[:4] == ['sacct', '-X', '-P', '-j'] and command[5] == '-o',
             'Raw sacct proof must retain the exact allocation-level actual_command sacct -X -P -j <ids> -o <fields>')
    ids = command[4].split(',')
    _require(all(re.fullmatch('[1-9][0-9]*', v) for v in ids) and len(ids) == len(set(ids)),
             'Raw sacct query IDs must be unique canonical positive ASCII native IDs')
    columns = [field.split('%', 1)[0] for field in command[6].split(',')]
    _require(all(columns) and len(columns) == len(set(columns)) and set(REQUIRED_COLUMNS) <= set(columns),
             'Raw sacct query fields must be unique and include ' + ','.join(REQUIRED_COLUMNS))
    return ids, columns


def query_provenance(proof, captured_utc):
    """Verify retained query provenance against the caller's explicit capture time; never waived.

    Return zone, query instant, queried IDs and header columns.
    """
    _require(isinstance(proof, dict) and PROVENANCE_KEYS <= set(proof),
             'Actual scheduler query provenance required: ' + ','.join(sorted(PROVENANCE_KEYS)))
    zone = retained_zone(proof['scheduler_timezone'], 'Raw sacct scheduler_timezone')
    _require(proof['actual_environment'] == {'TZ': zone.key},
             'Raw sacct proof must retain actual_environment exactly {"TZ": the retained IANA timezone ' + zone.key + '}')
    _require(proof['unset_environment'] == list(UNSET_ENVIRONMENT),
             'Raw sacct proof must retain that SLURM_TIME_FORMAT was removed from the query environment')
    queried = explicit_utc(proof['queried_utc'], 'Raw sacct queried_utc')
    _require(queried <= explicit_utc(captured_utc, 'Accounting captured_utc'),
             'Raw sacct query time is after the accounting capture time')
    ids, columns = sacct_command(proof['actual_command'])
    return {'zone': zone, 'queried': queried, 'job_ids': ids, 'columns': columns}


def parse_sacct(text, columns):
    """Exact sacct -P rows whose header equals the queried fields; never guessed."""
    _require(type(text) is str and '\r' not in text and text.endswith('\n'),
             'Raw sacct proof must be complete newline-terminated sacct -P output')
    lines = text[:-1].split('\n')
    _require(lines[0].split('|') == list(columns), 'Raw sacct header differs from the retained queried fields')
    rows = []
    for line in lines[1:]:
        fields = line.split('|')
        _require(len(fields) == len(columns), 'Raw sacct row is not exactly parsable; never guessed')
        rows.append(dict(zip(columns, fields)))
    return rows


def raw_times(row, job, zone):
    """Parsed Submit/Start/End (UTC) with Unknown/None kept as None; never filled in."""
    times = {k: None if row[k] in UNKNOWN_TIME else wall(row[k], 'Raw ' + k + ' for ' + job, zone) for k in TIME_COLUMNS}
    _require(times['Submit'] is not None, 'Raw Submit is Unknown/None for ' + job + '; never guessed')
    known = [times[k] for k in TIME_COLUMNS if times[k] is not None]
    _require(known == sorted(known), 'Raw Submit<=Start<=End chronology is impossible for ' + job)
    return times, max(known)


def allocation_chronology(row, job, zone, queried, *, state, elapsed, cpu, gpu):
    """Possible terminal allocation chronology; returns the exact raw strings it verified."""
    times, latest = raw_times(row, job, zone)
    unstarted = state == 'CANCELLED' and times['Start'] is None and elapsed == cpu == gpu == 0
    _require(times['Start'] is not None or unstarted,
             'Raw Start is Unknown/None for a started or costed allocation: ' + job)
    _require(times['End'] is not None or unstarted,
             'Raw End is Unknown/None for a terminal started allocation: ' + job)
    _require(latest <= queried, 'Raw scheduler timestamp is after its own query time: ' + job)
    _require(times['Start'] is None or elapsed <= (times['End'] - times['Start']).total_seconds(),
             'Raw ElapsedRaw exceeds End-Start for ' + job + '; cost is never truncated')
    return {'job_id': job, 'submit': row['Submit'], 'start': row['Start'], 'end': row['End'],
            'unstarted_cancelled': unstarted}


def declared_times_match(declared, row, job):
    """Declared scheduler_* timestamp fields, when emitted, must equal the exact raw values."""
    unknown = sorted(k for k in declared if k.startswith('scheduler_') and k not in DECLARED_TIME_FIELDS)
    _require(not unknown, 'Undeclared scheduler timestamp fields for ' + job + ': ' + ','.join(unknown))
    _require(all(declared[k] == row[c] for k, c in DECLARED_TIME_FIELDS.items() if k in declared),
             'Declared scheduler timestamps differ from the actual raw sacct values for ' + job)
