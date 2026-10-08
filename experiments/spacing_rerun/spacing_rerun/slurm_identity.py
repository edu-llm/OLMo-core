"""Canonical scheduler identities shared by orchestration and cost guards."""
import contextlib
import signal
import re
import threading


def native_id(value, context="native Slurm ID"):
    if (type(value) is not str or not value.isascii() or not value.isdigit()
            or int(value) <= 0 or str(int(value)) != value):
        raise ValueError(context + ": canonical positive ASCII native string required")
    return value


@contextlib.contextmanager
def registration_signals():
    """Turn allocation-registration interruptions into retained failures."""
    previous = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Native registration interrupted by signal " + str(signum))
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
                previous[signum] = signal.getsignal(signum)
                signal.signal(signum, interrupted)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def atomic_registration(lock_context, operation, retain_failure):
    """Keep interrupted registration and its failure receipt under one lock.

    Defer INT/TERM/USR1 while acquiring the lock and writing the receipt.
    A pending interruption is delivered inside the protected try after the
    lock is acquired. Storage/lock acquisition failures cannot establish a
    critical section and still require exhaustive scheduler reconciliation.
    """
    signals = {signal.SIGINT, signal.SIGTERM, signal.SIGUSR1}
    maskable = (hasattr(signal, "pthread_sigmask")
                and threading.current_thread() is threading.main_thread())
    with registration_signals():
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals) if maskable else None
        failure_handled = False
        try:
            with lock_context:
                try:
                    if maskable:
                        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                    return operation()
                except BaseException as error:
                    if maskable:
                        signal.pthread_sigmask(signal.SIG_BLOCK, signals)
                    failure_handled = True
                    retain_failure(error)
                    raise
        except BaseException as error:
            if not failure_handled:
                # The lock itself was unavailable or inaccessible. Never claim
                # this fallback was serialized under a lock we did not acquire.
                retain_failure(error)
            raise
        finally:
            if maskable:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def count(value, context="scheduler count"):
    if (type(value) is not str or not value.isascii() or not value.isdigit()
            or str(int(value)) != value):
        raise ValueError(context + ": canonical nonnegative ASCII count required")
    return int(value)


def parse_tres(text):
    """Reject duplicate fields and reconcile generic/typed GPU counts."""
    if type(text) is not str:
        raise ValueError("Actual TRES text required")
    fields = {}
    for part in text.split(",") if text else ():
        if "=" not in part:
            raise ValueError("Malformed actual allocation TRES")
        key, value = part.split("=", 1)
        if not key or not value or key in fields:
            raise ValueError("Duplicate or empty actual allocation TRES")
        fields[key] = value
    typed = [count(v, "Typed GPU count") for k, v in fields.items() if k.startswith("gres/gpu:")]
    generic = count(fields["gres/gpu"], "Generic GPU count") if "gres/gpu" in fields else None
    if generic is not None and typed and generic != sum(typed):
        raise ValueError("Actual typed and generic GPU counts disagree")
    if "cpu" in fields:
        count(fields["cpu"], "TRES CPU count")
    return fields, generic if generic is not None else sum(typed)


def raw_allocation_counts(tres_text, allocated_cpus, elapsed, state, start=None):
    """Reconcile the two raw CPU fields before accounting can be complete."""
    fields, gpu = parse_tres(tres_text)
    cpu = count(allocated_cpus, "AllocCPUS")
    seconds = count(elapsed, "ElapsedRaw")
    unstarted = (state == "CANCELLED" and start in ("Unknown", "None", "")
                 and seconds == 0 and cpu == 0 and gpu == 0)
    if ("cpu" in fields and count(fields["cpu"], "TRES CPU count") != cpu
            or "cpu" not in fields and not unstarted):
        raise ValueError("Actual AllocTRES CPU and AllocCPUS disagree or CPU proof is missing")
    if cpu == 0 and not unstarted:
        raise ValueError("Zero actual CPU count requires an explicitly unstarted cancelled allocation")
    return seconds, gpu, cpu


def verify_live_tres(fields, prescription):
    """Match requested resources while retaining memory-expanded allocated CPUs."""
    allocated, gpu = parse_tres(fields.get("AllocTRES", fields.get("TRES", "")))
    requested, requested_gpu = parse_tres(fields.get("ReqTRES", ""))
    cpu = count(fields.get("NumCPUs", ""), "NumCPUs")
    maximum = prescription.get("maximum_allocated_cpus")
    nodes = prescription.get("measured_nodes")
    if (type(maximum) is not int or maximum < 4 or type(nodes) is not list or not nodes
            or any(type(node) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", node) for node in nodes)
            or len(nodes) != len(set(nodes)) or "wheat-01" in nodes):
        raise ValueError("Actual measured allocated CPU maximum and exact node set required")
    actual_node = fields.get("NodeList")
    if (actual_node not in nodes or fields.get("BatchHost", actual_node) != actual_node
            or "NumNodes" in fields and fields["NumNodes"] != "1"):
        raise ValueError("Actual allocation node differs from the measured prescription")
    memory = str(prescription["memory_gb"]) + "G"
    if (allocated.get("gres/gpu") != "1" or gpu != 1 or requested.get("gres/gpu") != "1"
            or requested_gpu != 1 or requested.get("cpu") != "4" or not 4 <= cpu <= maximum
            or allocated.get("cpu") != str(cpu) or requested.get("mem") != memory
            or allocated.get("mem") != memory or fields.get("MinMemoryNode") != memory):
        raise ValueError("Actual GPU/CPU/memory allocation differs from frozen prescription")
    return {"allocated_cpu_count": cpu, "allocated_gpu_count": gpu,
            "requested_cpu_count": 4, "requested_memory": memory}
