"""A parallel portfolio of the cover variants (official Q&A #3: multiprocessing inside the solver is allowed).

Without the pool the main process builds four cover variants and finishes only the one with the lowest capture proxy.
With it, worker processes finish the other variants at the same time, and solve() keeps the plan with the lowest exact
score (the main plan wins ties).

The pool starts at import (outside the scored solve() time, Q&A #4) only when every condition holds; otherwise no
process is created and the solver runs alone:
  - Windows: the 'spawn' start method (also inside a daemon process such as a multiprocessing.Pool worker: the daemon
    flag is lowered only while the workers start, _allow_children). Linux: the 'fork' start method, in a process that is
    not a daemon and has no other thread (fork safety).
  - the set-cover kernels available (setcover.ENABLED: the compiled ones, or the plain-Python ones without numba)
  - usable CPUs >= MIN_CPUS: the CPU affinity (Linux: also every cgroup v2 cpu.max / v1 CFS quota from this process's
    cgroup up to the root; any read error: not usable)
  - a calibration: every process burns CALIBRATION_S at once and each gets CPU time / wall time >= MIN_EFFICIENCY;
    when it does not, the workers stay and the pool starts paused, and it is tried again after PROBE_START boards (so
    load at import time, when every process compiles, does not switch the pool off for the whole run).
Worker count: min(WORKERS, usable CPUs - 1). On Windows, when the affinity holds at least EXTRA_MIN_CORES physical
cores, EXTRA_WORKERS extra workers start after the base pool passed its calibration; they finish only the extra cover
variants the main process never builds (solver.EXTRA_VARIANTS). An extra worker's error, death or staleness, or a
failed extra start, retires the extras only; the base pool goes on.

Workers never write to the dataset folder and never start workers themselves (a spawned child is named WORKER_NAME and
skips start()); they end with their parent. Windows: while the workers start, the parent's __main__ script is hidden
(so a harness without an if __name__ == '__main__' guard is not run again in them); each child imports ARGOS (numba
compiles there); a request carries only the absolute dataset folder and the variant number, and the worker reads the
board and builds that cover itself, sending the cover's capture proxy first and the finished plan last; worker errors
are counted (ERROR_LIMIT stops the pool); stopping terminates the workers. Linux: the parent sends the board and the
built cover; writes and reads keep absolute deadlines, and a partial reply is kept and finished at the next read.
Requests carry a generation number: the parent sends only to idle workers, collects against an absolute deadline, and
drops late or foreign replies (a late reply is normal with the wait cap). A runtime guard pauses the pool when the
workers keep getting too little CPU (other load on the machine) and probes it again after some boards; a worker busy
longer than STALE_S (stuck) or a dead worker stops it. Any other failure stops the pool for the rest of the process
(the solver then runs alone)."""
import contextlib
import math
import multiprocessing as mp
import os
import signal
import sys
import threading
import time

WORKERS = 3             # at most this many worker processes: min(WORKERS, usable CPUs - 1)
MIN_CPUS = 2            # usable CPUs needed for one worker (the main process finishes its own variant)
MIN_EFFICIENCY = 0.65   # calibration: every process burns CALIBRATION_S at once; the lowest CPU time / wall time
CALIBRATION_S = 0.3     #   must reach this (several CFS quota periods, so a CPU quota shows)
GUARD_WINDOW = 8        # runtime guard: over the last boards, the median of the workers' CPU time / wall time
GUARD_CPU = 0.8         #   below GUARD_CPU pauses the pool (the CPUs are shared with other load; 1.0 when not)
PROBE_START = 16        # boards the pool sits out after a pause before it is tried again (a probe)
PROBE_MAX = 128         #   a pause less than PROBE_RESET boards after a probe doubles the wait, up to PROBE_MAX
PROBE_RESET = 32        #   a probe that runs this many boards without a pause resets the wait to PROBE_START
READY_TIMEOUT = 5.0     # seconds for a worker to answer at start
SEND_S = 0.25           # every request of a board must be written within this time, else the pool stops
STALE_S = 3.6           # a worker busy this long since its request is stuck (a live task ends within the 3 s
                        # time budget): the pool stops at the next dispatch
WIN = sys.platform == 'win32'  # spawned workers (Windows has no fork)
READY_TIMEOUT_SPAWN = 120.0     # one limit for creating the spawned workers and their handshake (each child
                                #   imports ARGOS, numba compiles its kernels, before it answers)
WORKER_NAME = 'ARGOS-worker'    # a spawned child importing ARGOS never starts a pool of its own
ERROR_LIMIT = 3                 # spawned workers' error replies that stop the pool
EXTRA_WORKERS = 1               # extra spawned workers (Windows) for the extra cover variants only
EXTRA_MIN_CORES = 5             # physical cores in the affinity they need (main + 3 base + 1 extra)

IN_WORKER = False
_PROXY_NOTE = None  # set by a spawned worker's loop: sends the cover proxy as soon as the cover is built
STATE = {'active': False, 'reason': 'not started', 'workers': [], 'gen': 0, 'cpus': None, 'efficiency': None,
         'guard': True, 'ratios': [], 'missing_run': 0, 'paused': False, 'wait': 0, 'probe_every': 16, 'run': 0,
         'pauses': 0, 'probes': 0, 'spawn': False, 'errors': 0, 'last_error': None, 'start_s': None,
         'extras': 0, 'extra_reason': None, 'phys_cores': None, 'core_classes': None, 'daemon_parent': None}


# ----------------------------------------------------------------------------------------------------------- CPUs
def _cgroup_quota():
    """The smallest CPU quota (in CPUs) of this process's cgroup and its ancestors, v2 and v1; None when unlimited."""
    quotas = []
    for line in open('/proc/self/cgroup'):
        hid, ctrls, rel = line.rstrip('\n').split(':', 2)
        if hid == '0' and ctrls == '':  # cgroup v2
            base = '/sys/fs/cgroup'
            parts = [p for p in rel.split('/') if p]
            for i in range(len(parts), -1, -1):
                f = os.path.join(base, *parts[:i], 'cpu.max')
                if os.path.exists(f):
                    q, p = open(f).read().split()
                    if q != 'max':
                        quotas.append(int(q) / int(p))
        elif 'cpu' in ctrls.split(','):  # cgroup v1
            for base in ('/sys/fs/cgroup/cpu,cpuacct', '/sys/fs/cgroup/cpu', '/sys/fs/cgroup/cpuacct,cpu'):
                if not os.path.isdir(base):
                    continue
                parts = [p for p in rel.split('/') if p]
                for i in range(len(parts), -1, -1):
                    d = os.path.join(base, *parts[:i])
                    fq, fp = os.path.join(d, 'cpu.cfs_quota_us'), os.path.join(d, 'cpu.cfs_period_us')
                    if os.path.exists(fq) and os.path.exists(fp):
                        q, p = int(open(fq).read()), int(open(fp).read())
                        if q > 0 and p > 0:
                            quotas.append(q / p)
                break
    return min(quotas) if quotas else None


def _win_affinity_count():
    """The CPUs in this process's affinity mask (Windows)."""
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
    k.GetProcessAffinityMask.restype = wintypes.BOOL
    proc, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not k.GetProcessAffinityMask(k.GetCurrentProcess(), ctypes.byref(proc), ctypes.byref(system)):
        raise OSError('GetProcessAffinityMask failed')
    return bin(proc.value).count('1')


def _win_physical_cores():
    """(physical cores that share a logical processor with this process's affinity, their EfficiencyClass
    values), from GetLogicalProcessorInformationEx(RelationProcessorCore); (None, None) when unknown: not 64-bit, a
    failed call, or a core outside processor group 0 (several groups are not handled)."""
    import ctypes
    import struct
    from ctypes import wintypes
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        return None, None
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
    k.GetProcessAffinityMask.restype = wintypes.BOOL
    k.GetLogicalProcessorInformationEx.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD))
    k.GetLogicalProcessorInformationEx.restype = wintypes.BOOL
    proc, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not k.GetProcessAffinityMask(k.GetCurrentProcess(), ctypes.byref(proc), ctypes.byref(system)):
        return None, None
    n = wintypes.DWORD(0)
    k.GetLogicalProcessorInformationEx(0, None, ctypes.byref(n))  # RelationProcessorCore: the size first
    if not n.value:
        return None, None
    buf = ctypes.create_string_buffer(n.value)
    if not k.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(n)):
        return None, None
    raw, off, cores, classes = buf.raw, 0, 0, []
    while off + 8 <= n.value:
        rel, size = struct.unpack_from('<II', raw, off)
        if size < 48:
            return None, None
        if rel == 0:  # PROCESSOR_RELATIONSHIP at +8: Flags, EfficiencyClass, Reserved[20], GroupCount, GroupMask[]
            eff, count = raw[off + 9], struct.unpack_from('<H', raw, off + 30)[0]
            hit = False
            for g in range(count):
                mask, group = struct.unpack_from('<QH', raw, off + 32 + 16 * g)
                if group != 0:
                    return None, None
                hit = hit or bool(mask & proc.value)
            if hit:
                cores += 1
                classes.append(eff)
        off += size
    return cores, classes


def usable_cpus():
    """CPUs this process may use: the affinity count, and on Linux also the smallest cgroup CPU quota."""
    if WIN:
        return _win_affinity_count()
    n = len(os.sched_getaffinity(0))
    q = _cgroup_quota()
    return int(math.floor(min(n, q))) if q is not None else n


# --------------------------------------------------------------------------------------------------------- worker
def _die_with_parent():
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def _burn(seconds):
    """Busy for `seconds` of wall time; returns CPU time / wall time of this process meanwhile."""
    t, c = time.perf_counter(), time.process_time()
    s = 0
    while time.perf_counter() - t < seconds:
        for i in range(2000):
            s += i * i
    return (time.process_time() - c) / (time.perf_counter() - t)


def _solve_variant(P, fv, config, t0=None, c0=None):
    """Finish a cover in memory (Linux: the Problem and the FOVs arrive pickled from the main process); nothing is
    written to the dataset folder. Returns (plan, seq, exact score, wall time, CPU share) or None when the finished plan
    does not verify."""
    try:
        from . import solver as S
    except ImportError:
        import solver as S
    if t0 is None:  # a spawned worker passes its request's arrival (reading the board and building the cover count)
        t0, c0 = time.perf_counter(), time.process_time()
    # as solve() does per board: _polish reads LAST_STATS['lattice'], which plan() only ever sets
    S.LAST_STATS.clear()
    S.LAST_STATS.update(io_retries=0, io_retry_wait_s=0., io_errors=[], polished=False, lattice=False,
                        members_polished=False)
    cfg = S.get_config(config)
    fovs, seq = S.plan(P, cfg, t_start=t0, early_write=None, prebuilt=fv)
    fovs, seq = S._polish(P, cfg, fovs, seq, t0, None)
    if S.verify(P, fovs, seq):
        return None
    score = float(S.ct_sim.evaluate_sweep(P, fovs, seq))
    plan = [(float(f.x), float(f.y), [int(m) for m in f.members], int(f.step), bool(f.fixed), bool(f.exclusive))
            for f in fovs]
    wall = time.perf_counter() - t0
    return plan, [int(q) for q in seq], score, wall, (time.process_time() - c0) / max(wall, 1e-6)


def _solve_variant_path(folder, v, config):
    """Spawned worker: read the board and build cover variant v itself (build_fovs has no clock, so the cover equals
    the one the main process would build), send its capture proxy, then finish it as _solve_variant does; time budget
    and CPU share count from the request's arrival."""
    t0, c0 = time.perf_counter(), time.process_time()
    import pandas as pd
    try:
        from . import solver as S
    except ImportError:
        import solver as S
    P = S.Problem(pd.read_csv(os.path.join(folder, 'input_component.csv')),
                  pd.read_csv(os.path.join(folder, 'input_size.csv')),
                  pd.read_csv(os.path.join(folder, 'input_parameter.csv')))
    fv = S.build_fovs(P, S.get_config(config), **S.variant_def(v))  # the extra variants too
    note = _PROXY_NOTE
    if note is not None:  # the cover's proxy goes to the main now, long before the finished plan
        try:
            note(int(v), float(S._cap_proxy(P, fv)))
        except Exception:
            pass
    return _solve_variant(P, fv, config, t0, c0)


def _worker_main_spawn(conn):
    """The loop of a spawned worker (Windows). It imported ARGOS on the way here (start() skipped itself because of
    the process name); it ends when the pipe breaks or the parent is gone."""
    global IN_WORKER, _PROXY_NOTE
    IN_WORKER = True
    parent = mp.parent_process()
    while True:
        try:
            if not conn.poll(1.0):
                if parent is not None and not parent.is_alive():
                    return
                continue
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        kind, gen = msg[0], msg[1]
        try:
            if kind == 'ping':
                out = ('pong', gen)
            elif kind == 'burn':
                out = ('burn', gen, _burn(msg[2]))
            else:  # ('solve_path', gen, v, folder, config)
                _PROXY_NOTE = lambda vv, p, g=gen: conn.send(('proxy', g, vv, p))  # the cover proxy, sent first
                try:
                    out = ('solve', gen, msg[2], _solve_variant_path(msg[3], msg[2], msg[4]))
                finally:
                    _PROXY_NOTE = None
        except Exception as e:  # reported, the worker stays alive
            out = ('error', gen, repr(e)[:200])
        try:
            conn.send(out)
        except (EOFError, OSError):
            return


def _worker_main(conn, foreign):
    """The loop of a forked worker (Linux): closes the pipe ends it does not own, dies with its parent
    (PR_SET_PDEATHSIG, and a parent-pid check)."""
    global IN_WORKER
    IN_WORKER = True
    for c in foreign:  # pipe ends of the parent and of the other workers
        try:
            c.close()
        except Exception:
            pass
    _die_with_parent()
    ppid = os.getppid()
    while True:
        try:
            if not conn.poll(1.0):
                if os.getppid() != ppid:
                    return
                continue
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        kind, gen = msg[0], msg[1]
        try:
            if kind == 'ping':
                out = ('pong', gen)
            elif kind == 'burn':
                out = ('burn', gen, _burn(msg[2]))
            else:  # ('solve', gen, v, P, fv, config)
                out = ('solve', gen, msg[2], _solve_variant(msg[3], msg[4], msg[5]))
        except Exception as e:  # reported, the worker stays alive
            out = ('error', gen, repr(e)[:200])
        try:
            conn.send(out)
        except (EOFError, OSError):
            return


# --------------------------------------------------------------------------------------------------------- parent
def _stop(reason):
    """Stop the pool for good: forked workers get a stop message (never waiting on a blocked pipe), spawned workers
    are terminated; every process is joined briefly and killed if still alive."""
    for w in STATE['workers']:
        try:
            if not STATE['spawn']:  # spawned workers get no cooperative message
                _send(w['conn'], None, time.perf_counter() + 0.05)  # never wait on a blocked pipe
        except Exception:
            pass
    for w in STATE['workers']:
        try:
            if STATE['spawn']:
                w['proc'].kill()  # TerminateProcess
            w['proc'].join(timeout=0.2)
            if w['proc'].is_alive():
                w['proc'].kill()
                w['proc'].join(timeout=0.2)
        except Exception:
            pass
        try:
            w['conn'].close()
        except Exception:
            pass
    STATE.update(active=False, paused=False, reason=reason, workers=[], spawn=False)


def stop(reason='stopped'):
    _stop(reason)


def _read_exact(fd, n, deadline):
    import select
    buf = bytearray()
    while len(buf) < n:
        left = deadline - time.perf_counter()
        if left <= 0 or not select.select([fd], [], [], left)[0]:
            raise TimeoutError('reply not complete before the deadline')
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            raise EOFError('worker pipe closed')
        buf += chunk
    return bytes(buf)


def _send(conn, obj, deadline):
    """One message with Connection.send's framing ('!i' length, or -1 then '!Q', then the ForkingPickler bytes),
    written with non-blocking writes before an absolute deadline, so a worker that does not read cannot block the main
    process (Connection.send waits without a limit once the pipe buffer is full)."""
    import select
    import struct
    from multiprocessing.reduction import ForkingPickler
    buf = bytes(ForkingPickler.dumps(obj))
    head = struct.pack('!i', len(buf)) if len(buf) <= 0x7fffffff else struct.pack('!i', -1) + struct.pack('!Q', len(buf))
    data = memoryview(head + buf)
    fd = conn.fileno()
    blocking = os.get_blocking(fd)
    os.set_blocking(fd, False)
    try:
        while len(data):
            left = deadline - time.perf_counter()
            if left <= 0 or not select.select([], [fd], [], left)[1]:
                raise TimeoutError('request not written before the deadline')
            try:
                data = data[os.write(fd, data):]
            except BlockingIOError:
                pass
    finally:
        os.set_blocking(fd, blocking)


def _recv(conn, deadline):
    """One multiprocessing message (Connection.send framing: '!i' length, or -1 then '!Q'), read with an absolute
    deadline (Connection.recv() waits for the whole message without a limit)."""
    import pickle
    import struct
    fd = conn.fileno()
    size = struct.unpack('!i', _read_exact(fd, 4, deadline))[0]
    if size == -1:
        size = struct.unpack('!Q', _read_exact(fd, 8, deadline))[0]
    return pickle.loads(_read_exact(fd, size, deadline))


def _recv_conn(conn, deadline):
    """Spawned workers: one whole message of a Windows pipe, or TimeoutError when none arrives before the deadline."""
    if not conn.poll(max(0.0, deadline - time.perf_counter())):
        raise TimeoutError('no reply before the deadline')
    return conn.recv()


def _recv_msg(w, deadline):
    """A reply of worker w: whole Windows pipe messages (spawn; wait()/poll() said one is there), or the fork pool's
    partial-message reader."""
    if STATE['spawn']:
        return w['conn'].recv()
    return _recv_part(w, deadline)


def _note_error(msg):
    """Spawned workers: an error reply is counted and kept; ERROR_LIMIT of them stop the pool."""
    if STATE['spawn'] and msg and msg[0] == 'error':
        STATE['errors'] += 1
        STATE['last_error'] = str(msg[2])[:200]


@contextlib.contextmanager
def _hide_main():
    """While spawned workers start, the parent's __main__ has no file and no spec, so multiprocessing does not run the
    harness script again in them (it would without an if __name__ == '__main__' guard)."""
    main = sys.modules.get('__main__')
    saved = {k: main.__dict__[k] for k in ('__file__', '__spec__') if main is not None and k in main.__dict__}
    try:
        if main is not None:
            main.__dict__.pop('__file__', None)
            main.__dict__['__spec__'] = None
        yield
    finally:
        if main is not None:
            main.__dict__.pop('__spec__', None)
            main.__dict__.update(saved)


@contextlib.contextmanager
def _allow_children():
    """multiprocessing refuses to start a child from a daemon process (a multiprocessing.Pool worker) by its daemon
    flag only; the flag is lowered while the spawned workers start and restored at once. The workers stay
    daemons: a daemon parent that exits normally terminates them (multiprocessing's exit function), one that is killed
    leaves them to end by themselves (the parent check, a broken pipe)."""
    me = mp.current_process()
    was = bool(me.daemon)
    try:
        if was:
            me.daemon = False
        yield
    finally:
        if was:
            me.daemon = True


def _start_spawn(workers, min_cpus, setcover):
    """start() on Windows (also in a daemon process: _allow_children)."""
    STATE['daemon_parent'] = bool(mp.current_process().daemon)
    if 'spawn' not in mp.get_all_start_methods():
        STATE.update(active=False, reason='no spawn start method')
        return False
    if not setcover.ENABLED:
        STATE.update(active=False, reason='set-cover kernels unavailable')
        return False
    STATE['cpus'] = usable_cpus()
    if workers is None:
        workers = min(WORKERS, STATE['cpus'] - 1)
    if workers < 1 or STATE['cpus'] < min_cpus:
        STATE.update(active=False, reason=f'usable CPUs {STATE["cpus"]} < {min_cpus}')
        return False
    ctx = mp.get_context('spawn')
    mine = []
    t_start = time.perf_counter()
    dl = t_start + READY_TIMEOUT_SPAWN  # the limit covers creating the processes too
    STATE.update(workers=mine, spawn=True, errors=0, last_error=None)
    try:
        with _hide_main(), _allow_children():  # a daemon parent too
            for k in range(workers):
                if time.perf_counter() > dl:
                    raise TimeoutError('workers not created before the start limit')
                a, b = ctx.Pipe()
                p = ctx.Process(target=_worker_main_spawn, args=(b,), daemon=True, name=f'{WORKER_NAME}-{k}')
                p.start()
                b.close()
                mine.append({'conn': a, 'proc': p, 'busy': False})
        for w in mine:
            w['conn'].send(('ping', 0))
        for w in mine:
            if _recv_conn(w['conn'], dl) != ('pong', 0):
                raise RuntimeError('bad handshake')
        for w in mine:
            w['conn'].send(('burn', 0, CALIBRATION_S))
        dl2 = time.perf_counter() + READY_TIMEOUT
        effs = [_burn(CALIBRATION_S)] + [_recv_conn(w['conn'], dl2)[2] for w in mine]
    finally:
        STATE['start_s'] = time.perf_counter() - t_start
    STATE['efficiency'] = min(effs)
    if STATE['efficiency'] < MIN_EFFICIENCY:  # paused: tried again after PROBE_START boards
        _pause(f'calibration efficiency {STATE["efficiency"]:.2f} < {MIN_EFFICIENCY}')
        return False
    STATE.update(active=True, reason='active')
    _start_extras(ctx, workers)  # only after the base pool is up and calibrated
    return True


def _start_extras(ctx, base):
    """Start EXTRA_WORKERS extra spawned workers after `base` base workers, when the affinity holds at least
    EXTRA_MIN_CORES physical cores and a usable CPU per process. Their own handshake and a calibration of the main
    process with the extras only (the base workers get no message here). Any failure retires the extras only."""
    try:
        cores, classes = _win_physical_cores()
    except Exception:
        cores, classes = None, None
    STATE.update(phys_cores=cores, core_classes=classes)
    n = EXTRA_WORKERS
    if n < 1 or cores is None or cores < EXTRA_MIN_CORES or STATE['cpus'] - 1 < base + n:
        STATE['extra_reason'] = f'base workers only (physical cores {cores}, usable CPUs {STATE["cpus"]})'
        return
    mine = STATE['workers']
    t0 = time.perf_counter()
    dl = t0 + READY_TIMEOUT_SPAWN
    try:
        with _hide_main(), _allow_children():  # a daemon parent too
            for k in range(base, base + n):
                if time.perf_counter() > dl:
                    raise TimeoutError('extra workers not created before the start limit')
                a, b = ctx.Pipe()
                p = ctx.Process(target=_worker_main_spawn, args=(b,), daemon=True, name=f'{WORKER_NAME}-{k}')
                p.start()
                b.close()
                mine.append({'conn': a, 'proc': p, 'busy': False, 'extra': True})
        extras = [w for w in mine if w.get('extra')]
        for w in extras:
            w['conn'].send(('ping', 0))
        for w in extras:
            if _recv_conn(w['conn'], dl) != ('pong', 0):
                raise RuntimeError('bad handshake of an extra worker')
        for w in extras:
            w['conn'].send(('burn', 0, CALIBRATION_S))
        dl2 = time.perf_counter() + READY_TIMEOUT
        effs = [_burn(CALIBRATION_S)] + [_recv_conn(w['conn'], dl2)[2] for w in extras]
        if min(effs) < MIN_EFFICIENCY:
            raise RuntimeError(f'extra calibration efficiency {min(effs):.2f} < {MIN_EFFICIENCY}')
        STATE.update(extras=n, extra_reason=f'active ({n} extra, {time.perf_counter() - t0:.1f} s)')
    except Exception as e:
        _retire_extras('extra start failed: ' + repr(e)[:120])


def _retire_extras(reason):
    """End the extra workers only (killed; kept in the list, retired, so no index moves). The base workers and the
    pool go on."""
    for w in STATE['workers']:
        if w.get('extra') and not w.get('retired'):
            try:
                w['proc'].kill()
                w['proc'].join(timeout=0.2)
            except Exception:
                pass
            try:
                w['conn'].close()
            except Exception:
                pass
            w.update(retired=True, busy=False)
    STATE.update(extras=0, extra_reason='retired: ' + reason)


def _recv_part(w, deadline):
    """Forked workers: read one framed message of worker w as far as it arrives before the absolute deadline. A
    partial message stays in w['rbuf'] and the next call goes on from there, so a reply that is still being written
    when the wait ends is simply late (never a broken pipe). Returns the message, or None when it is not complete yet."""
    import pickle
    import select
    import struct
    buf = w.setdefault('rbuf', bytearray())
    fd = w['conn'].fileno()
    while True:
        head = 4
        size = struct.unpack('!i', bytes(buf[:4]))[0] if len(buf) >= 4 else None
        if size == -1:
            head = 12
            size = struct.unpack('!Q', bytes(buf[4:12]))[0] if len(buf) >= 12 else None
        need = head if size is None else head + size
        if size is not None and len(buf) >= need:
            msg = pickle.loads(bytes(buf[head:need]))
            w['rbuf'] = bytearray()
            return msg
        left = deadline - time.perf_counter()
        if left <= 0 or not select.select([fd], [], [], left)[0]:
            return None
        chunk = os.read(fd, need - len(buf))
        if not chunk:
            raise EOFError('worker pipe closed')
        buf += chunk


def start(workers=None, min_cpus=None, guard=True):
    """Starts the pool when every condition holds; returns STATE['active']. workers=None: min(WORKERS, CPUs - 1)."""
    if STATE['workers']:
        _stop('restart')
    min_cpus = MIN_CPUS if min_cpus is None else min_cpus
    STATE.update(guard=bool(guard), ratios=[], missing_run=0, paused=False, wait=0, probe_every=PROBE_START, run=0,
                 pauses=0, probes=0, extras=0, extra_reason=None, phys_cores=None, core_classes=None)
    try:
        try:
            from . import setcover
        except ImportError:
            import setcover
        if IN_WORKER or mp.current_process().name.startswith(WORKER_NAME):  # a spawned child importing ARGOS
            STATE.update(active=False, reason='inside a worker')
            return False
        if WIN:
            return _start_spawn(workers, min_cpus, setcover)
        if not sys.platform.startswith('linux') or 'fork' not in mp.get_all_start_methods():
            STATE.update(active=False, reason='no fork (not Linux)')
            return False
        if mp.current_process().daemon or threading.active_count() != 1 or IN_WORKER:
            STATE.update(active=False, reason='daemon process, other threads, or inside a worker')
            return False
        if not setcover.ENABLED:  # the plain-Python kernels (setcover.PY) are allowed too
            STATE.update(active=False, reason='set-cover kernels unavailable')
            return False
        STATE['cpus'] = usable_cpus()
        if workers is None:
            workers = min(WORKERS, STATE['cpus'] - 1)
        if workers < 1 or STATE['cpus'] < min_cpus:
            STATE.update(active=False, reason=f'usable CPUs {STATE["cpus"]} < {min_cpus}')
            return False
        ctx = mp.get_context('fork')
        mine = []
        for _ in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker_main, args=(b, [x['conn'] for x in mine] + [a]), daemon=True)
            p.start()
            b.close()
            mine.append({'conn': a, 'proc': p, 'busy': False})
        STATE['workers'] = mine
        dl = time.perf_counter() + READY_TIMEOUT
        for w in mine:
            w['conn'].send(('ping', 0))
        for w in mine:
            if _recv(w['conn'], dl) != ('pong', 0):
                raise RuntimeError('bad handshake')
        # calibration: every worker and the main process burn CPU at once; each one's CPU time / wall time
        for w in mine:
            w['conn'].send(('burn', 0, CALIBRATION_S))
        effs = [_burn(CALIBRATION_S)] + [_recv(w['conn'], dl)[2] for w in mine]
        STATE['efficiency'] = min(effs)
        if STATE['efficiency'] < MIN_EFFICIENCY:  # paused: tried again after PROBE_START boards
            _pause(f'calibration efficiency {STATE["efficiency"]:.2f} < {MIN_EFFICIENCY}')
            return False
        STATE.update(active=True, reason='active')
        return True
    except Exception as e:
        _stop('start failed: ' + repr(e)[:120])
        return False


def dispatch(P, covers, config, folder=None, extras=()):
    """Sends covers [(variant index, FOVs)] to idle base workers (one each), in the given order: forked workers get
    the Problem and the FOVs, spawned workers only (variant, absolute dataset folder) and build the cover themselves.
    `extras` [(variant, config override)] go to idle extra workers. Returns (generation, [(variant, worker index)]);
    (None, []) when the pool is not active."""
    if not STATE['active'] or IN_WORKER:
        return None, []
    try:
        STATE['gen'] += 1
        gen, sent = STATE['gen'], []
        now = time.perf_counter()
        if any(w['busy'] and now - w.get('t_sent', now) > STALE_S for w in STATE['workers'] if not w.get('extra')):
            _stop(f'a worker has been busy for more than {STALE_S} s')  # a stuck base worker stops the pool
            return None, []
        if any(w['busy'] and now - w.get('t_sent', now) > STALE_S for w in STATE['workers']
               if w.get('extra') and not w.get('retired')):  # a stale extra worker retires the extras only
            _retire_extras(f'an extra worker has been busy for more than {STALE_S} s')
        idle = [i for i, w in enumerate(STATE['workers']) if not w['busy'] and not w.get('extra')]
        idle_x = [i for i, w in enumerate(STATE['workers'])
                  if not w['busy'] and w.get('extra') and not w.get('retired')]
        deadline = now + SEND_S
        for (v, fv), i in zip(covers, idle):
            if STATE['spawn']:
                if folder is None:
                    raise RuntimeError('spawned workers need the dataset folder')
                # resolved here, so a later change of the working folder cannot mislead the worker
                STATE['workers'][i]['conn'].send(('solve_path', gen, int(v), os.path.abspath(str(folder)), config))
            else:
                _send(STATE['workers'][i]['conn'], ('solve', gen, int(v), P, fv, config), deadline)
            STATE['workers'][i]['busy'] = True
            STATE['workers'][i]['gen'] = gen
            STATE['workers'][i]['t_sent'] = time.perf_counter()
            sent.append((int(v), i))
        for (v, over), i in zip(extras if STATE['spawn'] else (), idle_x):  # extra members, extra workers only
            try:  # a failed send to an extra worker retires the extras only
                STATE['workers'][i]['conn'].send(('solve_path', gen, int(v), os.path.abspath(str(folder)),
                                                  dict(config or {}, **over) if over else config))
            except Exception as e:
                _retire_extras('extra send failed: ' + repr(e)[:80])
                break
            STATE['workers'][i]['busy'] = True
            STATE['workers'][i]['gen'] = gen
            STATE['workers'][i]['t_sent'] = time.perf_counter()
            sent.append((int(v), i))
        return gen, sent
    except Exception as e:
        _stop('dispatch failed: ' + repr(e)[:120])
        return None, []


def collect(gen, sent, deadline, proxies=None, extend=None):
    """Replies of this generation received before the absolute deadline:
    [(variant, plan, seq, score, rt, arrival, cpu share)].
    Replies of older generations are dropped (their worker becomes idle again); a dead worker stops the pool.
    Cover proxies ('proxy' messages) of this generation go into `proxies` (the worker stays busy); when the deadline
    passes with members still pending, extend(pending variants, proxies) may return a later deadline, once (an
    exception in extend keeps the first deadline)."""
    from multiprocessing.connection import wait
    got, pending = [], {i for _, i in sent}
    try:
        while pending:
            conns = {STATE['workers'][i]['conn']: i for i in pending}
            left = deadline - time.perf_counter()
            ready = wait(list(conns), timeout=left) if left > 0 else []
            if not ready:
                try:  # a failing extension keeps the first deadline and never stops the pool
                    new = extend({v for v, i in sent if i in pending}, proxies) if extend is not None else None
                except Exception:
                    new = None
                extend = None  # once
                if new is not None and new > time.perf_counter():
                    deadline = new
                    continue
                break
            for c in ready:
                i = conns[c]
                if STATE['workers'][i].get('retired'):  # retired during this collect
                    pending.discard(i)
                    continue
                if STATE['workers'][i].get('extra'):  # a failed receive retires the extras only
                    try:
                        msg = _recv_msg(STATE['workers'][i], deadline)
                    except Exception as e:
                        _retire_extras('extra receive failed: ' + repr(e)[:80])
                        pending.difference_update([j for j in pending if STATE['workers'][j].get('retired')])
                        continue
                else:
                    msg = _recv_msg(STATE['workers'][i], deadline)  # a partial reply is kept, not fatal
                if msg is None:
                    continue
                if msg[0] == 'proxy':  # a cover proxy on the way; the worker is still busy with the request
                    if msg[1] == gen and proxies is not None:
                        proxies[int(msg[2])] = float(msg[3])
                    continue
                STATE['workers'][i]['busy'] = False
                if STATE['workers'][i].get('extra') and msg[0] == 'error':  # retires the extras only
                    _retire_extras('extra worker error: ' + str(msg[2])[:120])
                    pending.difference_update([j for j in pending if STATE['workers'][j].get('retired')])
                    continue
                _note_error(msg)
                if msg[1] != gen:
                    _note_late(msg)  # a late reply to an older board (its CPU share still counts)
                    continue
                pending.discard(i)
                if STATE['spawn'] and time.perf_counter() > deadline:  # received after the wait cap: late
                    _note_late(msg)
                    continue
                if msg[0] == 'solve' and msg[3] is not None:
                    plan, seq, score, rt, cpu = msg[3]
                    got.append((msg[2], plan, seq, score, rt, time.perf_counter(), cpu))
        # workers still pending keep busy=True; their late replies are dropped at a later collect
        for i in list(pending):
            if STATE['workers'][i].get('extra'):  # a dead extra worker retires the extras only
                if not STATE['workers'][i].get('retired') and not STATE['workers'][i]['proc'].is_alive():
                    _retire_extras('an extra worker died')
            elif not STATE['workers'][i]['proc'].is_alive():
                raise RuntimeError('worker died')
        if STATE['spawn'] and STATE['errors'] >= ERROR_LIMIT:
            _stop('worker errors: ' + str(STATE['last_error']))
        return got
    except Exception as e:
        _stop('collect failed: ' + repr(e)[:120])
        return got


def drain_late():
    """Before a new board: take any late replies already waiting, so their workers become idle again (also while
    the pool is paused)."""
    if not (STATE['active'] or STATE['paused']):
        return
    try:
        for w in STATE['workers']:
            if w.get('extra'):
                continue  # the extra workers in _drain_extras(), on their own
            while w['busy'] and w['conn'].poll(0):
                msg = _recv_msg(w, time.perf_counter() + 0.05)  # goes on with a kept partial reply
                if msg is None:
                    break
                if msg[0] == 'proxy':  # not a reply; the worker is still busy
                    continue
                _note_late(msg)
                _note_error(msg)
                w['busy'] = False
        if STATE['spawn'] and STATE['errors'] >= ERROR_LIMIT:
            _stop('worker errors: ' + str(STATE['last_error']))
    except Exception as e:
        _stop('drain failed: ' + repr(e)[:120])
    _drain_extras()


def _drain_extras():
    """Late replies of the live extra workers; an error reply, or a failed poll or receive (the pipe of an extra worker
    that died while computing a request), retires the extras only."""
    if not (STATE['active'] or STATE['paused']):
        return
    for w in STATE['workers']:
        if not w.get('extra') or w.get('retired'):
            continue
        try:
            while w['busy'] and w['conn'].poll(0):
                msg = _recv_msg(w, time.perf_counter() + 0.05)
                if msg is None:
                    break
                if msg[0] == 'proxy':  # not a reply; the worker is still busy
                    continue
                _note_late(msg)
                w['busy'] = False
                if msg[0] == 'error':
                    _retire_extras('extra worker error: ' + str(msg[2])[:120])
                    return
        except Exception as e:
            _retire_extras('extra drain failed: ' + repr(e)[:80])
            return


def _note_late(msg):
    """A late reply still tells how much CPU its worker got. Without this, workers that always finish after the wait
    cap under outside load would never feed the guard, and the pool would keep paying the wait."""
    if STATE['guard'] and msg and msg[0] == 'solve' and msg[3] is not None:
        STATE['ratios'] = (STATE['ratios'] + [float(msg[3][4])])[-GUARD_WINDOW:]


def tick():
    """Once at the start of every solve(). Drains late replies; stops the pool for good when a worker died or
    has been busy longer than STALE_S (also while paused); counts the boards of a paused pool and, after
    probe_every of them, uses the pool again with an empty guard window (a probe); counts the boards of a probe."""
    if IN_WORKER or not (STATE['active'] or STATE['paused']):
        return
    drain_late()
    if not (STATE['active'] or STATE['paused']):
        return
    try:
        if not all(w['proc'].is_alive() for w in STATE['workers'] if not w.get('extra')):
            _stop('a worker died')
            return
        if any(not w['proc'].is_alive() for w in STATE['workers'] if w.get('extra') and not w.get('retired')):
            _retire_extras('an extra worker died')
        now = time.perf_counter()
        if any(w['busy'] and now - w.get('t_sent', now) > STALE_S for w in STATE['workers'] if not w.get('extra')):
            _stop(f'a worker has been busy for more than {STALE_S} s')
            return
        if any(w['busy'] and now - w.get('t_sent', now) > STALE_S for w in STATE['workers']
               if w.get('extra') and not w.get('retired')):
            _retire_extras(f'an extra worker has been busy for more than {STALE_S} s')
    except Exception as e:
        _stop('tick failed: ' + repr(e)[:120])
        return
    if STATE['paused']:
        STATE['wait'] += 1
        if STATE['wait'] >= STATE['probe_every']:
            STATE.update(active=True, paused=False, ratios=[], run=0, probes=STATE['probes'] + 1,
                         reason=f'active (probe {STATE["probes"] + 1} after {STATE["wait"]} boards)')
    else:
        STATE['run'] += 1
        if STATE['probes'] and STATE['run'] == PROBE_RESET:
            STATE['probe_every'] = PROBE_START  # the probe held: the next pause waits the short time again


def _pause(reason):
    """The CPU guard pauses the pool instead of stopping it. The workers stay alive and idle (tick() drains
    late replies and checks them every board); after probe_every boards the pool is tried again. A pause less than
    PROBE_RESET boards after a probe doubles the wait, up to PROBE_MAX."""
    if STATE['probes'] and STATE['run'] < PROBE_RESET:
        STATE['probe_every'] = min(PROBE_MAX, STATE['probe_every'] * 2)
    STATE.update(active=False, paused=True, wait=0, pauses=STATE['pauses'] + 1, reason='paused: ' + reason)


def note_board(cpu, missing):
    """Runtime guard, after each board with a portfolio: pauses the pool when the workers keep getting too little CPU
    (median CPU time / wall time over the last GUARD_WINDOW samples below GUARD_CPU; one sample per board from the
    on-time replies, one per late reply). `missing` is not used: with the wait cap a late reply is normal (dead or
    stuck workers are caught elsewhere)."""
    if not STATE['active'] or not STATE['guard']:
        return
    if cpu is not None:
        STATE['ratios'] = (STATE['ratios'] + [float(cpu)])[-GUARD_WINDOW:]
    r = sorted(STATE['ratios'])
    if len(r) >= GUARD_WINDOW // 2 and r[len(r) // 2] < GUARD_CPU:
        _pause(f'guard: median worker CPU share {r[len(r) // 2]:.2f} < {GUARD_CPU}')
