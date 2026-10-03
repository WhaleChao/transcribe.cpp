"""Model-wide compute lock and stream lease: deterministic tests that need NO
model.

The native library allows one compute call in flight per model across all of
its sessions, and an active stream occupies the model from begin until it
ends. The binding enforces that with ``Model._exclusive(kind)`` and the
model-wide stream lease (``Model._stream_lease``). These
tests build Model/Session objects around fake handles and replace the native
entry points the high-level code calls (``transcribe_cpp._lib.<fn>``, looked
up at call time) with Python probes, so overlap, waiting, deferred frees and
re-entrancy are observed directly — no GGUF, no native compute.

Every native call that could touch a fake handle is patched, and the fixture
closes every fake (and collects garbage) BEFORE monkeypatch restores the real
functions, so a fake handle never reaches the real library.
"""

from __future__ import annotations

import ctypes
import gc
import itertools
import threading
import time
import weakref
from types import SimpleNamespace

import pytest

import transcribe_cpp as t
from transcribe_cpp import _generated

TIMEOUT = 10.0  # generous upper bound for "must not deadlock" joins


def _in_thread(fn, *args):
    """Run fn in a daemon thread; returns (thread, box) where box collects the
    return value or the exception."""
    box: dict = {}

    def target():
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # surfaced by the test, never swallowed
            box["error"] = exc

    th = threading.Thread(target=target, daemon=True)
    th.start()
    return th, box


def _join(th, what):
    th.join(TIMEOUT)
    assert not th.is_alive(), f"{what} did not finish (deadlock?)"


@pytest.fixture
def fake(monkeypatch):
    ids = itertools.count(0x1000, 0x10)
    frees: list = []  # ("session" | "model", handle) in native-free order
    made: list = []

    monkeypatch.setattr(t._lib, "transcribe_session_free",
                        lambda h: frees.append(("session", h.value)))
    monkeypatch.setattr(t._lib, "transcribe_model_free",
                        lambda h: frees.append(("model", h.value)))
    resets: list = []  # session handles transcribe_stream_reset was given
    monkeypatch.setattr(t._lib, "transcribe_stream_reset",
                        lambda h: resets.append(h.value))
    state = {"value": _generated.TRANSCRIBE_STREAM_ACTIVE}
    monkeypatch.setattr(t._lib, "transcribe_stream_get_state",
                        lambda h: state["value"])
    # The copy-out reads many native accessors; stub it to report which
    # handle it was given (proves the in-flight call kept its handle).
    monkeypatch.setattr(t.Session, "_materialize",
                        lambda self, h=None, utt=None:
                        ("result", (h or self._h).value, utt))

    def model():
        m = t.Model.__new__(t.Model)
        m._sessions = weakref.WeakSet()
        m._init_compute_state()  # the same init Model.__init__ runs
        m._handle = ctypes.c_void_p(next(ids))
        made.append(m)
        return m

    def session(m, track=True):
        s = t.Session.__new__(t.Session)
        s._model = m
        s._handle = ctypes.c_void_p(next(ids))
        s._cancel = threading.Event()
        s._abort_trampoline = None
        m._sessions.add(s)
        if track:
            made.append(s)
        return s

    yield SimpleNamespace(model=model, session=session, frees=frees,
                          resets=resets, state=state)

    for obj in reversed(made):
        obj.close()
    made.clear()
    gc.collect()


PCM = [0.0] * 160


class _LockSpy:
    """Wraps a model's compute lock and signals when a thread starts a
    BLOCKING acquire (i.e. is parked waiting for it). The non-blocking
    acquires of the deferred-free path do not count. Install it while the
    real lock is held: release() is looked up on the attribute at call
    time, so the holder releases the same underlying lock."""

    def __init__(self, model):
        self._lock = model._compute_lock
        self.waiting = threading.Event()
        model._compute_lock = self

    def acquire(self, blocking=True, timeout=-1):
        if blocking:
            self.waiting.set()
        return self._lock.acquire(blocking, timeout)

    def release(self):
        self._lock.release()

    def locked(self):
        return self._lock.locked()


class _Gate:
    """A native probe that parks inside the 'native call' until released."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, *args):
        self.entered.set()
        assert self.release.wait(TIMEOUT), "gate never released"
        return 0


# --- serialization ------------------------------------------------------------


def test_calls_on_two_sessions_never_overlap(fake, monkeypatch):
    m = fake.model()
    s1, s2, s3 = fake.session(m), fake.session(m), fake.session(m)
    stream = t.Stream(s3)

    guard = threading.Lock()
    state = {"active": 0, "max": 0, "calls": 0}

    def probe(*args):
        with guard:
            state["active"] += 1
            state["calls"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep(0.02)  # widen the window an unlocked overlap would hit
        with guard:
            state["active"] -= 1
        return 0

    monkeypatch.setattr(t._lib, "transcribe_run", probe)
    monkeypatch.setattr(t._lib, "transcribe_stream_feed", probe)

    start = threading.Barrier(3)
    n = 5

    def runs(session):
        start.wait(TIMEOUT)
        return [session.run(PCM) for _ in range(n)]

    def feeds():
        start.wait(TIMEOUT)
        return [stream.feed(PCM) for _ in range(n)]

    workers = [_in_thread(runs, s1), _in_thread(runs, s2), _in_thread(feeds)]
    for th, _ in workers:
        _join(th, "worker")
    for _, box in workers:
        assert "error" not in box, box["error"]

    assert state["calls"] == 3 * n
    assert state["max"] == 1, "compute calls on one model overlapped"
    assert workers[0][1]["value"] == [("result", s1._handle.value, None)] * n
    assert workers[1][1]["value"] == [("result", s2._handle.value, None)] * n


def test_contended_call_waits_instead_of_raising(fake, monkeypatch):
    m = fake.model()
    s1, s2 = fake.session(m), fake.session(m)
    gate = _Gate()
    second_entered = threading.Event()

    def probe(h, *rest):
        if h.value == s1._handle.value:
            return gate(h)
        second_entered.set()
        return 0

    monkeypatch.setattr(t._lib, "transcribe_run", probe)
    try:
        a, a_box = _in_thread(s1.run, PCM)
        assert gate.entered.wait(TIMEOUT)
        b, b_box = _in_thread(s2.run, PCM)
        # B must be parked on the lock: not in native, not failed, not done.
        assert not second_entered.wait(0.3), "second call overlapped the first"
        assert b.is_alive() and not b_box, f"contended call did not wait: {b_box}"
    finally:
        gate.release.set()
    _join(a, "first run")
    _join(b, "second run")
    assert "error" not in a_box and "error" not in b_box, (a_box, b_box)
    assert b_box["value"] == ("result", s2._handle.value, None)


def test_lock_is_per_model(fake, monkeypatch):
    s1 = fake.session(fake.model())
    s2 = fake.session(fake.model())
    gate = _Gate()

    def probe(h, *rest):
        return gate(h) if h.value == s1._handle.value else 0

    monkeypatch.setattr(t._lib, "transcribe_run", probe)
    try:
        a, _ = _in_thread(s1.run, PCM)
        assert gate.entered.wait(TIMEOUT)
        b, b_box = _in_thread(s2.run, PCM)
        _join(b, "run on another model")  # must not wait behind model 1
        assert "error" not in b_box, b_box
    finally:
        gate.release.set()
    _join(a, "first run")


def test_every_compute_site_holds_the_lock(fake, monkeypatch):
    m = fake.model()
    s = fake.session(m)
    seen: list = []

    def holds(name, ret=0):
        def probe(*args):
            seen.append((name, m._compute_lock.locked()
                         and m._compute_owner == threading.get_ident()))
            return ret
        return probe

    for name in ("transcribe_run", "transcribe_run_batch",
                 "transcribe_stream_begin", "transcribe_stream_feed",
                 "transcribe_stream_finalize", "transcribe_stream_reset",
                 "transcribe_batch_status"):
        monkeypatch.setattr(t._lib, name, holds(name))
    monkeypatch.setattr(t._lib, "transcribe_batch_n_results",
                        holds("transcribe_batch_n_results", ret=1))
    real_materialize = t.Session._materialize

    def materialize(self, h=None, utt=None):
        seen.append(("copy-out", m._compute_lock.locked()))
        return real_materialize(self, h, utt)

    monkeypatch.setattr(t.Session, "_materialize", materialize)

    s.run(PCM)
    assert s.run_batch([PCM]) == [("result", s._handle.value, 0)]
    stream = s.stream()
    stream.feed(PCM)
    stream.finalize()
    stream.reset()

    names = [n for n, _ in seen]
    for site in ("transcribe_run", "transcribe_run_batch",
                 "transcribe_batch_n_results", "transcribe_batch_status",
                 "transcribe_stream_begin", "transcribe_stream_feed",
                 "transcribe_stream_finalize", "transcribe_stream_reset",
                 "copy-out"):
        assert site in names, f"{site} never reached"
    assert all(ok for _, ok in seen), [n for n, ok in seen if not ok]
    assert not m._compute_lock.locked()


def test_lock_released_after_native_error(fake, monkeypatch):
    m = fake.model()
    s = fake.session(m)
    monkeypatch.setattr(t._lib, "transcribe_run",
                        lambda *a: _generated.TRANSCRIBE_ERR_INVALID_ARG)
    with pytest.raises(t.InvalidArgument):
        s.run(PCM)
    assert not m._compute_lock.locked()
    monkeypatch.setattr(t._lib, "transcribe_run", lambda *a: 0)
    assert s.run(PCM) == ("result", s._handle.value, None)


# --- cancellation -------------------------------------------------------------


def test_cancel_while_queued_is_kept(fake, monkeypatch):
    # cancel() is lock-free and the flag is cleared BEFORE the lock wait, so
    # a cancel issued while a call is queued is still set when it starts.
    m = fake.model()
    s1, s2 = fake.session(m), fake.session(m)
    gate = _Gate()
    flag_at_start: list = []

    def probe(h, *rest):
        if h.value == s1._handle.value:
            return gate(h)
        flag_at_start.append(s2._cancel.is_set())
        return 0

    monkeypatch.setattr(t._lib, "transcribe_run", probe)
    try:
        a, _ = _in_thread(s1.run, PCM)
        assert gate.entered.wait(TIMEOUT)
        spy = _LockSpy(m)
        b, _ = _in_thread(s2.run, PCM)
        assert spy.waiting.wait(TIMEOUT)  # B cleared its flag and is parked
        s2.cancel()  # must not block although the lock is held
    finally:
        gate.release.set()
    _join(a, "first run")
    _join(b, "queued run")
    assert flag_at_start == [True]


# --- close / GC during an in-flight call ------------------------------------------


def test_session_close_during_in_flight_call_defers_free(fake, monkeypatch):
    m = fake.model()
    s = fake.session(m)
    handle = s._handle.value
    gate = _Gate()
    monkeypatch.setattr(t._lib, "transcribe_run", gate)
    try:
        a, a_box = _in_thread(s.run, PCM)
        assert gate.entered.wait(TIMEOUT)
        closer, _ = _in_thread(s.close)
        _join(closer, "session.close() during an in-flight call")
        assert fake.frees == [], "session freed under an in-flight call"
    finally:
        gate.release.set()
    _join(a, "in-flight run")
    # The in-flight call finished its copy-out on the handle it captured...
    assert a_box.get("value") == ("result", handle, None), a_box
    # ...and the deferred free ran exactly once, after it.
    assert fake.frees == [("session", handle)]
    with pytest.raises(t.TranscribeError, match="closed"):
        s.run(PCM)


def test_model_close_during_in_flight_call_defers_frees(fake, monkeypatch):
    m = fake.model()
    s1, s2 = fake.session(m), fake.session(m)
    h1, h2, hm = s1._handle.value, s2._handle.value, m._handle.value
    gate = _Gate()

    def probe(h, *rest):
        return gate(h) if h.value == h1 else 0

    monkeypatch.setattr(t._lib, "transcribe_run", probe)
    try:
        a, a_box = _in_thread(s1.run, PCM)
        assert gate.entered.wait(TIMEOUT)
        spy = _LockSpy(m)
        queued, q_box = _in_thread(s2.run, PCM)
        assert spy.waiting.wait(TIMEOUT)  # parked on the lock before close
        closer, _ = _in_thread(m.close)
        _join(closer, "model.close() during an in-flight call")
        assert fake.frees == [], "freed under an in-flight call"
    finally:
        gate.release.set()
    _join(a, "in-flight run")
    _join(queued, "queued run")
    assert a_box.get("value") == ("result", h1, None), a_box
    # The queued call sees the closed model once it gets the lock.
    assert isinstance(q_box.get("error"), t.TranscribeError), q_box
    assert "closed" in str(q_box["error"])
    # Sessions first (either order), then the model; each exactly once.
    assert sorted(fake.frees[:2]) == sorted([("session", h1), ("session", h2)])
    assert fake.frees[2:] == [("model", hm)]


def test_gc_on_holder_thread_defers_free(fake, monkeypatch):
    # A finalizer can run on the thread that holds the lock (GC mid copy-out).
    # It must neither deadlock nor free under the in-flight call.
    m = fake.model()
    s = fake.session(m)
    doomed: list = []
    during: dict = {}

    def probe(*args):
        victim = fake.session(m, track=False)
        victim._cycle = victim  # only the cyclic GC can reclaim it
        doomed.append(victim._handle.value)
        del victim
        gc.collect()  # runs Session.__del__ -> close() on THIS thread
        during["frees"] = list(fake.frees)
        return 0

    monkeypatch.setattr(t._lib, "transcribe_run", probe)
    th, box = _in_thread(s.run, PCM)
    _join(th, "run with GC on the holder thread")
    assert "error" not in box, box
    assert doomed, "probe never ran"
    victim = ("session", doomed[0])
    assert victim not in during["frees"], "freed while the lock was held"
    assert fake.frees.count(victim) == 1


def test_close_from_holder_thread_does_not_deadlock(fake, monkeypatch):
    m = fake.model()
    s, other = fake.session(m), fake.session(m)
    hother = other._handle.value

    def probe(*args):
        other.close()  # same thread as the lock holder: must return at once
        assert fake.frees == []
        return 0

    monkeypatch.setattr(t._lib, "transcribe_run", probe)
    th, box = _in_thread(s.run, PCM)
    _join(th, "run closing another session from inside the call")
    assert "error" not in box, box
    assert fake.frees == [("session", hother)]


def test_reentrant_compute_raises_instead_of_deadlocking(fake, monkeypatch):
    # e.g. a log handler that calls run() from inside a native call.
    m = fake.model()
    s, other = fake.session(m), fake.session(m)
    inner: dict = {}

    def probe(h, *rest):
        if h.value == s._handle.value:
            try:
                other.run(PCM)
            except t.TranscribeError as exc:
                inner["error"] = exc
        return 0

    monkeypatch.setattr(t._lib, "transcribe_run", probe)
    th, box = _in_thread(s.run, PCM)
    _join(th, "re-entrant run")
    assert "error" not in box, box
    assert "re-entrant" in str(inner.get("error")), inner
    assert not m._compute_lock.locked()


# --- stream lease ---------------------------------------------------------------
#
# From a successful stream begin until that stream ends, run / run_batch /
# stream on ANY session of the model raise Busy at once (the C contract
# forbids a run between the feeds of an active stream on another session).
# The stream itself keeps feeding. These mirror the Rust/Swift/TS bindings.

RUN_BUSY = "a stream is active on this model; finish or drop it before run()"
BATCH_BUSY = ("a stream is active on this model; "
              "finish or drop it before run_batch()")
BEGIN_BUSY = "a stream is already active on this model"


@pytest.fixture
def native(fake, monkeypatch):
    """Probes for the native compute entry points. ``calls`` records
    (name, session handle) for every native compute call that ran;
    ``status`` sets what each one returns."""
    calls: list = []
    status = {"run": 0, "begin": 0, "feed": 0, "finalize": 0}

    def probe(name, key):
        def fn(h, *rest):
            calls.append((name, h.value))
            return status[key]
        return fn

    monkeypatch.setattr(t._lib, "transcribe_run", probe("run", "run"))
    monkeypatch.setattr(t._lib, "transcribe_run_batch", probe("run_batch", "run"))
    monkeypatch.setattr(t._lib, "transcribe_batch_n_results", lambda h: 1)
    monkeypatch.setattr(t._lib, "transcribe_batch_status", lambda h, i: 0)
    monkeypatch.setattr(t._lib, "transcribe_stream_begin", probe("begin", "begin"))
    monkeypatch.setattr(t._lib, "transcribe_stream_feed", probe("feed", "feed"))
    monkeypatch.setattr(t._lib, "transcribe_stream_finalize",
                        probe("finalize", "finalize"))
    return SimpleNamespace(calls=calls, status=status)


def _assert_busy(session, *, ran: list):
    """Every lease-checked call on *session* raises Busy without reaching
    native code."""
    before = list(ran)
    with pytest.raises(t.Busy) as ei:
        session.run(PCM)
    assert str(ei.value) == RUN_BUSY
    assert ei.value.status == 0 and isinstance(ei.value, t.TranscribeError)
    with pytest.raises(t.Busy) as ei:
        session.run_batch([PCM])
    assert str(ei.value) == BATCH_BUSY
    with pytest.raises(t.Busy) as ei:
        session.stream()
    assert str(ei.value) == BEGIN_BUSY
    assert ran == before, "a refused call reached native code"
    assert not session._model._compute_lock.locked()


def _assert_free(session):
    """The model is free again: a run on *session* goes through."""
    assert session.run(PCM) == ("result", session._handle.value, None)


def test_active_stream_refuses_sibling_and_same_session_calls(fake, native):
    m = fake.model()
    s1, s2 = fake.session(m), fake.session(m)
    stream = s1.stream()
    _assert_busy(s2, ran=native.calls)  # sibling session
    _assert_busy(s1, ran=native.calls)  # the stream's own session
    # The lease holder itself feeds and finalizes without a busy check.
    stream.feed(PCM)
    stream.feed(PCM)
    stream.finalize()
    assert [n for n, _ in native.calls] == ["begin", "feed", "feed", "finalize"]
    _assert_free(s2)
    _assert_free(s1)


def test_busy_is_exported():
    assert "Busy" in t.__all__
    assert t.Busy is t.errors.Busy
    assert issubclass(t.Busy, t.TranscribeError)
    assert not issubclass(t.Busy, t.InvalidArgument)


def test_lease_is_per_model(fake, native):
    s1 = fake.session(fake.model())
    other = fake.session(fake.model())
    s1.stream()
    _assert_free(other)


def test_failed_begin_takes_no_lease(fake, native):
    m = fake.model()
    s1, s2 = fake.session(m), fake.session(m)
    native.status["begin"] = _generated.TRANSCRIBE_ERR_NOT_IMPLEMENTED
    with pytest.raises(t.NotImplementedByModel):
        s1.stream()
    assert m._stream_lease.owner is None
    _assert_free(s2)


@pytest.mark.parametrize("ending", [
    "finalize", "finalize_error", "reset", "context_exit", "gc",
    "session_close",
])
def test_lease_released_when_stream_ends(fake, native, ending):
    m = fake.model()
    s1, s2 = fake.session(m), fake.session(m)
    h1 = s1._handle.value
    stream = s1.stream()
    stream.feed(PCM)
    _assert_busy(s2, ran=native.calls)

    if ending == "finalize":
        stream.finalize()
    elif ending == "finalize_error":
        native.status["finalize"] = _generated.TRANSCRIBE_ERR_BACKEND
        with pytest.raises(t.BackendError):
            stream.finalize()
    elif ending == "reset":
        stream.reset()
        assert fake.resets == [h1]
    elif ending == "context_exit":
        with stream:
            pass
        assert fake.resets == [h1]
    elif ending == "gc":
        stream._cycle = stream  # only the cyclic GC can reclaim it
        del stream
        gc.collect()
        assert fake.resets == [h1], "GC'd active stream was not reset once"
    elif ending == "session_close":
        s1.close()
        assert fake.frees == [("session", h1)]

    assert m._stream_lease.owner is None
    _assert_free(s2)
    if ending == "session_close":
        # The stream outlived its session: dropping it now must not reset
        # a freed native session.
        del stream
        gc.collect()
        assert fake.resets == []


def test_model_close_releases_lease(fake, native):
    m = fake.model()
    s1 = fake.session(m)
    stream = s1.stream()
    m.close()
    assert m._stream_lease.owner is None
    with pytest.raises(t.TranscribeError, match="closed"):
        stream.feed(PCM)
    del stream
    gc.collect()
    assert fake.resets == []


def test_rejected_feed_keeps_lease_while_stream_active(fake, native):
    # A feed rejected before the family hook (NaN/Inf -> InvalidArgument)
    # leaves the native stream ACTIVE: the lease stays with it.
    m = fake.model()
    s1, s2 = fake.session(m), fake.session(m)
    stream = s1.stream()
    native.status["feed"] = _generated.TRANSCRIBE_ERR_INVALID_ARG
    fake.state["value"] = _generated.TRANSCRIBE_STREAM_ACTIVE
    with pytest.raises(t.InvalidArgument):
        stream.feed(PCM)
    assert not m._compute_lock.locked()
    _assert_busy(s2, ran=native.calls)
    _assert_busy(s1, ran=native.calls)
    # A later valid feed and finalize then release it.
    native.status["feed"] = 0
    stream.feed(PCM)
    _assert_busy(s2, ran=native.calls)
    stream.finalize()
    _assert_free(s2)


def test_feed_failure_that_ends_stream_releases_lease(fake, native):
    # A failure inside the family hook leaves the stream FAILED, which is no
    # longer an active stream: the lease is released with the error.
    m = fake.model()
    s1, s2 = fake.session(m), fake.session(m)
    stream = s1.stream()
    native.status["feed"] = _generated.TRANSCRIBE_ERR_ABORTED
    fake.state["value"] = _generated.TRANSCRIBE_STREAM_FAILED
    with pytest.raises(t.Aborted):
        stream.feed(PCM)
    assert m._stream_lease.owner is None
    _assert_free(s2)


def test_ended_stream_never_clears_a_later_lease(fake, native):
    m = fake.model()
    sa, sb, sc = fake.session(m), fake.session(m), fake.session(m)
    a = sa.stream()
    a.finalize()
    b = sb.stream()  # B now holds the lease
    _assert_busy(sc, ran=native.calls)

    # A, already ended, fails a feed (not ACTIVE), finalizes again, resets
    # and is garbage-collected: none of that may touch B's lease.
    native.status["feed"] = _generated.TRANSCRIBE_ERR_INVALID_ARG
    fake.state["value"] = _generated.TRANSCRIBE_STREAM_FINISHED
    with pytest.raises(t.InvalidArgument):
        a.feed(PCM)
    _assert_busy(sc, ran=native.calls)
    a.finalize()
    _assert_busy(sc, ran=native.calls)
    a.reset()
    _assert_busy(sc, ran=native.calls)
    a._cycle = a
    del a
    gc.collect()
    _assert_busy(sc, ran=native.calls)
    sa.close()  # A's session closing leaves B's lease alone too
    _assert_busy(sc, ran=native.calls)

    native.status["feed"] = 0
    fake.state["value"] = _generated.TRANSCRIBE_STREAM_ACTIVE
    b.feed(PCM)
    b.finalize()
    _assert_free(sc)


@pytest.mark.parametrize("begin_ok", [True, False])
def test_busy_check_happens_after_acquiring_the_lock(fake, native, monkeypatch,
                                                     begin_ok):
    # A run that is already waiting for the lock when a stream begins must
    # see that stream's lease once it gets the lock; if the begin fails,
    # no lease was taken and the queued run proceeds.
    m = fake.model()
    s1, s2 = fake.session(m), fake.session(m)
    gate = _Gate()
    begin_status = 0 if begin_ok else _generated.TRANSCRIBE_ERR_INVALID_ARG

    def begin(h, *rest):
        gate(h)
        return begin_status

    monkeypatch.setattr(t._lib, "transcribe_stream_begin", begin)
    try:
        a, a_box = _in_thread(s1.stream)
        assert gate.entered.wait(TIMEOUT)
        spy = _LockSpy(m)
        b, b_box = _in_thread(s2.run, PCM)
        assert spy.waiting.wait(TIMEOUT)  # parked on the lock, before the lease
    finally:
        gate.release.set()
    _join(a, "stream begin")
    _join(b, "queued run")
    if begin_ok:
        assert isinstance(a_box.get("value"), t.Stream), a_box
        assert isinstance(b_box.get("error"), t.Busy), b_box
        assert native.calls == []  # the queued run never reached native
        a_box["value"].reset()
    else:
        assert isinstance(a_box.get("error"), t.InvalidArgument), a_box
        assert b_box.get("value") == ("result", s2._handle.value, None), b_box
    assert not m._compute_lock.locked()


def test_stream_gc_on_holder_thread_defers_reset(fake, native, monkeypatch):
    # GC can finalize an abandoned stream on the thread that holds the lock.
    # Its reset + lease release must neither deadlock nor run under the
    # in-flight call; it runs right after, and frees the model.
    m = fake.model()
    s, victim_session = fake.session(m), fake.session(m)
    hv = victim_session._handle.value
    during: dict = {}

    def run(h, *rest):
        # Simulate a stream on victim_session that holds the lease and is
        # dropped mid-call (a real begin here would be a re-entrant call).
        lease = t._StreamLease(victim_session._handle)
        m._stream_lease.owner = lease
        victim = t.Stream(victim_session, _lease=lease)
        victim._cycle = victim
        del victim
        gc.collect()  # runs Stream.__del__ on THIS thread, lock held
        during["resets"] = list(fake.resets)
        during["owner"] = m._stream_lease.owner
        return 0

    monkeypatch.setattr(t._lib, "transcribe_run", run)
    th, box = _in_thread(s.run, PCM)
    _join(th, "run with a stream GC'd on the holder thread")
    assert "error" not in box, box
    assert during["resets"] == [], "stream reset under the in-flight call"
    assert during["owner"] is not None, "lease released under the lock early"
    assert fake.resets == [hv]
    assert m._stream_lease.owner is None
    monkeypatch.setattr(t._lib, "transcribe_run", lambda *a: 0)
    _assert_free(s)


def test_session_close_releases_lease_after_free(fake, native, monkeypatch):
    # Closing the stream's session while another call holds the lock defers
    # the free; the lease is released only after it, so the next call never
    # starts while the stream's native session still exists.
    m = fake.model()
    s1, s2 = fake.session(m), fake.session(m)
    h1 = s1._handle.value
    stream = s1.stream()
    gate = _Gate()
    seen: dict = {}

    def run(h, *rest):
        seen["frees"] = list(fake.frees)
        return 0

    monkeypatch.setattr(t._lib, "transcribe_stream_feed", gate)
    monkeypatch.setattr(t._lib, "transcribe_run", run)
    try:
        a, a_box = _in_thread(stream.feed, PCM)
        assert gate.entered.wait(TIMEOUT)
        s1.close()  # never waits; the free is queued behind the feed
        assert fake.frees == [] and m._stream_lease.owner is not None
        spy = _LockSpy(m)
        b, b_box = _in_thread(s2.run, PCM)
        assert spy.waiting.wait(TIMEOUT)
    finally:
        gate.release.set()
    _join(a, "in-flight feed")
    _join(b, "queued run")
    assert "error" not in a_box, a_box
    assert b_box.get("value") == ("result", s2._handle.value, None), b_box
    assert seen["frees"] == [("session", h1)]
    assert m._stream_lease.owner is None
    del stream
    gc.collect()
    assert fake.resets == []
