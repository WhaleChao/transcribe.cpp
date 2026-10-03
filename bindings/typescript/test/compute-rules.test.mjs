// Characterization of the rules every native compute call obeys: model-wide
// exclusion, active-stream rejection, the disposed recheck and its order
// against Busy, cancel install/cleanup, the model staying alive for the call,
// deferred frees when dispose() races an in-flight worker call, and results
// copied out before the model lock is released. These pin today's behavior,
// including where call sites differ, so routing compute through one helper can
// be checked as behavior-preserving.

import assert from "node:assert/strict";
import { getEventListeners } from "node:events";
import { modelTest, MODEL, STREAMING_MODEL, jfk } from "./common.mjs";
import { TranscribeModel, Aborted, Busy, InvalidArgument } from "../dist/index.js";

const second = () => jfk().subarray(0, 16000);
const half = () => {
  const pcm = jfk();
  return pcm.subarray(0, pcm.length / 2);
};

// ---- model-wide exclusion --------------------------------------------------

modelTest("compute is exclusive model-wide: a sibling run waits its turn", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const a = m.createSession();
    const b = m.createSession();
    const order = [];
    const pa = a.run(jfk()).then((r) => (order.push("a"), r));
    const pb = b.run(jfk()).then((r) => (order.push("b"), r));
    await Promise.resolve(); // a's body reaches its worker; b's is queued
    // Only the lock holder is marked in flight; the queued sibling is not.
    assert.throws(() => a.limits, /run\(\).*in flight/);
    assert.doesNotThrow(() => b.limits);
    const [ra, rb] = await Promise.all([pa, pb]);
    assert.deepEqual(order, ["a", "b"], "FIFO: the queued sibling runs after");
    assert.match(ra.text, /ask not/i);
    assert.match(rb.text, /ask not/i);
    a.dispose();
    b.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("runBatch and finalize mark the session in flight with their own label", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    const s = m.createSession();
    const pending = s.runBatch([second()]);
    await Promise.resolve();
    assert.throws(() => s.limits, /runBatch\(\).*in flight/);
    await pending;
    assert.doesNotThrow(() => s.limits);

    const stream = await s.stream({ commitPolicy: "stable_prefix" });
    await stream.feed(second());
    const fin = stream.finalize();
    await Promise.resolve();
    assert.throws(() => stream.text, /feed\(\)\/finalize\(\).*in flight/);
    assert.throws(() => s.wasAborted, /feed\(\)\/finalize\(\).*in flight/);
    await fin;
    assert.doesNotThrow(() => stream.text);
    stream.reset();
    s.dispose();
  } finally {
    m.dispose();
  }
});

// ---- active-stream rejection -----------------------------------------------

modelTest("an active stream refuses every offline call and begin, same or sibling session", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    const a = m.createSession();
    const b = m.createSession();
    const stream = await a.stream({ commitPolicy: "stable_prefix" });
    await stream.feed(second());
    for (const s of [a, b]) {
      await assert.rejects(() => s.run(second()), (e) => e instanceof Busy && /^cannot run:/.test(e.message));
      await assert.rejects(
        () => s.runBatch([second()]),
        (e) => e instanceof Busy && /^cannot runBatch:/.test(e.message),
      );
      await assert.rejects(
        () => s.stream(),
        (e) => e instanceof Busy && /^cannot begin a stream:/.test(e.message),
      );
    }
    // The stream holding the lease keeps feeding through all of that.
    await stream.feed(second());
    await stream.finalize();
    stream.reset();
    a.dispose();
    b.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("disposal is checked before the stream lease inside the lock", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    const a = m.createSession();
    const stream = await a.stream({ commitPolicy: "stable_prefix" });
    await stream.feed(second());
    // Each call is queued while a's stream holds the lease, then its session
    // is disposed before the queued body runs: the caller sees "disposed",
    // not Busy.
    for (const call of [(s) => s.run(second()), (s) => s.runBatch([second()]), (s) => s.stream()]) {
      const b = m.createSession();
      const p = call(b);
      b.dispose();
      await assert.rejects(p, (e) => !(e instanceof Busy) && /disposed/i.test(e.message));
    }
    stream.reset();
    a.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("a feed or finalize queued before session dispose still runs", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    // feed/finalize check disposal only before queuing, not inside the lock:
    // a call already queued when the session is disposed runs to completion
    // (the native free is queued behind it).
    const a = m.createSession();
    const sa = await a.stream({ commitPolicy: "stable_prefix" });
    const feed = sa.feed(second());
    a.dispose();
    const u = await feed;
    assert.equal(typeof u.revision, "number");

    const b = m.createSession();
    const sb = await b.stream({ commitPolicy: "stable_prefix" });
    await sb.feed(second());
    const fin = sb.finalize();
    b.dispose();
    const f = await fin;
    assert.equal(f.isFinal, true);

    // Both leases are gone: a fresh session can stream.
    const c = m.createSession();
    const sc = await c.stream({ commitPolicy: "stable_prefix" });
    sc.reset();
    c.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("a rejected feed releases the stream lease (current behavior)", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    const a = m.createSession();
    const b = m.createSession();
    const stream = await a.stream({ commitPolicy: "stable_prefix" });
    await stream.feed(second());
    const bad = new Float32Array(1600);
    bad[5] = Number.NaN;
    await assert.rejects(() => stream.feed(bad), InvalidArgument);
    // The native stream stays ACTIVE after a non-finite feed, but the binding
    // releases the model-wide lease on any non-OK feed status, so a sibling
    // run is no longer refused with Busy. Pinned as-is so that changing it is
    // a deliberate, visible decision. The probe run is itself non-finite, so
    // native rejects it up front and no compute overlaps the live stream.
    assert.equal(stream.state, "active");
    await assert.rejects(() => b.run(bad), (e) => e instanceof InvalidArgument && !(e instanceof Busy));
    stream.reset();
    a.dispose();
    b.dispose();
  } finally {
    m.dispose();
  }
});

// ---- cancel install / cleanup ----------------------------------------------

modelTest("the abort listener lives only for the call and is removed after", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const s = m.createSession();
    const ac = new AbortController();
    const p = s.run(jfk(), { signal: ac.signal });
    await Promise.resolve();
    assert.equal(getEventListeners(ac.signal, "abort").length, 1, "installed for the run");
    const r = await p;
    assert.equal(r.aborted, false);
    assert.equal(getEventListeners(ac.signal, "abort").length, 0, "removed after the run");

    const pb = s.runBatch([second()], { signal: ac.signal });
    await Promise.resolve();
    assert.equal(getEventListeners(ac.signal, "abort").length, 1, "installed for the batch");
    await pb;
    assert.equal(getEventListeners(ac.signal, "abort").length, 0, "removed after the batch");
    s.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("an aborted call uninstalls its callback; the next call is not aborted", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const s = m.createSession();
    const ac = new AbortController();
    ac.abort();
    const items = await s.runBatch([jfk(), second()], { signal: ac.signal });
    assert.equal(items.length, 2);
    for (const it of items) {
      assert.equal(it.ok, false);
      assert.ok(it.error instanceof Aborted);
      assert.ok(it.error.partialResult);
    }
    assert.equal(getEventListeners(ac.signal, "abort").length, 0);
    await assert.rejects(() => s.run(jfk(), { signal: ac.signal }), Aborted);
    assert.equal(getEventListeners(ac.signal, "abort").length, 0);

    const r = await s.run(jfk());
    assert.equal(r.aborted, false);
    assert.equal(s.wasAborted, false);
    assert.match(r.text, /ask not/i);
    s.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("a refused call never installs its abort listener", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    const a = m.createSession();
    const b = m.createSession();
    const stream = await a.stream({ commitPolicy: "stable_prefix" });
    const ac = new AbortController();
    await assert.rejects(() => b.run(second(), { signal: ac.signal }), Busy);
    await assert.rejects(() => b.runBatch([second()], { signal: ac.signal }), Busy);
    assert.equal(getEventListeners(ac.signal, "abort").length, 0, "Busy is decided before install");

    const c = m.createSession();
    const p = c.run(second(), { signal: ac.signal });
    c.dispose();
    await assert.rejects(p, /disposed/i);
    assert.equal(getEventListeners(ac.signal, "abort").length, 0, "disposed is decided before install");

    stream.reset();
    a.dispose();
    b.dispose();
  } finally {
    m.dispose();
  }
});

// ---- dispose racing an in-flight call; keepalive; copy-out -----------------

modelTest("session dispose during an in-flight run defers the free; the result survives", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const s = m.createSession();
    const p = s.run(jfk());
    await Promise.resolve(); // the run is on its worker
    s.dispose();
    // The in-flight guard is checked before disposal on reads.
    assert.throws(() => s.limits, /run\(\).*in flight/);
    const r = await p;
    assert.match(r.text, /ask not what your country/i);
    assert.ok(r.segments.length >= 1);
    assert.throws(() => s.limits, /disposed/);

    const s2 = m.createSession();
    const pb = s2.runBatch([jfk(), half()]);
    await Promise.resolve();
    s2.dispose();
    const items = await pb;
    assert.equal(items.length, 2);
    assert.ok(items[0].ok && /ask not what your country/i.test(items[0].result.text));
  } finally {
    m.dispose();
  }
});

modelTest("model dispose during an in-flight run keeps the model alive for the call", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  const s = m.createSession();
  const p = s.run(jfk());
  await Promise.resolve();
  m.dispose(); // disposes s too; both native frees queue behind the run
  assert.throws(() => m.capabilities, /disposed/);
  const r = await p;
  assert.match(r.text, /ask not what your country/i);
  assert.throws(() => s.limits, /disposed/);

  const m2 = await TranscribeModel.load(MODEL);
  const s2 = m2.createSession();
  const pb = s2.runBatch([jfk()]);
  await Promise.resolve();
  m2.dispose();
  const items = await pb;
  assert.ok(items[0].ok && /ask not what your country/i.test(items[0].result.text));
});

modelTest("model dispose during an in-flight feed keeps the model alive for the call", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  const s = m.createSession();
  const stream = await s.stream({ commitPolicy: "stable_prefix" });
  const p = stream.feed(second());
  await Promise.resolve();
  m.dispose();
  const u = await p;
  assert.equal(typeof u.revision, "number");
  assert.throws(() => stream.text, /disposed/i);
});

modelTest("results are copied out before the next call on the session overwrites them", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const s = m.createSession();
    // Both run on the same session back to back; the second overwrites the
    // session's native result storage as soon as it gets the lock.
    const [full, part] = await Promise.all([s.run(jfk()), s.run(half())]);
    assert.match(full.text, /ask what you can do for your country/i);
    assert.doesNotMatch(part.text, /ask what you can do for your country/i);

    const [b1, b2] = await Promise.all([s.runBatch([jfk()]), s.runBatch([half()])]);
    assert.match(b1[0].result.text, /ask what you can do for your country/i);
    assert.doesNotMatch(b2[0].result.text, /ask what you can do for your country/i);
    s.dispose();
  } finally {
    m.dispose();
  }
});
