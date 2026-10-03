// DIARIZE role: model.roles, diarizeInfo, DiarizeSession, and the compute
// rules a diarize run shares with Session (lock, in-flight mark, abort,
// disposed recheck, deferred free).

import { test } from "node:test";
import assert from "node:assert/strict";
import { getEventListeners } from "node:events";
import { modelTest, MODEL, SORTFORMER_MODEL, SORTFORMER_AUDIO, readWav, jfk } from "./common.mjs";
import {
  TranscribeModel,
  DiarizeSession,
  Aborted,
  Busy,
  InvalidArgument,
  TranscribeError,
  UnsupportedRole,
} from "../dist/index.js";

const mix = () => readWav(SORTFORMER_AUDIO);

test("DiarizeSession exposes only its public surface", () => {
  assert.deepEqual(Object.getOwnPropertyNames(DiarizeSession.prototype).sort(), [
    "constructor",
    "dispose",
    "run",
    "timings",
  ]);
});

modelTest("an ASR-only model refuses the diarize role with UnsupportedRole", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    assert.deepEqual(m.roles, ["asr"]);
    assert.throws(() => m.diarizeInfo, UnsupportedRole);
    assert.throws(() => m.createDiarizeSession(), UnsupportedRole);
    assert.equal(m.accepts({ kind: "sortformer_diarize" }), false);
    await assert.rejects(
      () => m.createSession().run(jfk(), { family: { kind: "sortformer_diarize" } }),
      InvalidArgument,
    );
  } finally {
    m.dispose();
  }
});

modelTest("sortformer serves asr and diarize; diarizeInfo", SORTFORMER_MODEL, async () => {
  const m = await TranscribeModel.load(SORTFORMER_MODEL);
  try {
    assert.deepEqual(m.roles, ["asr", "diarize"]);
    assert.deepEqual(m.diarizeInfo, { sampleRate: 16000, maxSpeakers: 4 });
    assert.equal(m.accepts({ kind: "sortformer_diarize" }), true);
  } finally {
    m.dispose();
  }
});

// Compares against the ASR path (Session.run with the SFST preset), which a
// later commit removes; drop the comparison then.
modelTest("diarize run returns the ASR path's speaker turns for the same preset", SORTFORMER_MODEL, async () => {
  const m = await TranscribeModel.load(SORTFORMER_MODEL);
  try {
    const pcm = mix();
    const d = m.createDiarizeSession({ nThreads: 4 });
    const s = m.createSession();
    for (const preset of ["default", "low_latency"]) {
      const rows = await d.run(pcm, { family: { kind: "sortformer_diarize", preset } });
      assert.ok(rows.length > 0, preset);
      for (const r of rows) assert.ok(r.speakerId >= 1 && r.speakerId <= 4 && r.t1Ms > r.t0Ms);
      const asr = await s.run(pcm, { family: { kind: "sortformer", preset } });
      assert.deepEqual(rows, asr.speakerSegments, preset);
    }
    assert.ok(d.timings.encodeMs > 0);
    d.dispose();
    s.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("a bad preset or wrong-slot extension is rejected", SORTFORMER_MODEL, async () => {
  const m = await TranscribeModel.load(SORTFORMER_MODEL);
  try {
    const d = m.createDiarizeSession();
    const pcm = mix();
    await assert.rejects(
      () => d.run(pcm, { family: { kind: "sortformer_diarize", preset: "ultra_low_latency" } }),
      (e) => e instanceof TranscribeError && /invalid sortformer preset/.test(e.message),
    );
    await assert.rejects(() => d.run(pcm, { family: { kind: "sortformer" } }), InvalidArgument);
    assert.ok((await d.run(pcm)).length > 0);
    d.dispose();
  } finally {
    m.dispose();
  }
});

// ---- compute rules shared with Session -------------------------------------
// Busy is not covered here: Sortformer cannot begin a stream, so no model
// serves both a stream lease and the diarize role. The refusal is the same
// SessionCore.exclusive gate compute-rules.test.mjs covers for Session.

// Also uses the Sortformer ASR path (Session.run); drop it with that path.
modelTest("a diarize run and an ASR run on one model are exclusive", SORTFORMER_MODEL, async () => {
  const m = await TranscribeModel.load(SORTFORMER_MODEL);
  try {
    const d = m.createDiarizeSession();
    const s = m.createSession();
    const order = [];
    const pd = d.run(mix()).then((r) => (order.push("d"), r));
    const ps = s.run(mix()).then((r) => (order.push("s"), r));
    await Promise.resolve();
    assert.throws(() => d.timings, /run\(\).*in flight/);
    assert.doesNotThrow(() => s.limits);
    const [rows, r] = await Promise.all([pd, ps]);
    assert.deepEqual(order, ["d", "s"]);
    assert.deepEqual(rows, r.speakerSegments);
  } finally {
    m.dispose();
  }
});

modelTest("the abort listener lives only for the call; a pre-aborted run raises Aborted", SORTFORMER_MODEL, async () => {
  const m = await TranscribeModel.load(SORTFORMER_MODEL);
  try {
    const d = m.createDiarizeSession();
    const ac = new AbortController();
    const p = d.run(mix(), { signal: ac.signal });
    await Promise.resolve();
    assert.equal(getEventListeners(ac.signal, "abort").length, 1);
    assert.ok((await p).length > 0);
    assert.equal(getEventListeners(ac.signal, "abort").length, 0);

    ac.abort();
    await assert.rejects(() => d.run(mix(), { signal: ac.signal }), Aborted);
    assert.equal(getEventListeners(ac.signal, "abort").length, 0);
    assert.ok((await d.run(mix())).length > 0, "the next run is not aborted");
  } finally {
    m.dispose();
  }
});

modelTest("a diarize run queued before its dispose is rejected as disposed", SORTFORMER_MODEL, async () => {
  const m = await TranscribeModel.load(SORTFORMER_MODEL);
  try {
    const running = m.createDiarizeSession().run(mix());
    const d = m.createDiarizeSession();
    const ac = new AbortController();
    const queued = d.run(mix(), { signal: ac.signal });
    d.dispose();
    await assert.rejects(queued, (e) => !(e instanceof Busy) && /disposed/i.test(e.message));
    assert.equal(getEventListeners(ac.signal, "abort").length, 0);
    await running;
    await assert.rejects(() => d.run(mix()), /disposed/);
  } finally {
    m.dispose();
  }
});

modelTest("dispose during an in-flight diarize run defers the free; the rows survive", SORTFORMER_MODEL, async () => {
  const m = await TranscribeModel.load(SORTFORMER_MODEL);
  const d = m.createDiarizeSession();
  const p = d.run(mix());
  await Promise.resolve();
  d.dispose();
  assert.throws(() => d.timings, /run\(\).*in flight/);
  assert.ok((await p).length > 0);
  assert.throws(() => d.timings, /disposed/);

  const d2 = m.createDiarizeSession();
  const p2 = d2.run(mix());
  await Promise.resolve();
  m.dispose(); // disposes d2 too; both native frees queue behind the run
  assert.throws(() => m.diarizeInfo, /disposed/);
  assert.ok((await p2).length > 0);
});
