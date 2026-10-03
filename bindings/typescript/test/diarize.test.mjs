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

modelTest("sortformer serves only diarize; diarizeInfo; ASR calls are refused", SORTFORMER_MODEL, async () => {
  const m = await TranscribeModel.load(SORTFORMER_MODEL);
  try {
    assert.deepEqual(m.roles, ["diarize"]);
    assert.deepEqual(m.diarizeInfo, { sampleRate: 16000, maxSpeakers: 4 });
    assert.equal(m.accepts({ kind: "sortformer_diarize" }), true);
    assert.throws(() => m.capabilities, UnsupportedRole);
    assert.throws(() => m.createSession(), UnsupportedRole);
  } finally {
    m.dispose();
  }
});

// Goldens are CPU (as sortformer_diarize_unit): (t0Ms, t1Ms, speakerId),
// grouped by speaker and time-ordered within one.
const GOLDEN = {
  default: [[320, 2400, 1], [7360, 9360, 1], [10240, 10640, 1], [4240, 6640, 2], [9760, 12000, 2]],
  low_latency: [[320, 2480, 1], [7360, 9360, 1], [10240, 10640, 1], [4160, 6640, 2], [9760, 12000, 2]],
};

modelTest("diarize run returns the golden speaker turns per preset", SORTFORMER_MODEL, async () => {
  const m = await TranscribeModel.load(SORTFORMER_MODEL, { backend: "cpu" });
  try {
    const pcm = mix();
    const d = m.createDiarizeSession({ nThreads: 4 });
    for (const [preset, want] of Object.entries(GOLDEN)) {
      const rows = await d.run(pcm, { family: { kind: "sortformer_diarize", preset } });
      assert.deepEqual(rows.map((r) => [r.t0Ms, r.t1Ms, r.speakerId]), want, preset);
    }
    assert.ok(d.timings.encodeMs > 0);
    d.dispose();
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
    await assert.rejects(() => d.run(pcm, { family: { kind: "whisper" } }), InvalidArgument);
    await assert.rejects(
      () => d.run(pcm, { family: { kind: "sortformer" } }),
      (e) => e instanceof InvalidArgument && /unknown family extension kind/.test(e.message),
    );
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
  const ac = new AbortController();
  const p = d.run(mix(), { signal: ac.signal });
  await Promise.resolve();
  d.dispose();
  assert.throws(() => d.timings, /run\(\).*in flight/);
  assert.ok((await p).length > 0);
  assert.equal(getEventListeners(ac.signal, "abort").length, 0);
  assert.throws(() => d.timings, /disposed/);

  const d2 = m.createDiarizeSession();
  const p2 = d2.run(mix());
  await Promise.resolve();
  m.dispose(); // disposes d2 too; both native frees queue behind the run
  assert.throws(() => m.diarizeInfo, /disposed/);
  assert.ok((await p2).length > 0);
});
