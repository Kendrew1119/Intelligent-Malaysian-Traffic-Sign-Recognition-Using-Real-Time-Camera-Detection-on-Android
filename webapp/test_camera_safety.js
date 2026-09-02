const assert = require("node:assert/strict");
const safety = require("./static/camera_safety.js");

const box = (x1, y1, x2, y2) => ({ x1, y1, x2, y2 });
const detection = (confidence, bbox) => ({
  class_id: 30,
  class_name: "no-uturn",
  confidence,
  bbox,
});

assert.equal(safety.bboxIou(box(0, 0, 20, 20), box(0, 0, 20, 20)), 1);
assert.equal(safety.bboxIou(box(0, 0, 20, 20), box(30, 30, 50, 50)), 0);
assert.deepEqual(safety.fitFrameSize(1280, 720), { width: 960, height: 540 });
assert.deepEqual(safety.fitFrameSize(640, 480), { width: 640, height: 480 });

const classified = safety.classifyDetections([
  detection(0.72, box(100, 100, 180, 180)),
  detection(0.27, box(220, 100, 300, 180)),
  detection(0.88, box(10, 10, 22, 22)),
], 1280, 720);
assert.equal(classified.confirmed.length, 1);
assert.equal(classified.uncertain.length, 1);
assert.equal(classified.ignoredSmall.length, 1);

const unclearNumber = {
  ...detection(0.91, box(100, 100, 180, 180)),
  class_name: "speed-limit-80",
  detector_class_name: "speed-limit-30",
  speech_blocked: true,
};
const blocked = safety.classifyDetections([unclearNumber], 1280, 720);
assert.equal(blocked.confirmed.length, 0);
assert.equal(blocked.uncertain.length, 1);
assert.equal(blocked.uncertain[0].uncertainty_reason, "number-unclear");

const trackOptions = {
  minIou: 0.10,
  maxCenterShift: 1.25,
  minAreaRatio: 0.25,
  ttlMs: 1800,
  maxMisses: 1,
};
assert.equal(
  safety.movementMatches(box(100, 100, 180, 180), box(190, 100, 270, 180), trackOptions),
  true,
);
assert.equal(
  safety.movementMatches(box(100, 100, 180, 180), box(500, 300, 580, 380), trackOptions),
  false,
);

const first = safety.advanceTrack(
  null,
  detection(0.72, box(100, 100, 180, 180)),
  1000,
  trackOptions,
);
const missedOnce = safety.markTrackMissed(first, 1400, trackOptions);
const second = safety.advanceTrack(
  missedOnce,
  detection(0.68, box(190, 100, 270, 180)),
  1800,
  trackOptions,
);
const moved = safety.advanceTrack(
  second,
  detection(0.8, box(500, 300, 580, 380)),
  2200,
  trackOptions,
);
assert.equal(first.count, 1);
assert.equal(missedOnce.misses, 1);
assert.equal(second.count, 2);
assert.equal(moved.count, 1);
assert.equal(safety.confirmationRequirement(second), 2);

const normalFirst = safety.advanceTrack(
  null,
  detection(0.55, box(100, 100, 180, 180)),
  1000,
  trackOptions,
);
const normalSecond = safety.advanceTrack(
  normalFirst,
  detection(0.58, box(110, 105, 190, 185)),
  1400,
  trackOptions,
);
const normalThird = safety.advanceTrack(
  normalSecond,
  detection(0.57, box(120, 110, 200, 190)),
  1800,
  trackOptions,
);
assert.equal(safety.confirmationRequirement(normalSecond), 3);
assert.equal(safety.confirmationRequirement(normalThird), 3);
assert.equal(normalThird.count, 3);

const firstMiss = safety.markTrackMissed(normalThird, 2000, trackOptions);
assert.ok(firstMiss);
assert.equal(safety.markTrackMissed(firstMiss, 2200, trackOptions), null);
assert.equal(safety.markTrackMissed(normalThird, 4000, trackOptions), null);

console.log("Camera safety tests passed.");
