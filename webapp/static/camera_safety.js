(function cameraSafetyModule(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CameraSafety = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function buildCameraSafety() {
  function bboxIou(first, second) {
    const left = Math.max(first.x1, second.x1);
    const top = Math.max(first.y1, second.y1);
    const right = Math.min(first.x2, second.x2);
    const bottom = Math.min(first.y2, second.y2);
    const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
    const firstArea = Math.max(0, first.x2 - first.x1) * Math.max(0, first.y2 - first.y1);
    const secondArea = Math.max(0, second.x2 - second.x1) * Math.max(0, second.y2 - second.y1);
    const union = firstArea + secondArea - intersection;
    return union > 0 ? intersection / union : 0;
  }

  function minimumReadableSide(frameWidth, frameHeight, minPixels = 24, minRatio = 0.035) {
    return Math.max(minPixels, Math.round(Math.min(frameWidth, frameHeight) * minRatio));
  }

  function fitFrameSize(sourceWidth, sourceHeight, maxWidth = 960, maxHeight = 540) {
    if (sourceWidth <= 0 || sourceHeight <= 0) return { width: maxWidth, height: maxHeight };
    const scale = Math.min(1, maxWidth / sourceWidth, maxHeight / sourceHeight);
    return {
      width: Math.max(1, Math.round(sourceWidth * scale)),
      height: Math.max(1, Math.round(sourceHeight * scale)),
    };
  }

  function isReadableSize(detection, frameWidth, frameHeight, options = {}) {
    const minimum = minimumReadableSide(
      frameWidth,
      frameHeight,
      options.minPixels,
      options.minRatio,
    );
    const width = detection.bbox.x2 - detection.bbox.x1;
    const height = detection.bbox.y2 - detection.bbox.y1;
    return width >= minimum && height >= minimum;
  }

  function classifyDetections(detections, frameWidth, frameHeight, options = {}) {
    const confirmationThreshold = options.confirmationThreshold ?? 0.35;
    const uncertaintyFloor = options.uncertaintyFloor ?? 0.20;
    const confirmed = [];
    const uncertain = [];
    const ignoredSmall = [];

    detections.forEach((detection) => {
      if (!isReadableSize(detection, frameWidth, frameHeight, options)) {
        ignoredSmall.push({ ...detection, detection_state: "too-small" });
      } else if (detection.speech_blocked) {
        uncertain.push({
          ...detection,
          detection_state: "uncertain",
          uncertainty_reason: "number-unclear",
        });
      } else if (detection.confidence >= confirmationThreshold) {
        confirmed.push({ ...detection, detection_state: "candidate" });
      } else if (detection.confidence >= uncertaintyFloor) {
        uncertain.push({ ...detection, detection_state: "uncertain" });
      }
    });

    return { confirmed, uncertain, ignoredSmall };
  }

  function movementMatches(previousBox, currentBox, options = {}) {
    const minIou = options.minIou ?? 0.10;
    const maxCenterShift = options.maxCenterShift ?? 1.25;
    const minAreaRatio = options.minAreaRatio ?? 0.25;
    if (bboxIou(previousBox, currentBox) >= minIou) return true;

    const previousWidth = Math.max(1, previousBox.x2 - previousBox.x1);
    const previousHeight = Math.max(1, previousBox.y2 - previousBox.y1);
    const currentWidth = Math.max(1, currentBox.x2 - currentBox.x1);
    const currentHeight = Math.max(1, currentBox.y2 - currentBox.y1);
    const previousArea = previousWidth * previousHeight;
    const currentArea = currentWidth * currentHeight;
    const areaRatio = Math.min(previousArea, currentArea) / Math.max(previousArea, currentArea);
    if (areaRatio < minAreaRatio) return false;

    const previousCenterX = (previousBox.x1 + previousBox.x2) / 2;
    const previousCenterY = (previousBox.y1 + previousBox.y2) / 2;
    const currentCenterX = (currentBox.x1 + currentBox.x2) / 2;
    const currentCenterY = (currentBox.y1 + currentBox.y2) / 2;
    const centerDistance = Math.hypot(
      currentCenterX - previousCenterX,
      currentCenterY - previousCenterY,
    );
    const referenceDiagonal = Math.max(
      Math.hypot(previousWidth, previousHeight),
      Math.hypot(currentWidth, currentHeight),
    );
    return centerDistance <= referenceDiagonal * maxCenterShift;
  }

  function advanceTrack(previous, detection, now = Date.now(), options = {}) {
    const ttlMs = options.ttlMs ?? 1800;
    const maxMisses = options.maxMisses ?? 1;
    const previousSeenAt = previous?.lastSeenAt ?? now;
    const isRecent = previous && now - previousSeenAt <= ttlMs;
    const hasMissBudget = previous && (previous.misses ?? 0) <= maxMisses;
    const matches = isRecent
      && hasMissBudget
      && movementMatches(previous.bbox, detection.bbox, options);
    const confidences = matches
      ? [...(previous.confidences ?? []), detection.confidence].slice(-3)
      : [detection.confidence];
    return {
      count: matches ? previous.count + 1 : 1,
      bbox: { ...detection.bbox },
      confidences,
      lastSeenAt: now,
      misses: 0,
      announced: matches ? Boolean(previous.announced) : false,
    };
  }

  function markTrackMissed(track, now = Date.now(), options = {}) {
    const ttlMs = options.ttlMs ?? 1800;
    const maxMisses = options.maxMisses ?? 1;
    const misses = (track.misses ?? 0) + 1;
    if (now - track.lastSeenAt > ttlMs || misses > maxMisses) return null;
    return { ...track, misses };
  }

  function confirmationRequirement(track, options = {}) {
    const highConfidence = options.highConfidence ?? 0.60;
    const highConfidenceMatches = options.highConfidenceMatches ?? 2;
    const normalMatches = options.normalMatches ?? 3;
    const recent = (track.confidences ?? []).slice(-highConfidenceMatches);
    return recent.length >= highConfidenceMatches
      && recent.every((value) => value >= highConfidence)
      ? highConfidenceMatches
      : normalMatches;
  }

  return {
    bboxIou,
    fitFrameSize,
    minimumReadableSide,
    isReadableSize,
    classifyDetections,
    movementMatches,
    advanceTrack,
    markTrackMissed,
    confirmationRequirement,
  };
}));
