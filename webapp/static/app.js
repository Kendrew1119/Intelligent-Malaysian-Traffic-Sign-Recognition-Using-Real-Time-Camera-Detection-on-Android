const state = {
  mode: "upload",
  selectedFile: null,
  sourceImage: null,
  stream: null,
  cameraRunning: false,
  requestInFlight: false,
  lastCameraRequest: 0,
  latestDetections: [],
  signCatalog: new Map(),
  lastFrameBlob: null,
  hardCaseCount: 0,
  stability: new Map(),
  announcementCooldown: new Map(),
  history: [],
};

const ui = {
  serverStatus: document.querySelector("#serverStatus"),
  modeButtons: document.querySelectorAll(".mode-button"),
  uploadControls: document.querySelector("#uploadControls"),
  cameraControls: document.querySelector("#cameraControls"),
  imageInput: document.querySelector("#imageInput"),
  chooseButton: document.querySelector("#chooseButton"),
  detectButton: document.querySelector("#detectButton"),
  fileName: document.querySelector("#fileName"),
  cameraButton: document.querySelector("#cameraButton"),
  stopCameraButton: document.querySelector("#stopCameraButton"),
  speechToggle: document.querySelector("#speechToggle"),
  guidancePanel: document.querySelector("#guidancePanel"),
  guidanceMessage: document.querySelector("#guidanceMessage"),
  guidanceDetail: document.querySelector("#guidanceDetail"),
  guidanceState: document.querySelector("#guidanceState"),
  dropZone: document.querySelector("#dropZone"),
  emptyState: document.querySelector("#emptyState"),
  video: document.querySelector("#cameraVideo"),
  canvas: document.querySelector("#detectionCanvas"),
  processing: document.querySelector("#processingOverlay"),
  confidenceSlider: document.querySelector("#confidenceSlider"),
  confidenceValue: document.querySelector("#confidenceValue"),
  count: document.querySelector("#detectionCount"),
  resultEmpty: document.querySelector("#resultEmpty"),
  resultList: document.querySelector("#resultList"),
  timingRow: document.querySelector("#timingRow"),
  timingValue: document.querySelector("#timingValue"),
  historyEmpty: document.querySelector("#historyEmpty"),
  historyTableWrap: document.querySelector("#historyTableWrap"),
  historyBody: document.querySelector("#historyBody"),
  exportButton: document.querySelector("#exportButton"),
  clearButton: document.querySelector("#clearButton"),
  hardCaseButton: document.querySelector("#hardCaseButton"),
  hardCaseCount: document.querySelector("#hardCaseCount"),
  hardCaseDialog: document.querySelector("#hardCaseDialog"),
  hardCaseForm: document.querySelector("#hardCaseForm"),
  dialogCloseButton: document.querySelector("#dialogCloseButton"),
  dialogCancelButton: document.querySelector("#dialogCancelButton"),
  saveHardCaseButton: document.querySelector("#saveHardCaseButton"),
  issueType: document.querySelector("#issueType"),
  expectedClass: document.querySelector("#expectedClass"),
  hardCaseNotes: document.querySelector("#hardCaseNotes"),
  signOptions: document.querySelector("#signOptions"),
  toast: document.querySelector("#toast"),
};

const context = ui.canvas.getContext("2d");
const captureCanvas = document.createElement("canvas");
const captureContext = captureCanvas.getContext("2d");
const palette = ["#2468d8", "#e55252", "#16835d", "#b664d8", "#d88a21", "#0089a8"];
const UPLOAD_CONFIDENCE_PERCENT = 20;
const CAMERA_CONFIDENCE_PERCENT = 35;
const CAMERA_UNCERTAINTY_FLOOR = 0.20;
const CAMERA_REQUEST_INTERVAL_MS = 400;
const CAMERA_CAPTURE_MAX_WIDTH = 960;
const CAMERA_CAPTURE_MAX_HEIGHT = 540;
const CAMERA_HIGH_CONFIDENCE = 0.60;
const CAMERA_HIGH_CONFIDENCE_MATCHES = 2;
const CAMERA_NORMAL_MATCHES = 3;
const CAMERA_TRACK_OPTIONS = {
  minIou: 0.10,
  maxCenterShift: 1.25,
  minAreaRatio: 0.25,
  ttlMs: 1800,
  maxMisses: 1,
};

function friendlyName(name) {
  return name.split("-").map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
}

function detectionName(detection) {
  return friendlyName(detection.class_name);
}

function colourFor(classId) {
  return palette[classId % palette.length];
}

function confidence() {
  return Number(ui.confidenceSlider.value) / 100;
}

function speechPhrase(detection) {
  if (detection.class_name.startsWith("speed-limit-")) {
    const speed = detection.class_name.slice("speed-limit-".length);
    return `Speed limit ${speed} kilometres per hour ahead.`;
  }
  return state.signCatalog.get(detection.class_name)?.speech || `${friendlyName(detection.class_name)} ahead.`;
}

function signDetails(detection) {
  const catalogEntry = state.signCatalog.get(detection.class_name);
  if (catalogEntry) return catalogEntry;
  if (detection.class_name.startsWith("speed-limit-")) {
    const speed = detection.class_name.slice("speed-limit-".length);
    return {
      meaning: `Maximum posted speed is ${speed} km/h.`,
      action: `Do not exceed ${speed} km/h while the limit applies.`,
    };
  }
  return null;
}

function setGuidance(message, detail, status = "Ready", stateName = "idle") {
  ui.guidanceMessage.textContent = message;
  ui.guidanceDetail.textContent = detail;
  ui.guidanceState.textContent = status;
  ui.guidancePanel.dataset.state = stateName;
}

function showToast(message) {
  ui.toast.textContent = message;
  ui.toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => ui.toast.classList.remove("show"), 3600);
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    if (!response.ok || !health.ready) throw new Error(health.error || "Model is not ready");
    ui.serverStatus.className = "server-status ready";
    ui.serverStatus.innerHTML = '<span class="status-dot"></span><span>Model ready</span>';
  } catch (error) {
    ui.serverStatus.className = "server-status error";
    ui.serverStatus.innerHTML = '<span class="status-dot"></span><span>Model unavailable</span>';
    showToast(error.message);
  }
}

async function loadSignCatalog() {
  try {
    const response = await fetch("/api/signs");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Sign information is unavailable");
    state.signCatalog = new Map(payload.signs.map((sign) => [sign.class_name, sign]));
    ui.signOptions.innerHTML = "";
    payload.signs.forEach((sign) => {
      const option = document.createElement("option");
      option.value = sign.class_name;
      option.label = friendlyName(sign.class_name);
      ui.signOptions.appendChild(option);
    });
  } catch (error) {
    showToast(error.message);
  }
}

async function loadHardCaseCount() {
  try {
    const response = await fetch("/api/hard-cases");
    const payload = await response.json();
    if (!response.ok) return;
    updateHardCaseCount(payload.saved);
  } catch (_) {
    updateHardCaseCount(0);
  }
}

function updateHardCaseCount(count) {
  state.hardCaseCount = count;
  ui.hardCaseCount.textContent = `${count} difficult frame${count === 1 ? "" : "s"} saved`;
}

function setMode(mode) {
  if (state.mode === "camera" && mode !== "camera") stopCamera();
  state.mode = mode;
  state.stability.clear();
  ui.confidenceSlider.min = mode === "camera" ? String(CAMERA_CONFIDENCE_PERCENT) : "10";
  ui.confidenceSlider.value = String(mode === "camera" ? CAMERA_CONFIDENCE_PERCENT : UPLOAD_CONFIDENCE_PERCENT);
  ui.confidenceValue.value = `${ui.confidenceSlider.value}%`;
  ui.modeButtons.forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  ui.uploadControls.classList.toggle("hidden", mode !== "upload");
  ui.cameraControls.classList.toggle("hidden", mode !== "camera");
  ui.dropZone.setAttribute("aria-label", mode === "upload" ? "Choose or drop a traffic sign image" : "Live traffic sign camera");
  state.latestDetections = [];
  state.lastFrameBlob = mode === "upload" ? state.selectedFile : null;
  ui.hardCaseButton.disabled = !state.lastFrameBlob;
  renderResults([], null);
  setGuidance(
    mode === "camera" ? "Camera ready" : "Ready to detect",
    mode === "camera" ? "Moving signs are confirmed after two strong or three normal matching detections." : "Use an image or start the live camera.",
  );

  if (mode === "upload" && state.sourceImage) {
    drawUploadFrame([]);
  } else {
    resetStage(mode === "camera" ? "Camera is ready to start" : null);
  }
}

function resetStage(message = null) {
  ui.dropZone.classList.remove("has-media", "camera-active");
  ui.emptyState.classList.remove("hidden");
  ui.emptyState.querySelector("strong").textContent = message || "Drop a road image here";
  ui.emptyState.querySelector("span").textContent = message ? "Allow camera access when your browser asks." : "JPG, PNG, WebP or BMP · up to 12 MB";
  context.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
}

function selectFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showToast("Choose a JPG, PNG, WebP, or BMP image.");
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    showToast("Image must be 12 MB or smaller.");
    return;
  }
  const image = new Image();
  const url = URL.createObjectURL(file);
  image.onload = () => {
    URL.revokeObjectURL(url);
    state.selectedFile = file;
    state.lastFrameBlob = file;
    state.sourceImage = image;
    ui.fileName.textContent = file.name;
    ui.detectButton.disabled = false;
    ui.hardCaseButton.disabled = false;
    ui.emptyState.classList.add("hidden");
    ui.dropZone.classList.add("has-media");
    drawUploadFrame([]);
    renderResults([], null);
  };
  image.onerror = () => {
    URL.revokeObjectURL(url);
    showToast("That image could not be opened.");
  };
  image.src = url;
}

function fitCanvas(width, height) {
  ui.canvas.width = width;
  ui.canvas.height = height;
}

function drawBoxes(detections) {
  const lineWidth = Math.max(3, Math.round(Math.min(ui.canvas.width, ui.canvas.height) / 180));
  context.lineWidth = lineWidth;
  context.font = `700 ${Math.max(16, Math.round(ui.canvas.width / 45))}px Inter, system-ui, sans-serif`;
  context.textBaseline = "top";

  detections.forEach((detection) => {
    const { x1, y1, x2, y2 } = detection.bbox;
    const uncertain = detection.detection_state === "uncertain";
    const colour = uncertain ? "#a56700" : colourFor(detection.class_id);
    const prefix = uncertain ? "Uncertain: " : "";
    const label = `${prefix}${detectionName(detection)} ${Math.round(detection.confidence * 100)}%`;
    context.strokeStyle = colour;
    context.setLineDash(uncertain ? [12, 7] : []);
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const metrics = context.measureText(label);
    const labelHeight = Math.max(26, Math.round(ui.canvas.width / 35));
    const labelY = Math.max(0, y1 - labelHeight);
    context.fillStyle = colour;
    context.fillRect(x1, labelY, metrics.width + 18, labelHeight);
    context.fillStyle = "white";
    context.fillText(label, x1 + 9, labelY + 4);
    context.setLineDash([]);
  });
}

function drawUploadFrame(detections) {
  if (!state.sourceImage) return;
  fitCanvas(state.sourceImage.naturalWidth, state.sourceImage.naturalHeight);
  context.drawImage(state.sourceImage, 0, 0);
  drawBoxes(detections);
}

async function postImage(blob, fileName = "camera-frame.jpg", threshold = confidence()) {
  const form = new FormData();
  form.append("image", blob, fileName);
  form.append("confidence", threshold.toFixed(2));
  const response = await fetch("/api/detect", { method: "POST", body: form });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Detection failed");
  return payload;
}

async function detectUpload() {
  if (!state.selectedFile || state.requestInFlight) return;
  state.requestInFlight = true;
  ui.processing.classList.remove("hidden");
  ui.detectButton.disabled = true;
  try {
    const result = await postImage(state.selectedFile, state.selectedFile.name);
    state.latestDetections = result.detections;
    drawUploadFrame(result.detections);
    renderResults(result.detections, result.total_ms);
    if (result.detections.length) {
      const strongest = [...result.detections].sort((a, b) => b.confidence - a.confidence)[0];
      setGuidance(
        speechPhrase(strongest),
        `Detected in the image at ${Math.round(strongest.confidence * 100)}% confidence.`,
        "Detected",
        "confirmed",
      );
    } else {
      setGuidance("No supported sign found", "Try a clearer or closer image, or lower the threshold carefully.", "No match");
    }
    result.detections.forEach((detection) => addHistory(detection, "Image"));
  } catch (error) {
    showToast(error.message);
  } finally {
    state.requestInFlight = false;
    ui.processing.classList.add("hidden");
    ui.detectButton.disabled = false;
  }
}

function renderResults(detections, totalMs) {
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  ui.count.textContent = sorted.length;
  ui.resultEmpty.classList.toggle("hidden", sorted.length > 0);
  ui.resultList.innerHTML = "";
  sorted.forEach((detection) => {
    const item = document.createElement("li");
    const uncertain = detection.detection_state === "uncertain";
    item.className = `result-item${uncertain ? " uncertain" : ""}`;
    const sign = signDetails(detection);
    const guidance = sign && !uncertain ? `
      <div class="result-guidance">
        <p>${sign.meaning}</p>
        <p><strong>Action:</strong> ${sign.action}</p>
      </div>` : "";
    const ocrConfidence = detection.speed_limit_ocr?.confidence;
    const ocrLabel = detection.classification_source === "ocr" && ocrConfidence
      ? ` · OCR ${Math.round(ocrConfidence * 100)}%`
      : detection.classification_source === "yolo+ocr"
        ? " · OCR confirmed"
        : "";
    const stateLabel = `${uncertain ? "Uncertain · not spoken" : `Class ${detection.class_id}`}${ocrLabel}`;
    item.innerHTML = `
      <span class="result-color" style="background:${colourFor(detection.class_id)}"></span>
      <span><span class="result-name">${detectionName(detection)}</span><span class="result-id">${stateLabel}</span></span>
      <span class="result-confidence">${Math.round(detection.confidence * 100)}%</span>
      ${guidance}`;
    ui.resultList.appendChild(item);
  });
  ui.timingRow.classList.toggle("hidden", totalMs === null);
  if (totalMs !== null) ui.timingValue.textContent = `${Math.round(totalMs)} ms`;
  if (!sorted.length && totalMs !== null) ui.resultEmpty.textContent = "No sign passed the current threshold.";
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showToast("Camera access is not supported by this browser.");
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" }, audio: false });
    ui.video.srcObject = state.stream;
    await ui.video.play();
    state.cameraRunning = true;
    state.stability.clear();
    state.lastCameraRequest = 0;
    state.latestDetections = [];
    setGuidance("Scanning the road", "Voice guidance tracks a moving sign across matching detections.", "Scanning");
    ui.cameraButton.classList.add("hidden");
    ui.stopCameraButton.classList.remove("hidden");
    ui.emptyState.classList.add("hidden");
    ui.dropZone.classList.add("has-media", "camera-active");
    renderCamera();
  } catch (error) {
    showToast(error.name === "NotAllowedError" ? "Camera permission was not allowed." : `Camera could not start: ${error.message}`);
  }
}

function stopCamera() {
  state.cameraRunning = false;
  state.stability.clear();
  state.lastCameraRequest = 0;
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  ui.video.srcObject = null;
  ui.cameraButton.classList.remove("hidden");
  ui.stopCameraButton.classList.add("hidden");
  resetStage("Camera is ready to start");
  setGuidance("Camera stopped", "Start the camera when guidance is needed.", "Ready");
}

function renderCamera(timestamp = 0) {
  if (!state.cameraRunning) return;
  const frameSize = CameraSafety.fitFrameSize(
    ui.video.videoWidth || 1280,
    ui.video.videoHeight || 720,
    CAMERA_CAPTURE_MAX_WIDTH,
    CAMERA_CAPTURE_MAX_HEIGHT,
  );
  const { width, height } = frameSize;
  if (ui.canvas.width !== width || ui.canvas.height !== height) fitCanvas(width, height);
  context.drawImage(ui.video, 0, 0, width, height);
  drawBoxes(state.latestDetections);
  if (!state.requestInFlight && timestamp - state.lastCameraRequest >= CAMERA_REQUEST_INTERVAL_MS) {
    state.lastCameraRequest = timestamp;
    detectCameraFrame();
  }
  requestAnimationFrame(renderCamera);
}

async function detectCameraFrame() {
  if (!state.cameraRunning) return;
  state.requestInFlight = true;
  try {
    captureCanvas.width = ui.canvas.width || CAMERA_CAPTURE_MAX_WIDTH;
    captureCanvas.height = ui.canvas.height || CAMERA_CAPTURE_MAX_HEIGHT;
    captureContext.drawImage(ui.video, 0, 0, captureCanvas.width, captureCanvas.height);
    const blob = await new Promise((resolve) => captureCanvas.toBlob(resolve, "image/jpeg", 0.82));
    if (!blob) throw new Error("The camera frame could not be captured.");
    state.lastFrameBlob = blob;
    ui.hardCaseButton.disabled = false;
    const result = await postImage(blob, "camera-frame.jpg", CAMERA_UNCERTAINTY_FLOOR);
    const classified = CameraSafety.classifyDetections(
      result.detections,
      captureCanvas.width,
      captureCanvas.height,
      { confirmationThreshold: confidence(), uncertaintyFloor: CAMERA_UNCERTAINTY_FLOOR },
    );
    state.latestDetections = [...classified.confirmed, ...classified.uncertain];
    renderResults(state.latestDetections, result.total_ms);
    updateStableDetections(classified.confirmed, classified.uncertain);
  } catch (error) {
    showToast(error.message);
    stopCamera();
  } finally {
    state.requestInFlight = false;
  }
}

function updateStableDetections(detections, uncertainDetections = [], now = performance.now()) {
  const strongestByName = new Map();
  detections.forEach((detection) => {
    const current = strongestByName.get(detection.class_name);
    if (!current || detection.confidence > current.confidence) strongestByName.set(detection.class_name, detection);
  });
  const currentNames = new Set(strongestByName.keys());
  for (const [name, track] of state.stability.entries()) {
    if (currentNames.has(name)) continue;
    const missedTrack = CameraSafety.markTrackMissed(track, now, CAMERA_TRACK_OPTIONS);
    if (missedTrack) state.stability.set(name, missedTrack);
    else state.stability.delete(name);
  }

  const sorted = [...strongestByName.values()].sort((a, b) => b.confidence - a.confidence);
  sorted.forEach((detection) => {
    const previous = state.stability.get(detection.class_name);
    state.stability.set(
      detection.class_name,
      CameraSafety.advanceTrack(previous, detection, now, CAMERA_TRACK_OPTIONS),
    );
  });

  if (!sorted.length) {
    if (uncertainDetections.length) {
      const strongestUncertain = [...uncertainDetections].sort((a, b) => b.confidence - a.confidence)[0];
      setGuidance(
        `Uncertain ${detectionName(strongestUncertain)}`,
        `${Math.round(strongestUncertain.confidence * 100)}% confidence is below the voice threshold. Move closer or obtain a clearer view.`,
        "Not spoken",
        "uncertain",
      );
      return;
    }
    setGuidance("Scanning the road", "No supported sign is confirmed in the current frame.", "Scanning");
    return;
  }

  const strongest = sorted[0];
  const strongestTrack = state.stability.get(strongest.class_name);
  const count = strongestTrack?.count || 0;
  const requiredMatches = CameraSafety.confirmationRequirement(strongestTrack || {}, {
    highConfidence: CAMERA_HIGH_CONFIDENCE,
    highConfidenceMatches: CAMERA_HIGH_CONFIDENCE_MATCHES,
    normalMatches: CAMERA_NORMAL_MATCHES,
  });
  if (!strongestTrack?.announced && count < requiredMatches) {
    const remaining = requiredMatches - count;
    setGuidance(
      `Confirming ${detectionName(strongest)}`,
      `Keep the same sign in view for ${remaining} more matching detection${remaining === 1 ? "" : "s"}.`,
      `${count} of ${requiredMatches}`,
      "confirming",
    );
    return;
  }

  setGuidance(
    speechPhrase(strongest),
    ui.speechToggle.checked ? "Confirmed guidance. Voice repeats only after the cooldown." : "Confirmed guidance. Voice is muted.",
    "Confirmed",
    "confirmed",
  );
  sorted
    .filter((detection) => {
      const track = state.stability.get(detection.class_name);
      if (!track || track.announced) return false;
      const requirement = CameraSafety.confirmationRequirement(track, {
        highConfidence: CAMERA_HIGH_CONFIDENCE,
        highConfidenceMatches: CAMERA_HIGH_CONFIDENCE_MATCHES,
        normalMatches: CAMERA_NORMAL_MATCHES,
      });
      return track.count >= requirement;
    })
    .forEach((detection) => {
      addHistory(detection, "Camera");
      announce(detection);
      const track = state.stability.get(detection.class_name);
      state.stability.set(detection.class_name, { ...track, announced: true });
    });
}

function announce(detection) {
  if (!ui.speechToggle.checked || !("speechSynthesis" in window)) return;
  const now = Date.now();
  const last = state.announcementCooldown.get(detection.class_name) || 0;
  if (now - last < 5000) return;
  state.announcementCooldown.set(detection.class_name, now);
  const message = new SpeechSynthesisUtterance(speechPhrase(detection));
  message.rate = 0.92;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(message);
}

function addHistory(detection, source) {
  const entry = { time: new Date(), name: detection.class_name, confidence: detection.confidence, source };
  state.history.unshift(entry);
  state.history = state.history.slice(0, 100);
  renderHistory();
}

function renderHistory() {
  const hasEntries = state.history.length > 0;
  ui.historyEmpty.classList.toggle("hidden", hasEntries);
  ui.historyTableWrap.classList.toggle("hidden", !hasEntries);
  ui.exportButton.disabled = !hasEntries;
  ui.clearButton.disabled = !hasEntries;
  ui.historyBody.innerHTML = "";
  state.history.forEach((entry) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${entry.time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</td><td>${friendlyName(entry.name)}</td><td>${Math.round(entry.confidence * 100)}%</td><td>${entry.source}</td>`;
    ui.historyBody.appendChild(row);
  });
}

function exportHistory() {
  const rows = [["time", "class_name", "confidence", "source"], ...state.history.map((entry) => [entry.time.toISOString(), entry.name, entry.confidence, entry.source])];
  const csv = rows.map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `mysignvoice-detections-${new Date().toISOString().slice(0, 10)}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

function updateExpectedClassRequirement() {
  const required = ["missed", "wrong-class"].includes(ui.issueType.value);
  ui.expectedClass.required = required;
  ui.expectedClass.placeholder = required ? "Choose the correct class" : "Optional";
}

function openHardCaseDialog() {
  if (!state.lastFrameBlob) {
    showToast("Choose an image or wait for a camera frame first.");
    return;
  }
  ui.hardCaseForm.reset();
  updateExpectedClassRequirement();
  ui.hardCaseDialog.showModal();
}

async function saveHardCase(event) {
  event.preventDefault();
  if (!state.lastFrameBlob) return;
  const expected = ui.expectedClass.value.trim();
  if (expected && !state.signCatalog.has(expected)) {
    showToast("Choose a class from the 63-sign list.");
    ui.expectedClass.focus();
    return;
  }

  const form = new FormData();
  const fileName = state.mode === "camera" ? "camera-hard-case.jpg" : (state.selectedFile?.name || "upload-hard-case.jpg");
  form.append("image", state.lastFrameBlob, fileName);
  form.append("source", state.mode);
  form.append("issue_type", ui.issueType.value);
  form.append("expected_class", expected);
  form.append("predicted_classes", JSON.stringify([
    ...new Set(state.latestDetections.map((item) => item.detector_class_name || item.class_name)),
  ]));
  form.append("notes", ui.hardCaseNotes.value.trim());
  form.append("confidence", confidence().toFixed(2));

  ui.saveHardCaseButton.disabled = true;
  ui.saveHardCaseButton.textContent = "Saving";
  try {
    const response = await fetch("/api/hard-cases", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The difficult frame could not be saved");
    updateHardCaseCount(payload.total_saved);
    ui.hardCaseDialog.close();
    showToast("Difficult frame saved for later annotation.");
  } catch (error) {
    showToast(error.message);
  } finally {
    ui.saveHardCaseButton.disabled = false;
    ui.saveHardCaseButton.textContent = "Save frame";
  }
}

ui.modeButtons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
ui.chooseButton.addEventListener("click", () => ui.imageInput.click());
ui.imageInput.addEventListener("change", () => selectFile(ui.imageInput.files[0]));
ui.detectButton.addEventListener("click", detectUpload);
ui.cameraButton.addEventListener("click", startCamera);
ui.stopCameraButton.addEventListener("click", stopCamera);
ui.confidenceSlider.addEventListener("input", () => {
  ui.confidenceValue.value = `${ui.confidenceSlider.value}%`;
  if (state.mode === "camera") state.stability.clear();
});
ui.clearButton.addEventListener("click", () => { state.history = []; renderHistory(); });
ui.exportButton.addEventListener("click", exportHistory);
ui.hardCaseButton.addEventListener("click", openHardCaseDialog);
ui.hardCaseForm.addEventListener("submit", saveHardCase);
ui.dialogCloseButton.addEventListener("click", () => ui.hardCaseDialog.close());
ui.dialogCancelButton.addEventListener("click", () => ui.hardCaseDialog.close());
ui.issueType.addEventListener("change", updateExpectedClassRequirement);
ui.dropZone.addEventListener("click", () => { if (state.mode === "upload") ui.imageInput.click(); });
ui.dropZone.addEventListener("keydown", (event) => { if (state.mode === "upload" && (event.key === "Enter" || event.key === " ")) ui.imageInput.click(); });
["dragenter", "dragover"].forEach((eventName) => ui.dropZone.addEventListener(eventName, (event) => { event.preventDefault(); if (state.mode === "upload") ui.dropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((eventName) => ui.dropZone.addEventListener(eventName, (event) => { event.preventDefault(); ui.dropZone.classList.remove("dragging"); }));
ui.dropZone.addEventListener("drop", (event) => { if (state.mode === "upload") selectFile(event.dataTransfer.files[0]); });
window.addEventListener("beforeunload", stopCamera);

Promise.all([checkHealth(), loadSignCatalog(), loadHardCaseCount()]);
