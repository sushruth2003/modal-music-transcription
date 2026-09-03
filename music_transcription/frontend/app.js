const els = {
  authCard: document.querySelector("#auth-card"),
  authForm: document.querySelector("#auth-form"),
  accessToken: document.querySelector("#access-token"),
  authError: document.querySelector("#auth-error"),
  logout: document.querySelector("#logout-button"),
  form: document.querySelector("#upload-form"),
  jobCard: document.querySelector("#job-card"),
  file: document.querySelector("#audio-file"),
  fileLabel: document.querySelector("#file-label"),
  instruments: document.querySelector("#instruments"),
  dropZone: document.querySelector("#drop-zone"),
  submit: document.querySelector("#submit-button"),
  formError: document.querySelector("#form-error"),
  empty: document.querySelector("#empty-state"),
  processing: document.querySelector("#processing-state"),
  error: document.querySelector("#error-state"),
  result: document.querySelector("#result-state"),
  processingName: document.querySelector("#processing-name"),
  progressValue: document.querySelector("#progress-value"),
  progressFill: document.querySelector("#progress-fill"),
  jobIdLine: document.querySelector("#job-id-line"),
  jobError: document.querySelector("#job-error"),
  resultName: document.querySelector("#result-name"),
  midiDownload: document.querySelector("#midi-download"),
  audio: document.querySelector("#source-audio"),
  play: document.querySelector("#play-button"),
  timeline: document.querySelector("#timeline"),
  currentTime: document.querySelector("#current-time"),
  totalTime: document.querySelector("#total-time"),
  filters: document.querySelector("#instrument-filters"),
  canvas: document.querySelector("#roll-canvas"),
  rollEmpty: document.querySelector("#roll-empty"),
  metricNotes: document.querySelector("#metric-notes"),
  metricDuration: document.querySelector("#metric-duration"),
  metricInference: document.querySelector("#metric-inference"),
  metricCost: document.querySelector("#metric-cost"),
};

const state = {
  jobId: null,
  job: null,
  notes: [],
  duration: 0,
  colors: new Map(),
  enabledInstruments: new Set(),
  playbackMode: "source",
  synthContext: null,
  synthStartAt: 0,
  synthOffset: 0,
  synthScheduledUntil: 0,
  synthTimer: null,
  synthNodes: [],
};

const palette = ["#c8f560", "#59d4d8", "#a892ff", "#ff7968", "#efb94f", "#5ea0ff"];
const stageOrder = ["submitted", "preprocessing", "transcribing", "completed"];

class AuthenticationRequired extends Error {}

function showAuth(message = "") {
  els.authCard.hidden = false;
  els.form.hidden = true;
  els.jobCard.hidden = true;
  els.logout.hidden = true;
  els.authError.textContent = message;
  els.authError.hidden = !message;
  window.setTimeout(() => els.accessToken.focus(), 0);
}

function showWorkspace() {
  els.authCard.hidden = true;
  els.form.hidden = false;
  els.jobCard.hidden = false;
  els.logout.hidden = false;
  els.authError.hidden = true;
}

function showView(name) {
  for (const [key, element] of Object.entries({
    empty: els.empty,
    processing: els.processing,
    error: els.error,
    result: els.result,
  })) {
    element.hidden = key !== name;
  }
}

function formatTime(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safe / 60);
  const remainder = Math.floor(safe % 60).toString().padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function readableError(payload, fallback) {
  if (payload && typeof payload.detail === "string") return payload.detail;
  return fallback;
}

function setSelectedFile(file) {
  if (!file) {
    els.fileLabel.textContent = "Drop audio here";
    return;
  }
  els.fileLabel.textContent = file.name;
}

function submitUpload(formData) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/transcriptions");
    request.responseType = "json";
    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) return;
      const percent = Math.max(2, Math.round((event.loaded / event.total) * 10));
      updateProcessing("submitted", percent, "Uploading recording");
    });
    request.addEventListener("load", () => {
      if (request.status >= 200 && request.status < 300) resolve(request.response);
      else if (request.status === 401) {
        showAuth("Your session expired. Enter the access code again.");
        reject(new AuthenticationRequired());
      } else reject(new Error(readableError(request.response, "Upload failed")));
    });
    request.addEventListener("error", () => reject(new Error("Could not reach the service")));
    request.send(formData);
  });
}

async function handleSubmit(event) {
  event.preventDefault();
  const file = els.file.files[0];
  if (!file) return;
  els.formError.hidden = true;
  els.submit.disabled = true;
  els.submit.querySelector("span").textContent = "Uploading…";
  showView("processing");
  updateProcessing("submitted", 2, "Uploading recording");

  const body = new FormData();
  body.append("audio", file);
  if (els.instruments.value.trim()) body.append("instruments", els.instruments.value.trim());

  try {
    const submission = await submitUpload(body);
    state.jobId = submission.job_id;
    history.replaceState({}, "", submission.result_url);
    els.jobIdLine.textContent = `JOB ${state.jobId}`;
    await pollJob();
  } catch (error) {
    if (error instanceof AuthenticationRequired) return;
    showView("empty");
    els.formError.textContent = error.message;
    els.formError.hidden = false;
  } finally {
    els.submit.disabled = false;
    els.submit.querySelector("span").textContent = "Start transcription";
  }
}

function updateProcessing(jobState, progress, label) {
  showView("processing");
  els.processingName.textContent = label || {
    submitted: "Waiting for a worker",
    preprocessing: "Normalizing the audio",
    transcribing: "Reading notes on the L4",
    completed: "Artifacts ready",
  }[jobState];
  els.progressValue.textContent = `${progress}%`;
  els.progressFill.style.width = `${progress}%`;

  const activeIndex = stageOrder.indexOf(jobState);
  document.querySelectorAll(".stage-list li").forEach((item, index) => {
    item.classList.toggle("complete", index < activeIndex || jobState === "completed");
    item.classList.toggle("active", index === activeIndex && jobState !== "completed");
  });
}

async function getJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  const payload = await response.json().catch(() => null);
  if (response.status === 401) {
    showAuth("Your session expired. Enter the access code again.");
    throw new AuthenticationRequired();
  }
  if (!response.ok) throw new Error(readableError(payload, `Request failed (${response.status})`));
  return payload;
}

async function pollJob() {
  while (state.jobId) {
    try {
      const job = await getJson(`/transcriptions/${state.jobId}`);
      state.job = job;
      els.jobIdLine.textContent = `JOB ${state.jobId}`;
      if (job.state === "failed") {
        els.jobError.textContent = job.error || "The remote worker stopped before producing a result.";
        showView("error");
        return;
      }
      if (job.state === "completed") {
        updateProcessing("completed", 100);
        await loadResult(job);
        return;
      }
      updateProcessing(job.state, job.progress);
      await new Promise((resolve) => setTimeout(resolve, (job.retry_after_seconds || 2) * 1000));
    } catch (error) {
      if (error instanceof AuthenticationRequired) return;
      els.jobError.textContent = error.message;
      showView("error");
      return;
    }
  }
}

async function loadResult(job) {
  const roll = await getJson(job.links.piano_roll);
  state.notes = roll.notes;
  state.duration = roll.duration || job.result?.audio_seconds || 0;
  els.resultName.textContent = job.source_name;
  els.audio.src = job.links.audio;
  els.midiDownload.href = job.links.midi;
  els.totalTime.textContent = formatTime(state.duration);
  els.metricNotes.textContent = job.result?.note_count ?? state.notes.length;
  els.metricDuration.textContent = formatTime(state.duration);
  const seconds = job.result?.inference?.seconds;
  els.metricInference.textContent = Number.isFinite(seconds) ? `${seconds.toFixed(2)} s` : "—";
  const cost = job.result?.inference?.estimated_gpu_cost_usd;
  els.metricCost.textContent = Number.isFinite(cost) ? `$${cost.toFixed(4)}` : "—";
  buildInstrumentFilters(roll.instruments);
  showView("result");
  resizeCanvas();
  renderRoll(0);
}

function buildInstrumentFilters(instruments) {
  state.enabledInstruments = new Set(instruments);
  state.colors.clear();
  els.filters.replaceChildren();
  instruments.forEach((instrument, index) => {
    const color = palette[index % palette.length];
    state.colors.set(instrument, color);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "instrument-filter";
    button.style.setProperty("--instrument-color", color);
    const marker = document.createElement("i");
    const label = document.createElement("span");
    label.textContent = instrument.replaceAll("_", " ");
    button.append(marker, label);
    button.addEventListener("click", () => {
      if (state.enabledInstruments.has(instrument)) state.enabledInstruments.delete(instrument);
      else state.enabledInstruments.add(instrument);
      button.classList.toggle("off", !state.enabledInstruments.has(instrument));
      renderRoll(currentPlaybackTime());
    });
    els.filters.append(button);
  });
}

function resizeCanvas() {
  const rect = els.canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  els.canvas.width = Math.max(1, Math.round(rect.width * scale));
  els.canvas.height = Math.max(1, Math.round(rect.height * scale));
}

function renderRoll(playheadSeconds = 0) {
  const context = els.canvas.getContext("2d");
  const width = els.canvas.width;
  const height = els.canvas.height;
  const scale = window.devicePixelRatio || 1;
  context.clearRect(0, 0, width, height);
  els.rollEmpty.hidden = state.notes.length > 0;
  if (!state.notes.length || !state.duration) return;

  const pitches = state.notes.map((note) => note.pitch);
  const minPitch = Math.max(0, Math.min(...pitches) - 2);
  const maxPitch = Math.min(127, Math.max(...pitches) + 2);
  const pitchSpan = Math.max(1, maxPitch - minPitch + 1);

  context.lineWidth = scale;
  for (let pitch = minPitch; pitch <= maxPitch; pitch += 1) {
    const y = ((maxPitch - pitch) / pitchSpan) * height;
    const isC = pitch % 12 === 0;
    context.strokeStyle = isC ? "rgba(200,245,96,0.12)" : "rgba(255,255,255,0.035)";
    context.beginPath();
    context.moveTo(0, Math.round(y));
    context.lineTo(width, Math.round(y));
    context.stroke();
  }
  for (let second = 0; second <= state.duration; second += state.duration > 180 ? 15 : 5) {
    const x = (second / state.duration) * width;
    context.strokeStyle = "rgba(255,255,255,0.055)";
    context.beginPath();
    context.moveTo(Math.round(x), 0);
    context.lineTo(Math.round(x), height);
    context.stroke();
  }

  const rowHeight = height / pitchSpan;
  for (const note of state.notes) {
    if (!state.enabledInstruments.has(note.instrument)) continue;
    const x = (note.start / state.duration) * width;
    const noteWidth = Math.max(2 * scale, ((note.end - note.start) / state.duration) * width);
    const y = ((maxPitch - note.pitch) / pitchSpan) * height;
    context.fillStyle = state.colors.get(note.instrument) || palette[0];
    context.globalAlpha = note.start <= playheadSeconds && note.end >= playheadSeconds ? 1 : 0.78;
    context.fillRect(x, y + scale, noteWidth, Math.max(2 * scale, rowHeight - 2 * scale));
  }
  context.globalAlpha = 1;
  const playhead = (playheadSeconds / state.duration) * width;
  context.strokeStyle = "#f2f0e8";
  context.lineWidth = scale;
  context.beginPath();
  context.moveTo(playhead, 0);
  context.lineTo(playhead, height);
  context.stroke();
}

function currentPlaybackTime() {
  if (state.playbackMode === "source") return els.audio.currentTime || 0;
  if (!state.synthContext || !els.play.classList.contains("playing")) return state.synthOffset;
  return Math.min(state.duration, state.synthOffset + state.synthContext.currentTime - state.synthStartAt);
}

function updateTransport(time) {
  const safe = Math.min(state.duration || 0, Math.max(0, time || 0));
  els.currentTime.textContent = formatTime(safe);
  els.timeline.value = state.duration ? Math.round((safe / state.duration) * 1000) : 0;
  renderRoll(safe);
}

function stopSynth(keepOffset = true) {
  for (const node of state.synthNodes) {
    try { node.stop(); } catch (_) { /* already stopped */ }
  }
  state.synthNodes = [];
  state.synthScheduledUntil = 0;
  if (state.synthTimer) cancelAnimationFrame(state.synthTimer);
  state.synthTimer = null;
  if (keepOffset) state.synthOffset = currentPlaybackTime();
  els.play.classList.remove("playing");
}

function scheduleSynthWindow(from, to) {
  const context = state.synthContext;
  for (const note of state.notes) {
    const startsInWindow = note.start >= from && note.start < to;
    const overlapsInitialOffset = from === state.synthOffset && note.start < from && note.end > from;
    if (!state.enabledInstruments.has(note.instrument) || (!startsInWindow && !overlapsInitialOffset)) continue;
    const startAt = state.synthStartAt + Math.max(0, note.start - state.synthOffset);
    const endAt = state.synthStartAt + Math.max(0.04, note.end - state.synthOffset);
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.type = "triangle";
    oscillator.frequency.value = 440 * (2 ** ((note.pitch - 69) / 12));
    gain.gain.setValueAtTime(0, startAt);
    gain.gain.linearRampToValueAtTime(0.035, startAt + 0.01);
    gain.gain.setValueAtTime(0.035, Math.max(startAt + 0.01, endAt - 0.03));
    gain.gain.linearRampToValueAtTime(0, endAt);
    oscillator.connect(gain).connect(context.destination);
    oscillator.start(startAt);
    oscillator.stop(endAt);
    state.synthNodes.push(oscillator);
  }
  state.synthScheduledUntil = to;
}

function scheduleSynth() {
  if (!state.synthContext) state.synthContext = new AudioContext();
  const offset = state.synthOffset >= state.duration ? 0 : state.synthOffset;
  state.synthOffset = offset;
  state.synthStartAt = state.synthContext.currentTime;
  scheduleSynthWindow(offset, Math.min(state.duration, offset + 20));
  els.play.classList.add("playing");
  animateSynth();
}

function animateSynth() {
  const time = currentPlaybackTime();
  updateTransport(time);
  if (time >= state.duration) {
    stopSynth(false);
    state.synthOffset = 0;
    updateTransport(0);
    return;
  }
  if (time + 5 >= state.synthScheduledUntil && state.synthScheduledUntil < state.duration) {
    scheduleSynthWindow(state.synthScheduledUntil, Math.min(state.duration, state.synthScheduledUntil + 20));
  }
  state.synthTimer = requestAnimationFrame(animateSynth);
}

function togglePlayback() {
  if (state.playbackMode === "source") {
    if (els.audio.paused) els.audio.play();
    else els.audio.pause();
    return;
  }
  if (els.play.classList.contains("playing")) stopSynth();
  else scheduleSynth();
}

function setPlaybackMode(mode) {
  if (mode === state.playbackMode) return;
  if (state.playbackMode === "source") els.audio.pause();
  else stopSynth();
  state.playbackMode = mode;
  document.querySelectorAll(".source-switch button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  const time = Number(els.timeline.value) / 1000 * state.duration;
  if (mode === "source") els.audio.currentTime = time;
  else state.synthOffset = time;
  updateTransport(time);
}

async function handleLogin(event) {
  event.preventDefault();
  const button = els.authForm.querySelector("button");
  els.authError.hidden = true;
  button.disabled = true;
  button.textContent = "Checking…";
  try {
    const response = await fetch("/auth/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ access_token: els.accessToken.value }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(readableError(payload, "Could not unlock this workspace"));
    els.accessToken.value = "";
    showWorkspace();
    if (state.jobId) {
      updateProcessing("submitted", 10, "Loading durable job");
      await pollJob();
    } else showView("empty");
  } catch (error) {
    els.authError.textContent = error.message;
    els.authError.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Unlock";
  }
}

async function handleLogout() {
  els.audio.pause();
  stopSynth(false);
  await fetch("/auth/session", { method: "DELETE" }).catch(() => null);
  showAuth("Signed out.");
}

async function initialize() {
  const jobMatch = window.location.pathname.match(/^\/jobs\/([0-9a-f]{32})$/);
  if (jobMatch) state.jobId = jobMatch[1];
  try {
    const response = await fetch("/auth/session", { headers: { Accept: "application/json" } });
    const session = await response.json();
    if (!session.authenticated) {
      showAuth();
      return;
    }
    showWorkspace();
    if (state.jobId) {
      els.jobIdLine.textContent = `JOB ${state.jobId}`;
      updateProcessing("submitted", 10, "Loading durable job");
      await pollJob();
    } else showView("empty");
  } catch (_) {
    showAuth("Could not verify the session. Try again.");
  }
}

els.authForm.addEventListener("submit", handleLogin);
els.logout.addEventListener("click", handleLogout);
els.form.addEventListener("submit", handleSubmit);
els.file.addEventListener("change", () => setSelectedFile(els.file.files[0]));
for (const eventName of ["dragenter", "dragover"]) {
  els.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropZone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  els.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropZone.classList.remove("dragging");
  });
}
els.dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  els.file.files = transfer.files;
  setSelectedFile(file);
});
els.play.addEventListener("click", togglePlayback);
els.audio.addEventListener("play", () => els.play.classList.add("playing"));
els.audio.addEventListener("pause", () => els.play.classList.remove("playing"));
els.audio.addEventListener("timeupdate", () => {
  if (state.playbackMode === "source") updateTransport(els.audio.currentTime);
});
els.audio.addEventListener("ended", () => updateTransport(0));
els.timeline.addEventListener("input", () => {
  const time = Number(els.timeline.value) / 1000 * state.duration;
  if (state.playbackMode === "source") els.audio.currentTime = time;
  else {
    const wasPlaying = els.play.classList.contains("playing");
    stopSynth(false);
    state.synthOffset = time;
    if (wasPlaying) scheduleSynth();
  }
  updateTransport(time);
});
document.querySelectorAll(".source-switch button").forEach((button) => {
  button.addEventListener("click", () => setPlaybackMode(button.dataset.mode));
});
window.addEventListener("resize", () => {
  resizeCanvas();
  renderRoll(currentPlaybackTime());
});

initialize();
