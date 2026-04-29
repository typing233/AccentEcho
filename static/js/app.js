/* ================================================================
   AccentEcho – frontend application
   ================================================================ */

'use strict';

// ── State ────────────────────────────────────────────────────
let refBlob       = null;   // WAV blob for reference audio
let refId         = null;   // server-assigned reference ID
let audioId       = null;   // server-assigned output audio ID
let audioUrl      = null;   // object URL for the output WAV
let audioEl       = null;   // <audio> element for synthesised playback
let decodedBuffer = null;   // AudioBuffer of synthesised audio (for waveform)

// Reference characteristics (for radar chart)
let refCharacteristics = null;

// Recording state
let mediaRecorder    = null;
let recordedChunks   = [];
let isRecording      = false;
let recordTimerInt   = null;
let recordSeconds    = 0;
let analyserNode     = null;
let recordAnimFrame  = null;
let recordStream     = null;

// Repeat / Read Aloud recording state
let repeatRecorder      = null;
let repeatChunks        = [];
let isRepeatRecording   = false;
let repeatTimerInt      = null;
let repeatSeconds       = 0;
let repeatAnalyser      = null;
let repeatAnimFrame     = null;
let repeatStream        = null;

// AudioContext (shared)
let audioCtx = null;
function getAudioCtx() {
  if (!audioCtx || audioCtx.state === 'closed') {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return audioCtx;
}

// ── DOM helpers ──────────────────────────────────────────────
const $ = id => document.getElementById(id);

function showToast(msg, type = '') {
  const t = $('toast');
  t.textContent = msg;
  t.className = 'toast' + (type ? ' ' + type : '');
  t.hidden = false;
  t.classList.add('show');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => {
    t.classList.remove('show');
    setTimeout(() => { t.hidden = true; }, 350);
  }, 3200);
}

function setVisible(el, visible) {
  if (typeof el === 'string') el = $(el);
  el.hidden = !visible;
}

// ── WAV encoder ──────────────────────────────────────────────
/**
 * Encode an AudioBuffer (mono, down-mixed) into a WAV Blob.
 */
function encodeWAV(audioBuffer) {
  // Down-mix all channels to mono synchronously from raw channel data
  const SR    = audioBuffer.sampleRate;
  const numCh = audioBuffer.numberOfChannels;
  const len   = audioBuffer.length;
  const mono  = new Float32Array(len);

  for (let c = 0; c < numCh; c++) {
    const data = audioBuffer.getChannelData(c);
    for (let i = 0; i < len; i++) mono[i] += data[i];
  }
  if (numCh > 1) for (let i = 0; i < len; i++) mono[i] /= numCh;

  const byteLen = len * 2;
  const buf     = new ArrayBuffer(44 + byteLen);
  const view    = new DataView(buf);

  function ws(off, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  }

  ws(0, 'RIFF');
  view.setUint32(4, 36 + byteLen, true);
  ws(8, 'WAVE');
  ws(12, 'fmt ');
  view.setUint32(16, 16, true);      // chunk size
  view.setUint16(20, 1,  true);      // PCM
  view.setUint16(22, 1,  true);      // 1 channel
  view.setUint32(24, SR, true);
  view.setUint32(28, SR * 2, true);  // byte rate
  view.setUint16(32, 2,  true);      // block align
  view.setUint16(34, 16, true);      // bits/sample
  ws(36, 'data');
  view.setUint32(40, byteLen, true);

  let off2 = 44;
  for (let i = 0; i < len; i++) {
    const s = Math.max(-1, Math.min(1, mono[i]));
    view.setInt16(off2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off2 += 2;
  }
  return new Blob([buf], { type: 'audio/wav' });
}

/**
 * Decode any audio blob → WAV blob via the AudioContext.
 */
async function blobToWAV(blob) {
  const ctx       = getAudioCtx();
  const arrBuf    = await blob.arrayBuffer();
  const audioBuf  = await ctx.decodeAudioData(arrBuf);
  return encodeWAV(audioBuf);
}

// ── Waveform drawing ─────────────────────────────────────────
function drawWaveform(canvas, audioBuffer, playFrac = 0, color = '#6366f1', playColor = '#4f46e5') {
  const W   = canvas.offsetWidth || 600;
  const H   = canvas.offsetHeight || 60;
  canvas.width  = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const data   = audioBuffer.getChannelData(0);
  const step   = Math.ceil(data.length / W);
  const amp    = H / 2;
  const playX  = Math.round(playFrac * W);

  ctx.lineWidth = 1.5;
  for (let x = 0; x < W; x++) {
    let mn = 0, mx = 0;
    for (let s = x * step; s < Math.min((x + 1) * step, data.length); s++) {
      if (data[s] < mn) mn = data[s];
      if (data[s] > mx) mx = data[s];
    }
    ctx.strokeStyle = x < playX ? playColor : color;
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    ctx.moveTo(x + .5, amp + mn * amp * 0.9);
    ctx.lineTo(x + .5, amp + mx * amp * 0.9);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function drawLiveWave(canvas, analyser) {
  const W  = canvas.offsetWidth || 600;
  const H  = canvas.offsetHeight || 64;
  canvas.width  = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  const buf = new Uint8Array(analyser.fftSize);

  function loop() {
    recordAnimFrame = requestAnimationFrame(loop);
    analyser.getByteTimeDomainData(buf);
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const sliceW = W / buf.length;
    let x = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = buf[i] / 128 - 1;
      const y = (v * H / 2) + H / 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      x += sliceW;
    }
    ctx.stroke();
  }
  loop();
}

// ── Upload tab ───────────────────────────────────────────────
function setupUploadTab() {
  const zone  = $('upload-zone');
  const input = $('file-input');

  zone.addEventListener('click', () => input.click());

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleAudioFile(file);
  });

  input.addEventListener('change', () => {
    if (input.files[0]) handleAudioFile(input.files[0]);
    input.value = '';
  });
}

async function handleAudioFile(file) {
  if (!file.type.startsWith('audio/') && !/\.(wav|mp3|ogg|m4a|flac|aac)$/i.test(file.name)) {
    showToast('请选择音频文件', 'error');
    return;
  }
  showToast('正在处理文件…');
  try {
    const wavBlob = await blobToWAV(file);
    setReferenceAudio(wavBlob, URL.createObjectURL(file));
  } catch (e) {
    showToast('无法解码该音频文件，请尝试其他格式', 'error');
  }
}

// ── Record tab ───────────────────────────────────────────────
function setupRecordTab() {
  $('btn-record').addEventListener('click', toggleRecording);
}

async function toggleRecording() {
  if (!isRecording) {
    await startRecording();
  } else {
    stopRecording();
  }
}

async function startRecording() {
  try {
    recordStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch {
    showToast('无法访问麦克风，请检查权限', 'error');
    return;
  }

  const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg']
    .find(t => MediaRecorder.isTypeSupported(t)) || '';
  if (!mimeType) {
    console.warn('AccentEcho: no preferred MIME type supported; using browser default codec');
  }

  mediaRecorder  = new MediaRecorder(recordStream, mimeType ? { mimeType } : undefined);
  recordedChunks = [];
  mediaRecorder.ondataavailable = e => { if (e.data.size) recordedChunks.push(e.data); };
  mediaRecorder.onstop = finaliseRecording;
  mediaRecorder.start(100);

  // Live waveform
  const ctx      = getAudioCtx();
  const source   = ctx.createMediaStreamSource(recordStream);
  analyserNode   = ctx.createAnalyser();
  analyserNode.fftSize = 512;
  source.connect(analyserNode);
  drawLiveWave($('canvas-record'), analyserNode);

  // Timer
  recordSeconds = 0;
  updateRecordTimer();
  recordTimerInt = setInterval(() => {
    recordSeconds++;
    updateRecordTimer();
    if (recordSeconds >= 60) stopRecording();
  }, 1000);

  isRecording = true;
  $('btn-record').textContent = '⏹ 停止录制';
  $('btn-record').classList.add('recording');
}

function stopRecording() {
  if (!isRecording) return;
  mediaRecorder.stop();
  recordStream.getTracks().forEach(t => t.stop());
  cancelAnimationFrame(recordAnimFrame);
  clearInterval(recordTimerInt);
  isRecording = false;
  $('btn-record').textContent = '🎤 开始录制';
  $('btn-record').classList.remove('recording');
}

async function finaliseRecording() {
  const raw  = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
  showToast('正在处理录音…');
  try {
    const wav = await blobToWAV(raw);
    setReferenceAudio(wav, URL.createObjectURL(raw));
  } catch (e) {
    showToast('录音处理失败，请重试', 'error');
  }
}

function updateRecordTimer() {
  const m = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
  const s = String(recordSeconds % 60).padStart(2, '0');
  $('record-timer').textContent = `${m}:${s}`;
}

// ── Set reference audio ──────────────────────────────────────
async function setReferenceAudio(wavBlob, previewUrl) {
  refBlob = wavBlob;
  refId   = null;

  // Show preview
  $('audio-ref').src = previewUrl;
  setVisible('ref-preview', true);
  setVisible('panel-upload', false);
  setVisible('panel-record', false);
  setVisible('tab-upload-btn', true);
  setVisible('tab-record-btn', true);

  // Draw static waveform from the WAV blob
  try {
    const arrBuf = await wavBlob.arrayBuffer();
    const audioBuf = await getAudioCtx().decodeAudioData(arrBuf.slice(0));
    drawWaveform($('canvas-ref-waveform'), audioBuf, 0, '#8b5cf6');
  } catch { /* non-critical */ }

  // Auto-analyse
  $('tag-reference').textContent = '上传中…';
  $('analysis-msg').textContent  = '正在分析音频特征…';
  setVisible('analysis-status', true);
  setVisible('analysis-results', false);

  await analyseReference();
}

// ── Analyse reference ────────────────────────────────────────
async function analyseReference() {
  const form = new FormData();
  form.append('audio', refBlob, 'reference.wav');

  try {
    const res  = await fetch('/analyze', { method: 'POST', body: form });
    const json = await res.json();

    if (!res.ok || json.error) {
      $('analysis-msg').textContent = '分析失败：' + (json.error || res.statusText);
      $('tag-reference').textContent = '分析失败';
      showToast('音频分析失败', 'error');
      return;
    }

    refId = json.ref_id;
    showAnalysisResults(json.characteristics);
    $('tag-reference').textContent = '✓ 分析完成';
    $('analysis-msg').textContent  = '特征提取完成';
    updateSynthButton();
  } catch (e) {
    $('analysis-msg').textContent = '网络错误，请重试';
    $('tag-reference').textContent = '错误';
    showToast('服务器连接失败', 'error');
  }
}

function pitchLabel(hz) {
  if (hz < 130) return '低沉';
  if (hz < 200) return '中性';
  return '高亢';
}

function showAnalysisResults(c) {
  const grid = $('analysis-results');
  grid.innerHTML = '';

  const items = [
    { val: c.mean_pitch.toFixed(0) + ' Hz', lbl: '平均音调 · ' + pitchLabel(c.mean_pitch) },
    { val: c.pitch_range.toFixed(0) + ' Hz', lbl: '音调范围' },
    { val: c.speaking_rate.toFixed(1) + '/s', lbl: '语速（音节/秒）' },
    { val: c.duration.toFixed(1) + ' s', lbl: '音频时长' },
  ];

  items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'analysis-card';
    div.innerHTML = `<div class="val">${item.val}</div><div class="lbl">${item.lbl}</div>`;
    grid.appendChild(div);
  });

  setVisible('analysis-results', true);
}

// ── Text area ────────────────────────────────────────────────
function setupTextInput() {
  const ta = $('practice-text');
  ta.addEventListener('input', () => {
    $('char-count').textContent = ta.value.length;
    updateSynthButton();
  });
}

// ── Synthesize button state ──────────────────────────────────
function updateSynthButton() {
  const hasText = $('practice-text').value.trim().length > 0;
  $('btn-synthesize').disabled = !(refId && hasText);
}

// ── Reselect reference ───────────────────────────────────────
function setupReselect() {
  $('btn-reselect').addEventListener('click', () => {
    setVisible('ref-preview', false);
    refId   = null;
    refBlob = null;
    $('tag-reference').textContent = '';
    $('analysis-results').innerHTML = '';
    setVisible('analysis-results', false);
    // Go back to upload panel by default
    activateTab('upload');
    updateSynthButton();
  });
}

// ── Tab switching ─────────────────────────────────────────────
function activateTab(name) {
  const isUpload = name === 'upload';
  setVisible('panel-upload', isUpload);
  setVisible('panel-record', !isUpload);
  setVisible('ref-preview', false);

  $('tab-upload-btn').classList.toggle('active', isUpload);
  $('tab-record-btn').classList.toggle('active', !isUpload);
}

function setupTabs() {
  $('tab-upload-btn').addEventListener('click', () => activateTab('upload'));
  $('tab-record-btn').addEventListener('click', () => activateTab('record'));
}

// ── Synthesis ────────────────────────────────────────────────
function setupSynthesisButton() {
  $('btn-synthesize').addEventListener('click', runSynthesis);
}

async function runSynthesis() {
  const text = $('practice-text').value.trim();
  if (!text) { showToast('请输入练习文字', 'error'); return; }

  $('btn-synthesize').disabled = true;
  setVisible('synth-status', true);
  $('synth-msg').textContent = '正在分析口音特征并合成语音，请稍候（约20–40秒）…';
  setVisible('card-playback', false);

  try {
    const res  = await fetch('/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, ref_id: refId }),
    });
    const json = await res.json();

    if (!res.ok || json.error) {
      showToast('合成失败：' + (json.error || res.statusText), 'error');
      $('synth-msg').textContent = '合成失败：' + (json.error || res.statusText);
      return;
    }

    audioId  = json.audio_id;
    audioUrl = `/audio/${audioId}`;
    await setupPlayback(audioUrl);
    setVisible('synth-status', false);
    showToast('合成完成！', 'success');
  } catch (e) {
    showToast('网络错误，请重试', 'error');
    $('synth-msg').textContent = '网络错误，请重试';
  } finally {
    updateSynthButton();
  }
}

// ── Playback ─────────────────────────────────────────────────
let playbackAnimFrame = null;

async function setupPlayback(url) {
  // Tear down previous instance
  if (audioEl) { audioEl.pause(); audioEl.src = ''; audioEl = null; }
  cancelAnimationFrame(playbackAnimFrame);

  audioEl       = new Audio();
  audioEl.src   = url;
  audioEl.preload = 'auto';
  audioEl.loop  = $('loop-toggle').checked;

  // Decode for waveform drawing
  try {
    const resp    = await fetch(url);
    const arrBuf  = await resp.arrayBuffer();
    decodedBuffer = await getAudioCtx().decodeAudioData(arrBuf);
    drawWaveform($('canvas-playback'), decodedBuffer, 0);
  } catch { decodedBuffer = null; }

  // Time display
  audioEl.addEventListener('loadedmetadata', () => {
    $('time-total').textContent = formatTime(audioEl.duration);
  });

  audioEl.addEventListener('timeupdate', () => {
    const frac = audioEl.duration ? audioEl.currentTime / audioEl.duration : 0;
    $('time-thumb').style.width = (frac * 100) + '%';
    $('time-current').textContent = formatTime(audioEl.currentTime);
    if (decodedBuffer) drawWaveform($('canvas-playback'), decodedBuffer, frac);
  });

  audioEl.addEventListener('ended', () => {
    $('btn-play-pause').textContent = '▶';
    if (!audioEl.loop) {
      $('time-thumb').style.width = '100%';
    }
  });

  // Seek by clicking time track
  $('time-track').addEventListener('click', e => {
    if (!audioEl.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    audioEl.currentTime = frac * audioEl.duration;
  });

  // Waveform click for seek
  $('canvas-playback').addEventListener('click', e => {
    if (!audioEl.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    audioEl.currentTime = frac * audioEl.duration;
  });

  setVisible('card-playback', true);
  $('btn-play-pause').textContent = '▶';

  // Scroll into view
  setTimeout(() => $('card-playback').scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 100);
}

function setupPlaybackControls() {
  $('btn-play-pause').addEventListener('click', togglePlayback);

  $('speed-slider').addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    $('speed-value').textContent = formatSpeed(v);
    if (audioEl) audioEl.playbackRate = v;
  });

  $('loop-toggle').addEventListener('change', e => {
    if (audioEl) audioEl.loop = e.target.checked;
  });

  $('btn-download').addEventListener('click', () => {
    if (!audioId) return;
    const a = document.createElement('a');
    a.href = `/download/${audioId}`;
    a.download = 'accent_echo.wav';
    a.click();
  });
}

function togglePlayback() {
  if (!audioEl) return;
  if (audioEl.paused) {
    audioEl.playbackRate = parseFloat($('speed-slider').value);
    audioEl.play().catch(() => showToast('播放失败', 'error'));
    $('btn-play-pause').textContent = '⏸';
  } else {
    audioEl.pause();
    $('btn-play-pause').textContent = '▶';
  }
}

// ── Utilities ────────────────────────────────────────────────
function formatTime(sec) {
  if (!isFinite(sec)) return '0:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

/** Format a playback-rate number as a human-readable speed label, e.g. "1.0×". */
function formatSpeed(v) {
  // Show one decimal place; trim unnecessary trailing zero for whole numbers
  const s = v.toFixed(1);
  return (s.endsWith('.0') ? s.slice(0, -2) : s) + '×';
}

// ── Reference Characteristics Storage ────────────────────────
function showAnalysisResults(c) {
  refCharacteristics = c;
  const grid = $('analysis-results');
  grid.innerHTML = '';

  const items = [
    { val: c.mean_pitch.toFixed(0) + ' Hz', lbl: '平均音调 · ' + pitchLabel(c.mean_pitch) },
    { val: c.pitch_range.toFixed(0) + ' Hz', lbl: '音调范围' },
    { val: c.speaking_rate.toFixed(1) + '/s', lbl: '语速（音节/秒）' },
    { val: c.duration.toFixed(1) + ' s', lbl: '音频时长' },
  ];

  items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'analysis-card';
    div.innerHTML = `<div class="val">${item.val}</div><div class="lbl">${item.lbl}</div>`;
    grid.appendChild(div);
  });

  setVisible('analysis-results', true);
}


// ── Radar Chart ──────────────────────────────────────────────
function normalizeForRadar(chars) {
  return {
    pitch: Math.min(100, Math.max(0, (chars.mean_pitch - 80) / 2.5)),
    pitchRange: Math.min(100, Math.max(0, chars.pitch_range / 2)),
    speakingRate: Math.min(100, Math.max(0, chars.speaking_rate * 15)),
    volume: Math.min(100, Math.max(0, chars.rms_energy * 8000)),
    brightness: Math.min(100, Math.max(0, (chars.spectral_centroid - 800) / 15))
  };
}

function drawRadarChart(canvas, refData, synthData) {
  const W = Math.min(400, canvas.offsetWidth || 300);
  const H = W;
  canvas.width = W * 2;
  canvas.height = H * 2;
  const ctx = canvas.getContext('2d');
  ctx.scale(2, 2);

  const centerX = W / 2;
  const centerY = H / 2;
  const radius = Math.min(W, H) * 0.38;

  const labels = ['音调', '音域', '语速', '音量', '音色亮度'];
  const numAxes = labels.length;
  const angleStep = (Math.PI * 2) / numAxes;

  ctx.clearRect(0, 0, W, H);

  const levels = 5;
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  for (let i = 1; i <= levels; i++) {
    const r = (radius / levels) * i;
    ctx.beginPath();
    for (let j = 0; j <= numAxes; j++) {
      const angle = angleStep * j - Math.PI / 2;
      const x = centerX + r * Math.cos(angle);
      const y = centerY + r * Math.sin(angle);
      if (j === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
  }

  ctx.strokeStyle = '#cbd5e1';
  for (let i = 0; i < numAxes; i++) {
    const angle = angleStep * i - Math.PI / 2;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(
      centerX + radius * Math.cos(angle),
      centerY + radius * Math.sin(angle)
    );
    ctx.stroke();
  }

  ctx.fillStyle = '#64748b';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < numAxes; i++) {
    const angle = angleStep * i - Math.PI / 2;
    const labelRadius = radius + 20;
    const x = centerX + labelRadius * Math.cos(angle);
    const y = centerY + labelRadius * Math.sin(angle);
    ctx.fillText(labels[i], x, y);
  }

  function drawData(data, color, fillColor) {
    ctx.beginPath();
    for (let i = 0; i <= numAxes; i++) {
      const idx = i % numAxes;
      const angle = angleStep * idx - Math.PI / 2;
      const keys = ['pitch', 'pitchRange', 'speakingRate', 'volume', 'brightness'];
      const value = data[keys[idx]] || 0;
      const r = (value / 100) * radius;
      const x = centerX + r * Math.cos(angle);
      const y = centerY + r * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    for (let i = 0; i < numAxes; i++) {
      const angle = angleStep * i - Math.PI / 2;
      const keys = ['pitch', 'pitchRange', 'speakingRate', 'volume', 'brightness'];
      const value = data[keys[i]] || 0;
      const r = (value / 100) * radius;
      const x = centerX + r * Math.cos(angle);
      const y = centerY + r * Math.sin(angle);
      
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
  }

  if (refData) {
    drawData(normalizeForRadar(refData), '#8b5cf6', 'rgba(139, 92, 246, 0.15)');
  }
  if (synthData) {
    drawData(normalizeForRadar(synthData), '#06b6d4', 'rgba(6, 182, 212, 0.15)');
  }
}


// ── Line Chart for History ───────────────────────────────────
function drawLineChart(canvas, data) {
  const W = canvas.offsetWidth || 600;
  const H = canvas.offsetHeight || 200;
  canvas.width = W * 2;
  canvas.height = H * 2;
  const ctx = canvas.getContext('2d');
  ctx.scale(2, 2);

  const padding = { top: 20, right: 20, bottom: 40, left: 50 };
  const chartW = W - padding.left - padding.right;
  const chartH = H - padding.top - padding.bottom;

  ctx.clearRect(0, 0, W, H);

  if (!data || data.length < 1) {
    ctx.fillStyle = '#64748b';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('暂无练习记录', W / 2, H / 2);
    return;
  }

  const minScore = Math.max(0, Math.min(...data.map(d => d.score)) - 10);
  const maxScore = Math.min(100, Math.max(...data.map(d => d.score)) + 10);
  const scoreRange = maxScore - minScore || 100;

  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  const yLabels = 5;
  for (let i = 0; i <= yLabels; i++) {
    const y = padding.top + (chartH / yLabels) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(W - padding.right, y);
    ctx.stroke();

    const score = Math.round(maxScore - (scoreRange / yLabels) * i);
    ctx.fillStyle = '#64748b';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(score + '%', padding.left - 8, y + 4);
  }

  const points = data.map((d, i) => {
    const x = padding.left + (chartW / (data.length - 1 || 1)) * i;
    const y = padding.top + chartH - ((d.score - minScore) / scoreRange) * chartH;
    return { x, y, data: d };
  });

  if (points.length > 1) {
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.lineTo(points[points.length - 1].x, padding.top + chartH);
    ctx.lineTo(points[0].x, padding.top + chartH);
    ctx.closePath();
    ctx.fillStyle = 'rgba(99, 102, 241, 0.1)';
    ctx.fill();
  }

  points.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = p.data.score >= 80 ? '#10b981' : p.data.score >= 60 ? '#f59e0b' : '#ef4444';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    const date = new Date(p.data.timestamp);
    const label = (i + 1) + '#' + date.getMonth().toString().padStart(2, '0') + (date.getDate()).toString().padStart(2, '0');
    ctx.fillStyle = '#64748b';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, p.x, H - padding.bottom + 20);
  });

  ctx.fillStyle = '#1e293b';
  ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('练习次数', W / 2, H - 10);
}


// ── Practice History (localStorage) ──────────────────────────
const HISTORY_KEY = 'accent_echo_history';

function loadHistory() {
  try {
    const stored = localStorage.getItem(HISTORY_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function saveHistory(history) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch (e) {
    console.warn('Failed to save history:', e);
  }
}

function addHistoryEntry(score, pitchDev, rateDev) {
  const history = loadHistory();
  history.push({
    score: score,
    pitch_deviation: pitchDev,
    rate_deviation: rateDev,
    timestamp: Date.now(),
    text: $('practice-text').value.substring(0, 50)
  });
  if (history.length > 50) history.shift();
  saveHistory(history);
  return history;
}

function clearHistory() {
  localStorage.removeItem(HISTORY_KEY);
}


// ── Repeat / Read Aloud Recording ────────────────────────────
function setupRepeatButton() {
  $('btn-repeat-record').addEventListener('click', toggleRepeatRecording);
}

async function toggleRepeatRecording() {
  if (!isRepeatRecording) {
    await startRepeatRecording();
  } else {
    stopRepeatRecording();
  }
}

async function startRepeatRecording() {
  try {
    repeatStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch {
    showToast('无法访问麦克风，请检查权限', 'error');
    return;
  }

  const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg']
    .find(t => MediaRecorder.isTypeSupported(t)) || '';

  repeatRecorder = new MediaRecorder(repeatStream, mimeType ? { mimeType } : undefined);
  repeatChunks = [];
  repeatRecorder.ondataavailable = e => { if (e.data.size) repeatChunks.push(e.data); };
  repeatRecorder.onstop = finaliseRepeatRecording;
  repeatRecorder.start(100);

  const ctx = getAudioCtx();
  const source = ctx.createMediaStreamSource(repeatStream);
  repeatAnalyser = ctx.createAnalyser();
  repeatAnalyser.fftSize = 512;
  source.connect(repeatAnalyser);
  
  setVisible('canvas-repeat-wave', true);
  drawLiveRepeatWave($('canvas-repeat-wave'), repeatAnalyser);

  repeatSeconds = 0;
  updateRepeatTimer();
  repeatTimerInt = setInterval(() => {
    repeatSeconds++;
    updateRepeatTimer();
    if (repeatSeconds >= 30) stopRepeatRecording();
  }, 1000);

  isRepeatRecording = true;
  $('btn-repeat-record').textContent = '⏹ 停止录制';
  $('btn-repeat-record').classList.add('recording');
}

function drawLiveRepeatWave(canvas, analyser) {
  const W = canvas.offsetWidth || 600;
  const H = canvas.offsetHeight || 60;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  const buf = new Uint8Array(analyser.fftSize);

  function loop() {
    if (!isRepeatRecording) return;
    repeatAnimFrame = requestAnimationFrame(loop);
    analyser.getByteTimeDomainData(buf);
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const sliceW = W / buf.length;
    let x = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = buf[i] / 128 - 1;
      const y = (v * H / 2) + H / 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      x += sliceW;
    }
    ctx.stroke();
  }
  loop();
}

function stopRepeatRecording() {
  if (!isRepeatRecording) return;
  repeatRecorder.stop();
  repeatStream.getTracks().forEach(t => t.stop());
  cancelAnimationFrame(repeatAnimFrame);
  clearInterval(repeatTimerInt);
  isRepeatRecording = false;
  $('btn-repeat-record').textContent = '🎤 开始跟读';
  $('btn-repeat-record').classList.remove('recording');
}

function updateRepeatTimer() {
  const m = String(Math.floor(repeatSeconds / 60)).padStart(2, '0');
  const s = String(repeatSeconds % 60).padStart(2, '0');
  $('repeat-timer').textContent = `${m}:${s}`;
}

async function finaliseRepeatRecording() {
  const raw = new Blob(repeatChunks, { type: repeatRecorder.mimeType || 'audio/webm' });
  showToast('正在处理跟读录音…');
  
  try {
    const wav = await blobToWAV(raw);
    await submitRepeatRecording(wav);
  } catch (e) {
    showToast('录音处理失败，请重试', 'error');
  }
}

async function submitRepeatRecording(wavBlob) {
  if (!audioId) {
    showToast('请先合成语音', 'error');
    return;
  }

  setVisible('compare-status', true);
  setVisible('score-display', false);

  const form = new FormData();
  form.append('user_audio', wavBlob, 'user_recording.wav');
  form.append('audio_id', audioId);

  try {
    const res = await fetch('/compare', { method: 'POST', body: form });
    const json = await res.json();

    if (!res.ok || json.error) {
      showToast('评分失败：' + (json.error || res.statusText), 'error');
      return;
    }

    showScoreResult(json.result);
    
    const history = addHistoryEntry(
      json.result.similarity_score,
      json.result.pitch_deviation_percent,
      json.result.rate_deviation_percent
    );
    
    refreshHistoryChart();
    showToast('评分完成！', 'success');

  } catch (e) {
    showToast('网络错误，请重试', 'error');
  } finally {
    setVisible('compare-status', false);
  }
}

function showScoreResult(result) {
  const scoreEl = $('score-value');
  scoreEl.textContent = result.similarity_score;
  
  scoreEl.classList.remove('good', 'ok', 'needs-work');
  if (result.similarity_score >= 80) scoreEl.classList.add('good');
  else if (result.similarity_score >= 60) scoreEl.classList.add('ok');
  else scoreEl.classList.add('needs-work');

  const pitchEl = $('pitch-deviation');
  const rateEl = $('rate-deviation');
  
  pitchEl.textContent = (result.pitch_deviation_percent > 0 ? '+' : '') + result.pitch_deviation_percent + '%';
  rateEl.textContent = (result.rate_deviation_percent > 0 ? '+' : '') + result.rate_deviation_percent + '%';
  
  pitchEl.classList.remove('positive', 'negative');
  rateEl.classList.remove('positive', 'negative');
  
  if (Math.abs(result.pitch_deviation_percent) < 10) pitchEl.classList.add('positive');
  else pitchEl.classList.add('negative');
  
  if (Math.abs(result.rate_deviation_percent) < 10) rateEl.classList.add('positive');
  else rateEl.classList.add('negative');

  setVisible('score-display', true);
}


// ── Radar Chart Integration ──────────────────────────────────
async function loadAndDrawRadar() {
  if (!audioId || !refCharacteristics) {
    setVisible('card-radar', false);
    return;
  }

  try {
    const res = await fetch('/analyze-output/' + audioId);
    const json = await res.json();
    
    if (res.ok && json.characteristics) {
      drawRadarChart($('canvas-radar'), refCharacteristics, json.characteristics);
      setVisible('card-radar', true);
    }
  } catch (e) {
    console.warn('Failed to load synth characteristics:', e);
  }
}


// ── History Chart Integration ────────────────────────────────
function refreshHistoryChart() {
  const history = loadHistory();
  if (history.length > 0) {
    setVisible('history-hint', false);
    setVisible('card-history', true);
    setTimeout(() => drawLineChart($('canvas-history'), history), 50);
  } else {
    setVisible('card-history', false);
  }
}

function setupClearHistory() {
  $('btn-clear-history').addEventListener('click', () => {
    if (confirm('确定要清空所有练习记录吗？')) {
      clearHistory();
      setVisible('card-history', false);
      showToast('练习记录已清空', 'success');
    }
  });
}


// ── Modified setupPlayback to show repeat section ────────────
const originalSetupPlayback = setupPlayback;
async function setupPlayback(url) {
  await originalSetupPlayback(url);
  setVisible('repeat-section', true);
  setVisible('score-display', false);
  setVisible('canvas-repeat-wave', false);
  
  await loadAndDrawRadar();
  refreshHistoryChart();
  
  setTimeout(() => {
    $('repeat-section').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 200);
}


// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupUploadTab();
  setupRecordTab();
  setupTextInput();
  setupReselect();
  setupSynthesisButton();
  setupPlaybackControls();
  setupRepeatButton();
  setupClearHistory();
  
  refreshHistoryChart();
});
