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
let refPreviewUrl = null;   // current object URL used for the reference audio preview

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
  view.setUint32(16, 16, true);
  view.setUint16(20, 1,  true);
  view.setUint16(22, 1,  true);
  view.setUint32(24, SR, true);
  view.setUint32(28, SR * 2, true);
  view.setUint16(32, 2,  true);
  view.setUint16(34, 16, true);
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

  const ctx      = getAudioCtx();
  const source   = ctx.createMediaStreamSource(recordStream);
  analyserNode   = ctx.createAnalyser();
  analyserNode.fftSize = 512;
  source.connect(analyserNode);
  drawLiveWave($('canvas-record'), analyserNode);

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

  // Clear the element src before revoking the old URL so the browser releases its
  // hold on the resource before we invalidate the URL.
  $('audio-ref').src = '';
  if (refPreviewUrl) {
    URL.revokeObjectURL(refPreviewUrl);
  }
  refPreviewUrl = previewUrl;

  $('audio-ref').src = previewUrl;
  setVisible('ref-preview', true);
  setVisible('panel-upload', false);
  setVisible('panel-record', false);
  setVisible('tab-upload-btn', true);
  setVisible('tab-record-btn', true);

  try {
    const arrBuf = await wavBlob.arrayBuffer();
    const audioBuf = await getAudioCtx().decodeAudioData(arrBuf.slice(0));
    drawWaveform($('canvas-ref-waveform'), audioBuf, 0, '#8b5cf6');
  } catch { /* non-critical */ }

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
    $('audio-ref').src = '';
    if (refPreviewUrl) {
      URL.revokeObjectURL(refPreviewUrl);
      refPreviewUrl = null;
    }
    $('tag-reference').textContent = '';
    $('analysis-results').innerHTML = '';
    setVisible('analysis-results', false);
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
  $('synth-msg').textContent = '正在分析口音特征并合成语音，请稍候…';
  setVisible('card-playback', false);
  setVisible('repeat-section', false);
  setVisible('card-radar', false);

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
  if (audioEl) { audioEl.pause(); audioEl.src = ''; audioEl = null; }
  cancelAnimationFrame(playbackAnimFrame);

  audioEl       = new Audio();
  audioEl.src   = url;
  audioEl.preload = 'auto';
  audioEl.loop  = $('loop-toggle').checked;

  try {
    const resp    = await fetch(url);
    const arrBuf  = await resp.arrayBuffer();
    decodedBuffer = await getAudioCtx().decodeAudioData(arrBuf);
    drawWaveform($('canvas-playback'), decodedBuffer, 0);
  } catch { decodedBuffer = null; }

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

  setVisible('card-playback', true);
  $('btn-play-pause').textContent = '▶';

  setVisible('repeat-section', true);
  setVisible('score-display', false);
  setVisible('canvas-repeat-wave', false);
  setVisible('compare-status', false);

  await loadAndDrawRadar();
  refreshHistoryChart();

  setTimeout(() => $('card-playback').scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 100);
}

function _seekToFrac(frac) {
  if (audioEl && audioEl.duration) {
    audioEl.currentTime = frac * audioEl.duration;
  }
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

  // Seek listeners are registered once here to avoid accumulation across syntheses.
  $('time-track').addEventListener('click', e => {
    const rect = e.currentTarget.getBoundingClientRect();
    _seekToFrac((e.clientX - rect.left) / rect.width);
  });

  $('canvas-playback').addEventListener('click', e => {
    const rect = e.currentTarget.getBoundingClientRect();
    _seekToFrac((e.clientX - rect.left) / rect.width);
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

function formatSpeed(v) {
  const s = v.toFixed(1);
  return (s.endsWith('.0') ? s.slice(0, -2) : s) + '×';
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
    const label = (i + 1) + '#' + String(date.getMonth() + 1).padStart(2, '0') + String(date.getDate()).padStart(2, '0');
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
    
    addHistoryEntry(
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
      setTimeout(() => {
        drawRadarChart($('canvas-radar'), refCharacteristics, json.characteristics);
        setVisible('card-radar', true);
      }, 100);
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
    setTimeout(() => drawLineChart($('canvas-history'), history), 100);
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


// ── V2.0 New Features JavaScript ─────────────────────────────

let targetRefId = null;
let targetCharacteristics = null;
let targetBlob = null;
let targetAudioCtx = null;

let currentFingerprint = null;
let compareFingerprint = null;
let fpMode = 'single';

let chaseModeActive = false;
let chaseF0Data = null;
let chaseAudioContext = null;
let chaseAnalyser = null;
let chaseScriptProcessor = null;
let chaseMediaStream = null;
let chaseAnimFrame = null;
let chaseStartTime = 0;
let chaseStats = {
  frameScores: [],
  combo: 0,
  maxCombo: 0,
  perfectFrames: 0,
  goodFrames: 0,
  totalFrames: 0,
  finalScore: 0
};


// ── LLM Configuration Panel ──────────────────────────────────
function setupLLMPanel() {
  $('btn-settings').addEventListener('click', () => {
    setVisible('settings-modal', true);
    loadSavedLLMConfig();
  });
  
  function closeModal() {
    setVisible('settings-modal', false);
  }
  
  $('settings-close').addEventListener('click', closeModal);
  $('settings-cancel').addEventListener('click', closeModal);
  $('settings-save').addEventListener('click', async () => {
    await saveLLMConfig();
    closeModal();
  });
  
  $('settings-modal').addEventListener('click', (e) => {
    if (e.target.id === 'settings-modal') {
      closeModal();
    }
  });
  
  $('btn-toggle-api-key').addEventListener('click', () => {
    const input = $('llm-api-key');
    const btn = $('btn-toggle-api-key');
    if (input.type === 'password') {
      input.type = 'text';
      btn.textContent = '隐藏';
    } else {
      input.type = 'password';
      btn.textContent = '显示';
    }
  });
  
  $('btn-save-llm-config').addEventListener('click', saveLLMConfig);
  $('btn-test-llm').addEventListener('click', testLLMConnection);
}

async function loadSavedLLMConfig() {
  try {
    const res = await fetch('/llm/config');
    const json = await res.json();
    
    if (json.success && json.config) {
      const cfg = json.config;
      $('llm-base-url').value = cfg.base_url || '';
      $('llm-model-name').value = cfg.model_name || '';
      
      updateLLMStatusDisplay(cfg.connected, cfg.has_api_key);
    }
  } catch (e) {
    console.warn('Failed to load LLM config:', e);
  }
}

async function saveLLMConfig() {
  const baseUrl = $('llm-base-url').value.trim();
  const apiKey = $('llm-api-key').value.trim();
  const modelName = $('llm-model-name').value.trim();
  
  try {
    const res = await fetch('/llm/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        base_url: baseUrl,
        api_key: apiKey,
        model_name: modelName
      })
    });
    
    const json = await res.json();
    if (json.success) {
      showToast('LLM 配置已保存', 'success');
    } else {
      showToast(json.error || '保存失败', 'error');
    }
  } catch (e) {
    showToast('保存配置失败: ' + e.message, 'error');
  }
}

async function testLLMConnection() {
  const btn = $('btn-test-llm');
  const btnText = $('test-llm-text');
  const resultEl = $('llm-test-result');
  
  btn.disabled = true;
  btnText.textContent = '测试中...';
  setVisible(resultEl, false);
  
  const baseUrl = $('llm-base-url').value.trim();
  const apiKey = $('llm-api-key').value.trim();
  const modelName = $('llm-model-name').value.trim();
  
  try {
    const res = await fetch('/llm/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        base_url: baseUrl,
        api_key: apiKey,
        model_name: modelName
      })
    });
    
    const json = await res.json();
    
    setVisible(resultEl, true);
    resultEl.className = 'llm-status';
    
    if (json.success) {
      resultEl.classList.add('success');
      $('llm-status-icon').textContent = '✓';
      $('llm-status-text').textContent = json.message || '连接成功！';
      showToast('LLM 连接成功！', 'success');
    } else {
      resultEl.classList.add('error');
      $('llm-status-icon').textContent = '✗';
      $('llm-status-text').textContent = json.error || '连接失败';
      showToast('LLM 连接失败', 'error');
    }
    
    updateLLMStatusDisplay(json.success, !!apiKey);
    
  } catch (e) {
    setVisible(resultEl, true);
    resultEl.className = 'llm-status error';
    $('llm-status-icon').textContent = '✗';
    $('llm-status-text').textContent = '网络错误: ' + e.message;
    showToast('连接测试失败', 'error');
  } finally {
    btn.disabled = false;
    btnText.textContent = '🔍 测试连接';
  }
}

function updateLLMStatusDisplay(connected, hasKey) {
  const statusEl = $('llm-config-status');
  const dotEl = $('config-dot');
  const textEl = $('config-text');
  
  if (connected) {
    setVisible(statusEl, true);
    dotEl.className = 'config-dot';
    textEl.textContent = '已连接';
  } else if (hasKey || $('llm-base-url').value) {
    setVisible(statusEl, true);
    dotEl.className = 'config-dot disconnected';
    textEl.textContent = '配置已保存，未连接';
  } else {
    setVisible(statusEl, false);
  }
}


// ── Accent Morphing (Blender) ────────────────────────────────
function setupAccentMorphing() {
  const morphSlider = $('morph-slider');
  const morphValue = $('morph-value');
  
  morphSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    morphValue.innerHTML = `混合比例: <strong>${val}%</strong>`;
    
    const text = $('practice-text').value.trim();
    if (!text || !targetRefId) {
      return;
    }
    
    if (morphDebounceTimer) {
      clearTimeout(morphDebounceTimer);
    }
    
    morphDebounceTimer = setTimeout(async () => {
      if (isMorphSynthesizing) return;
      
      const currentVal = parseInt(morphSlider.value) / 100;
      if (currentVal === lastBlendFactor && blendedAudioId) {
        return;
      }
      
      await runBlendedSynthesis({ autoPlay: true, quiet: true });
    }, MORPH_DEBOUNCE_MS);
  });
  
  $('btn-select-target').addEventListener('click', () => {
    setVisible('target-select-modal', true);
    resetTargetSelection();
  });
  
  $('modal-close').addEventListener('click', () => setVisible('target-select-modal', false));
  $('btn-cancel-target').addEventListener('click', () => setVisible('target-select-modal', false));
  
  $('target-select-modal').addEventListener('click', (e) => {
    if (e.target.id === 'target-select-modal') {
      setVisible('target-select-modal', false);
    }
  });
  
  $('btn-upload-target').addEventListener('click', () => {
    $('target-file-input').click();
  });
  
  $('target-file-input').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
      await handleTargetAudioFile(file);
    }
    e.target.value = '';
  });
  
  $('btn-record-target').addEventListener('click', toggleTargetRecording);
  
  $('btn-confirm-target').addEventListener('click', () => {
    if (targetRefId && targetCharacteristics) {
      setVisible('target-select-modal', false);
      setVisible('morph-target-select', false);
      setVisible('morph-preview-target', true);
      
      drawMiniFingerprint($('canvas-morph-target'), targetCharacteristics, '#06b6d4');
      $('btn-morph-synthesize').disabled = false;
      
      showToast('目标口音已设置', 'success');
    }
  });
  
  $('btn-morph-synthesize').addEventListener('click', runBlendedSynthesis);
  $('btn-morph-preview').addEventListener('click', previewBlendedAudio);
}

async function handleTargetAudioFile(file) {
  if (!file.type.startsWith('audio/') && !/\.(wav|mp3|ogg|m4a|flac|aac)$/i.test(file.name)) {
    showToast('请选择音频文件', 'error');
    return;
  }
  
  showToast('正在处理目标音频…');
  
  try {
    const wavBlob = await blobToWAV(file);
    targetBlob = wavBlob;
    
    $('audio-target').src = URL.createObjectURL(file);
    setVisible('target-preview', true);
    
    try {
      const ctx = getAudioCtx();
      const arrBuf = await wavBlob.arrayBuffer();
      const audioBuf = await ctx.decodeAudioData(arrBuf.slice(0));
      drawWaveform($('canvas-target-wave'), audioBuf, 0, '#06b6d4');
    } catch { }
    
    setVisible('target-analysis-status', true);
    await analyzeTargetAudio(wavBlob);
    
  } catch (e) {
    showToast('无法解码该音频文件', 'error');
  }
}

async function analyzeTargetAudio(wavBlob) {
  const form = new FormData();
  form.append('audio', wavBlob, 'target.wav');
  
  try {
    const res = await fetch('/analyze', { method: 'POST', body: form });
    const json = await res.json();
    
    if (res.ok && json.success) {
      targetRefId = json.ref_id;
      targetCharacteristics = json.characteristics;
      $('btn-confirm-target').disabled = false;
      setVisible('target-analysis-status', false);
      showToast('目标口音分析完成', 'success');
    } else {
      showToast(json.error || '分析失败', 'error');
    }
  } catch (e) {
    showToast('网络错误', 'error');
  }
}

let targetRecorder = null;
let targetRecordChunks = [];
let targetIsRecording = false;
let targetRecordTimer = null;
let targetRecordSeconds = 0;
let targetRecordAnimFrame = null;

async function toggleTargetRecording() {
  if (!targetIsRecording) {
    await startTargetRecording();
  } else {
    stopTargetRecording();
  }
}

async function startTargetRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    targetMediaStream = stream;
    
    const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg']
      .find(t => MediaRecorder.isTypeSupported(t)) || '';
    
    targetRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    targetRecordChunks = [];
    targetRecorder.ondataavailable = e => { if (e.data.size) targetRecordChunks.push(e.data); };
    targetRecorder.onstop = finalizeTargetRecording;
    targetRecorder.start(100);
    
    const ctx = getAudioCtx();
    const source = ctx.createMediaStreamSource(stream);
    targetAnalyser = ctx.createAnalyser();
    targetAnalyser.fftSize = 512;
    source.connect(targetAnalyser);
    
    drawTargetLiveWave($('canvas-target-wave'));
    
    targetRecordSeconds = 0;
    updateTargetRecordTimer();
    targetRecordTimer = setInterval(() => {
      targetRecordSeconds++;
      updateTargetRecordTimer();
      if (targetRecordSeconds >= 60) stopTargetRecording();
    }, 1000);
    
    targetIsRecording = true;
    $('btn-record-target').textContent = '⏹ 停止录制';
    $('btn-record-target').classList.add('recording');
    setVisible('target-preview', true);
    
  } catch {
    showToast('无法访问麦克风', 'error');
  }
}

function stopTargetRecording() {
  if (!targetIsRecording) return;
  targetRecorder.stop();
  targetMediaStream.getTracks().forEach(t => t.stop());
  cancelAnimationFrame(targetRecordAnimFrame);
  clearInterval(targetRecordTimer);
  targetIsRecording = false;
  $('btn-record-target').textContent = '🎤 录制目标口音';
  $('btn-record-target').classList.remove('recording');
}

function updateTargetRecordTimer() {
  const m = String(Math.floor(targetRecordSeconds / 60)).padStart(2, '0');
  const s = String(targetRecordSeconds % 60).padStart(2, '0');
}

function drawTargetLiveWave(canvas) {
  const W = canvas.offsetWidth || 600;
  const H = canvas.offsetHeight || 60;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  const buf = new Uint8Array(targetAnalyser.fftSize);

  function loop() {
    if (!targetIsRecording) return;
    targetRecordAnimFrame = requestAnimationFrame(loop);
    targetAnalyser.getByteTimeDomainData(buf);
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

async function finalizeTargetRecording() {
  const raw = new Blob(targetRecordChunks, { type: targetRecorder.mimeType || 'audio/webm' });
  showToast('正在处理录音…');
  
  try {
    const wav = await blobToWAV(raw);
    targetBlob = wav;
    
    $('audio-target').src = URL.createObjectURL(raw);
    
    try {
      const ctx = getAudioCtx();
      const arrBuf = await wav.arrayBuffer();
      const audioBuf = await ctx.decodeAudioData(arrBuf.slice(0));
      drawWaveform($('canvas-target-wave'), audioBuf, 0, '#06b6d4');
    } catch { }
    
    setVisible('target-analysis-status', true);
    await analyzeTargetAudio(wav);
    
  } catch (e) {
    showToast('录音处理失败', 'error');
  }
}

function resetTargetSelection() {
  $('audio-target').src = '';
  setVisible('target-preview', false);
  setVisible('target-analysis-status', false);
  $('btn-confirm-target').disabled = true;
  targetRefId = null;
  targetCharacteristics = null;
}

function drawMiniFingerprint(canvas, chars, color) {
  const W = canvas.offsetWidth || 150;
  const H = canvas.offsetHeight || 150;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  
  const centerX = W / 2;
  const centerY = H / 2;
  const radius = Math.min(W, H) * 0.35;
  
  const dims = {
    pitch: Math.min(1, Math.max(0, ((chars.mean_pitch || 150) - 80) / 350)),
    pitchRange: Math.min(1, Math.max(0, (chars.pitch_range || 80) / 200)),
    speakingRate: Math.min(1, Math.max(0, (chars.speaking_rate || 4.5) / 8)),
    energy: Math.min(1, Math.max(0, (chars.rms_energy || 0.01) * 8000)),
    brightness: Math.min(1, Math.max(0, ((chars.spectral_centroid || 2000) - 800) / 4000)),
    pitchVariance: Math.min(1, Math.max(0, (chars.pitch_std || 30) / 80)),
  };
  
  const keys = ['pitch', 'pitchVariance', 'speakingRate', 'energy', 'brightness', 'pitchRange'];
  const n = keys.length;
  const angleStep = (Math.PI * 2) / n;
  
  ctx.clearRect(0, 0, W, H);
  
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  for (let level = 1; level <= 3; level++) {
    const r = (radius / 3) * level;
    ctx.beginPath();
    ctx.arc(centerX, centerY, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  
  for (let i = 0; i < n; i++) {
    const angle = angleStep * i - Math.PI / 2;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(centerX + radius * Math.cos(angle), centerY + radius * Math.sin(angle));
    ctx.stroke();
  }
  
  ctx.beginPath();
  for (let i = 0; i <= n; i++) {
    const idx = i % n;
    const angle = angleStep * idx - Math.PI / 2;
    const value = dims[keys[idx]];
    const r = value * radius;
    const x = centerX + r * Math.cos(angle);
    const y = centerY + r * Math.sin(angle);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = color + '25';
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
  
  for (let i = 0; i < n; i++) {
    const angle = angleStep * i - Math.PI / 2;
    const value = dims[keys[i]];
    const r = value * radius;
    const x = centerX + r * Math.cos(angle);
    const y = centerY + r * Math.sin(angle);
    
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
}

let lastBlendFactor = -1;
let blendedAudioId = null;
let morphDebounceTimer = null;
let isMorphSynthesizing = false;
const MORPH_DEBOUNCE_MS = 350;

async function runBlendedSynthesis(options = {}) {
  const { autoPlay = false, quiet = false } = options;
  const blend = parseInt($('morph-slider').value) / 100;
  const text = $('practice-text').value.trim();
  
  if (!text) {
    if (!quiet) showToast('请输入练习文字', 'error');
    return false;
  }
  
  if (!targetRefId) {
    if (!quiet) showToast('请先选择目标口音', 'error');
    return false;
  }
  
  if (blend === lastBlendFactor && blendedAudioId) {
    if (autoPlay && audioEl) {
      audioEl.currentTime = 0;
      audioEl.play().catch(() => {});
    }
    return true;
  }
  
  if (isMorphSynthesizing) return false;
  isMorphSynthesizing = true;
  
  const synthBtn = $('btn-morph-synthesize');
  const previewBtn = $('btn-morph-preview');
  if (synthBtn) synthBtn.disabled = true;
  if (previewBtn) previewBtn.disabled = true;
  
  setVisible('morph-status', true);
  $('morph-msg').textContent = `正在合成混合口音 ${Math.round(blend * 100)}%…`;
  
  try {
    const res = await fetch('/synthesize-blend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        ref_id1: refId,
        ref_id2: targetRefId,
        blend: blend
      })
    });
    
    const json = await res.json();
    
    if (res.ok && json.success) {
      blendedAudioId = json.audio_id;
      lastBlendFactor = blend;
      
      audioId = blendedAudioId;
      audioUrl = `/audio/${audioId}`;
      await setupPlayback(audioUrl);
      
      setVisible('morph-status', false);
      
      if (!quiet) {
        showToast(`混合口音已合成 (${Math.round(blend * 100)}%)`, 'success');
      }
      
      if (autoPlay && audioEl) {
        audioEl.currentTime = 0;
        audioEl.play().catch(() => {});
      }
      
      return true;
    } else {
      if (!quiet) showToast(json.error || '合成失败', 'error');
      return false;
    }
  } catch (e) {
    if (!quiet) showToast('网络错误', 'error');
    return false;
  } finally {
    isMorphSynthesizing = false;
    if (synthBtn) synthBtn.disabled = false;
    if (previewBtn) previewBtn.disabled = false;
  }
}

async function previewBlendedAudio() {
  const blend = parseInt($('morph-slider').value) / 100;
  const text = $('practice-text').value.trim();
  
  if (!text) {
    showToast('请输入练习文字', 'error');
    return;
  }
  
  if (!targetRefId) {
    showToast('请先选择目标口音', 'error');
    return;
  }
  
  if (audioEl && !audioEl.paused) {
    audioEl.pause();
    return;
  }
  
  if (blend === lastBlendFactor && blendedAudioId && audioEl) {
    audioEl.currentTime = 0;
    audioEl.play().catch(() => showToast('播放失败', 'error'));
    return;
  }
  
  await runBlendedSynthesis({ autoPlay: true, quiet: false });
}


// ── Voice Fingerprint Visualization ──────────────────────────
function setupFingerprint() {
  document.querySelectorAll('.fp-mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.fp-mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      fpMode = btn.dataset.mode;
      
      setVisible('fp-compare-select', fpMode === 'compare');
      setVisible('fp-similarity', false);
      
      if (fpMode === 'single') {
        loadAndDrawFingerprint();
      } else if (compareFingerprint) {
        loadAndDrawCompareFingerprint();
      }
    });
  });
  
  $('btn-select-fp-compare').addEventListener('click', () => {
    if (targetRefId && targetCharacteristics) {
      fetchFingerprint(targetRefId).then(fp => {
        compareFingerprint = fp;
        $('fp-compare-color2').style.background = fp.color.hex;
        loadAndDrawCompareFingerprint();
      });
    } else {
      showToast('请先在口音变形器中选择目标口音', 'error');
    }
  });
  
  $('btn-export-fp').addEventListener('click', exportFingerprint);
  $('btn-refresh-fp').addEventListener('click', () => {
    if (fpMode === 'single') {
      loadAndDrawFingerprint();
    } else {
      loadAndDrawCompareFingerprint();
    }
  });
}

async function fetchFingerprint(audioId) {
  try {
    const res = await fetch('/fingerprint/' + audioId);
    const json = await res.json();
    if (res.ok && json.fingerprint) {
      return json.fingerprint;
    }
  } catch (e) {
    console.warn('Failed to fetch fingerprint:', e);
  }
  return null;
}

async function loadAndDrawFingerprint() {
  if (!audioId && !refId) {
    setVisible('card-fingerprint', false);
    return;
  }
  
  const id = audioId || refId;
  const fp = await fetchFingerprint(id);
  
  if (fp) {
    currentFingerprint = fp;
    drawFingerprintCanvas($('canvas-fingerprint'), fp, fpMode === 'single');
    updateFingerprintInfo(fp);
    setVisible('card-fingerprint', true);
  }
}

async function loadAndDrawCompareFingerprint() {
  if (!currentFingerprint) {
    if (audioId || refId) {
      const id = audioId || refId;
      currentFingerprint = await fetchFingerprint(id);
    }
    if (!currentFingerprint) return;
  }
  
  if (!compareFingerprint) {
    if (targetRefId) {
      compareFingerprint = await fetchFingerprint(targetRefId);
      if (compareFingerprint) {
        $('fp-compare-color2').style.background = compareFingerprint.color.hex;
      }
    }
    if (!compareFingerprint) {
      showToast('请选择对比口音', 'error');
      return;
    }
  }
  
  drawDualFingerprintCanvas($('canvas-fingerprint'), currentFingerprint, compareFingerprint);
  
  try {
    const res = await fetch('/fingerprint/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio_id1: audioId || refId, audio_id2: targetRefId })
    });
    const json = await res.json();
    
    if (json.success && json.comparison) {
      $('fp-sim-value').textContent = json.comparison.overall_similarity + '%';
      setVisible('fp-similarity', true);
    }
  } catch (e) {
    console.warn('Compare failed:', e);
  }
  
  setVisible('card-fingerprint', true);
}

function drawFingerprintCanvas(canvas, fp, drawLabels = true) {
  const W = canvas.offsetWidth || 300;
  const H = W;
  canvas.width = W * 2;
  canvas.height = H * 2;
  const ctx = canvas.getContext('2d');
  ctx.scale(2, 2);
  
  const centerX = W / 2;
  const centerY = H / 2;
  const radius = Math.min(W, H) * 0.38;
  
  const angles = fp.polar_coords.angles;
  const radii = fp.polar_coords.radii;
  const color = fp.color.hex;
  const labels = fp.labels;
  
  ctx.clearRect(0, 0, W, H);
  
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  for (let level = 1; level <= 4; level++) {
    const r = (radius / 4) * level;
    ctx.beginPath();
    ctx.arc(centerX, centerY, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  
  const n = angles.length;
  for (let i = 0; i < n; i++) {
    const angle = angles[i];
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(centerX + radius * Math.cos(angle), centerY + radius * Math.sin(angle));
    ctx.stroke();
  }
  
  ctx.beginPath();
  for (let i = 0; i <= n; i++) {
    const idx = i % n;
    const angle = angles[idx];
    const r = radii[idx] * radius;
    const x = centerX + r * Math.cos(angle);
    const y = centerY + r * Math.sin(angle);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = color + '33';
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.stroke();
  
  for (let i = 0; i < n; i++) {
    const angle = angles[i];
    const r = radii[i] * radius;
    const x = centerX + r * Math.cos(angle);
    const y = centerY + r * Math.sin(angle);
    
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
  
  const texture = fp.texture || [];
  for (let i = 0; i < Math.min(texture.length, 24); i += 2) {
    const tx = (texture[i] - 0.5) * radius * 1.8 + centerX;
    const ty = (texture[i + 1] - 0.5) * radius * 1.8 + centerY;
    ctx.beginPath();
    ctx.arc(tx, ty, 3, 0, Math.PI * 2);
    ctx.fillStyle = color + '66';
    ctx.fill();
  }
  
  if (drawLabels && labels) {
    ctx.fillStyle = '#64748b';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    const keys = Object.keys(labels);
    for (let i = 0; i < keys.length && i < n; i++) {
      const angle = angles[i];
      const labelR = radius + 18;
      const lx = centerX + labelR * Math.cos(angle);
      const ly = centerY + labelR * Math.sin(angle);
      ctx.fillText(labels[keys[i]] || '', lx, ly);
    }
  }
}

function drawDualFingerprintCanvas(canvas, fp1, fp2) {
  const W = canvas.offsetWidth || 300;
  const H = W;
  canvas.width = W * 2;
  canvas.height = H * 2;
  const ctx = canvas.getContext('2d');
  ctx.scale(2, 2);
  
  const centerX = W / 2;
  const centerY = H / 2;
  const radius = Math.min(W, H) * 0.38;
  
  const color1 = fp1.color.hex;
  const color2 = fp2.color.hex;
  
  ctx.clearRect(0, 0, W, H);
  
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  for (let level = 1; level <= 4; level++) {
    const r = (radius / 4) * level;
    ctx.beginPath();
    ctx.arc(centerX, centerY, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  
  const n = fp1.polar_coords.angles.length;
  for (let i = 0; i < n; i++) {
    const angle = fp1.polar_coords.angles[i];
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(centerX + radius * Math.cos(angle), centerY + radius * Math.sin(angle));
    ctx.stroke();
  }
  
  function drawOne(fp, color) {
    const angles = fp.polar_coords.angles;
    const radii = fp.polar_coords.radii;
    
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const idx = i % n;
      const angle = angles[idx];
      const r = radii[idx] * radius;
      const x = centerX + r * Math.cos(angle);
      const y = centerY + r * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = color + '22';
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  
  drawOne(fp1, color1);
  drawOne(fp2, color2);
  
  ctx.fillStyle = '#64748b';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  
  const labels = fp1.labels;
  const keys = Object.keys(labels);
  for (let i = 0; i < keys.length && i < n; i++) {
    const angle = fp1.polar_coords.angles[i];
    const labelR = radius + 18;
    const lx = centerX + labelR * Math.cos(angle);
    const ly = centerY + labelR * Math.sin(angle);
    ctx.fillText(labels[keys[i]] || '', lx, ly);
  }
}

function updateFingerprintInfo(fp) {
  $('fp-color-dot').style.background = fp.color.hex;
  $('fp-id').textContent = 'ID: ' + fp.id;
  $('fp-uniqueness').textContent = '独特性: ' + fp.uniqueness;
}

function exportFingerprint() {
  if (!audioId && !refId) {
    showToast('没有可导出的音频', 'error');
    return;
  }
  
  if (fpMode === 'compare' && targetRefId) {
    const url = '/fingerprint/export-compare?id1=' + (audioId || refId) + '&id2=' + targetRefId;
    window.open(url, '_blank');
  } else {
    const url = '/fingerprint/export/' + (audioId || refId);
    window.open(url, '_blank');
  }
}


// ── Pitch Chase Mode ──────────────────────────────────────────
function setupPitchChase() {
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const mode = btn.dataset.mode;
      
      setVisible('standard-mode-panel', mode === 'standard');
      setVisible('chase-mode-panel', mode === 'chase');
      
      if (mode === 'chase') {
        initializeChaseMode();
      } else {
        stopChaseMode();
      }
    });
  });
  
  $('btn-start-chase').addEventListener('click', startChaseMode);
  $('btn-stop-chase').addEventListener('click', stopChaseMode);
  $('btn-reset-chase').addEventListener('click', resetChaseMode);
}

async function initializeChaseMode() {
  if (!audioId) {
    showToast('请先合成语音', 'error');
    return;
  }
  
  setVisible('chase-status', true);
  $('chase-msg').textContent = '正在提取目标音高曲线…';
  
  try {
    const res = await fetch('/extract-f0/' + audioId);
    const json = await res.json();
    
    if (res.ok && json.success) {
      chaseF0Data = json;
      drawChaseTrack($('canvas-pitch-chase'), json);
      setVisible('chase-stats', false);
      setVisible('chase-overlay', true);
      $('chase-start-hint').textContent = '点击「开始追逐」准备';
      showToast('音高曲线已就绪', 'success');
    } else {
      showToast(json.error || '提取失败', 'error');
    }
  } catch (e) {
    showToast('网络错误', 'error');
  } finally {
    setVisible('chase-status', false);
  }
}

function drawChaseTrack(canvas, f0Data) {
  const W = canvas.offsetWidth || 600;
  const H = canvas.offsetHeight || 200;
  canvas.width = W * 2;
  canvas.height = H * 2;
  const ctx = canvas.getContext('2d');
  ctx.scale(2, 2);
  
  const padding = { top: 30, right: 20, bottom: 30, left: 50 };
  const chartW = W - padding.left - padding.right;
  const chartH = H - padding.top - padding.bottom;
  
  const f0Values = f0Data.f0_interpolated.filter(v => v !== null);
  const minF0 = Math.min(...f0Values) * 0.95;
  const maxF0 = Math.max(...f0Values) * 1.05;
  const f0Range = maxF0 - minF0 || 100;
  
  const timePoints = f0Data.time_points;
  const maxTime = timePoints[timePoints.length - 1] || f0Data.duration_seconds;
  
  ctx.clearRect(0, 0, W, H);
  
  const gradient = ctx.createLinearGradient(0, 0, 0, H);
  gradient.addColorStop(0, '#0f172a');
  gradient.addColorStop(1, '#1e293b');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, W, H);
  
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 0.5;
  
  const yGridCount = 5;
  for (let i = 0; i <= yGridCount; i++) {
    const y = padding.top + (chartH / yGridCount) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(W - padding.right, y);
    ctx.stroke();
    
    const freq = maxF0 - (f0Range / yGridCount) * i;
    ctx.fillStyle = '#64748b';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(Math.round(freq) + ' Hz', padding.left - 8, y + 3);
  }
  
  ctx.strokeStyle = '#334155';
  const xGridCount = Math.min(10, Math.floor(maxTime));
  for (let i = 0; i <= xGridCount; i++) {
    const t = (maxTime / xGridCount) * i;
    const x = padding.left + (t / maxTime) * chartW;
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, H - padding.bottom);
    ctx.stroke();
    
    ctx.fillStyle = '#64748b';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(t.toFixed(1) + 's', x, H - padding.bottom + 15);
  }
  
  ctx.beginPath();
  ctx.strokeStyle = '#8b5cf6';
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  
  let started = false;
  for (let i = 0; i < f0Data.f0_interpolated.length; i++) {
    const f0 = f0Data.f0_interpolated[i];
    if (f0 === null) {
      if (started) {
        ctx.stroke();
        ctx.beginPath();
        started = false;
      }
      continue;
    }
    
    const t = timePoints[i];
    const x = padding.left + (t / maxTime) * chartW;
    const y = padding.top + chartH - ((f0 - minF0) / f0Range) * chartH;
    
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  if (started) ctx.stroke();
  
  ctx.fillStyle = '#e2e8f0';
  ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('目标音高曲线 (点击开始追逐)', W / 2, 18);
  
  canvas._chaseData = {
    minF0, maxF0, f0Range, maxTime,
    padding, chartW, chartH,
    f0Interpolated: f0Data.f0_interpolated,
    timePoints: f0Data.time_points
  };
}

let chaseCurrentF0 = null;
let chaseFrameIndex = 0;
let chaseHistory = [];

async function startChaseMode() {
  if (!chaseF0Data) {
    await initializeChaseMode();
    if (!chaseF0Data) return;
  }
  
  try {
    chaseMediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch {
    showToast('无法访问麦克风', 'error');
    return;
  }
  
  chaseAudioContext = getAudioCtx();
  const source = chaseAudioContext.createMediaStreamSource(chaseMediaStream);
  chaseAnalyser = chaseAudioContext.createAnalyser();
  chaseAnalyser.fftSize = 2048;
  source.connect(chaseAnalyser);
  
  chaseStats = {
    frameScores: [],
    combo: 0,
    maxCombo: 0,
    perfectFrames: 0,
    goodFrames: 0,
    totalFrames: 0,
    finalScore: 0
  };
  chaseHistory = [];
  chaseFrameIndex = 0;
  chaseStartTime = performance.now();
  chaseModeActive = true;
  
  setVisible('chase-overlay', false);
  setVisible('chase-stats', true);
  setVisible('chase-final-score', false);
  
  $('btn-start-chase').hidden = true;
  $('btn-stop-chase').hidden = false;
  
  chaseAnimFrame = requestAnimationFrame(chaseLoop);
  
  showToast('音高追逐已开始！对着麦克风说话', 'success');
}

function chaseLoop() {
  if (!chaseModeActive) return;
  
  chaseAnimFrame = requestAnimationFrame(chaseLoop);
  
  const canvas = $('canvas-pitch-chase');
  const data = canvas._chaseData;
  if (!data) return;
  
  const elapsed = (performance.now() - chaseStartTime) / 1000;
  
  const bufferLength = chaseAnalyser.frequencyBinCount;
  const timeData = new Float32Array(bufferLength);
  chaseAnalyser.getFloatTimeDomainData(timeData);
  
  chaseCurrentF0 = estimateF0FromTimeDomain(timeData, chaseAudioContext.sampleRate);
  
  const f0Interpolated = data.f0Interpolated;
  const timePoints = data.timePoints;
  
  let targetF0 = null;
  for (let i = 0; i < timePoints.length; i++) {
    if (timePoints[i] >= elapsed) {
      targetF0 = f0Interpolated[i];
      chaseFrameIndex = i;
      break;
    }
  }
  
  if (elapsed > data.maxTime) {
    stopChaseMode();
    return;
  }
  
  $('chase-current-pitch').textContent = chaseCurrentF0 ? Math.round(chaseCurrentF0) + ' Hz' : '— Hz';
  $('chase-target-pitch').textContent = targetF0 ? Math.round(targetF0) + ' Hz' : '— Hz';
  
  let frameScore = 0;
  if (chaseCurrentF0 !== null && targetF0 !== null) {
    const diff = Math.abs(chaseCurrentF0 - targetF0);
    const percentDiff = (diff / targetF0) * 100;
    
    if (percentDiff < 3) {
      frameScore = 100;
      chaseStats.perfectFrames++;
    } else if (percentDiff < 8) {
      frameScore = 80;
      chaseStats.goodFrames++;
    } else if (percentDiff < 15) {
      frameScore = 50;
    } else {
      frameScore = Math.max(0, 100 - percentDiff * 2);
    }
    
    chaseStats.totalFrames++;
    
    if (percentDiff < 8) {
      chaseStats.combo++;
      if (chaseStats.combo > chaseStats.maxCombo) {
        chaseStats.maxCombo = chaseStats.combo;
      }
    } else {
      chaseStats.combo = 0;
    }
    
    chaseStats.frameScores.push(frameScore);
    
    $('chase-realtime-score').textContent = Math.round(frameScore);
    $('chase-combo').textContent = chaseStats.combo;
    
    if (chaseStats.combo > 0) {
      const comboEl = $('chase-combo');
      comboEl.classList.remove('active');
      void comboEl.offsetWidth;
      comboEl.classList.add('active');
    }
  }
  
  chaseHistory.push({
    time: elapsed,
    userF0: chaseCurrentF0,
    targetF0: targetF0,
    score: frameScore
  });
  
  drawChaseWithProgress(canvas, data, elapsed, chaseHistory);
}

function estimateF0FromTimeDomain(timeData, sampleRate) {
  let rms = 0;
  for (let i = 0; i < timeData.length; i++) {
    rms += timeData[i] * timeData[i];
  }
  rms = Math.sqrt(rms / timeData.length);
  if (rms < 0.02) return null;
  
  let maxCorr = 0;
  let bestLag = 0;
  
  const minLag = Math.floor(sampleRate / 500);
  const maxLag = Math.floor(sampleRate / 80);
  
  for (let lag = minLag; lag <= maxLag; lag++) {
    let corr = 0;
    for (let i = 0; i < timeData.length - lag; i++) {
      corr += timeData[i] * timeData[i + lag];
    }
    if (corr > maxCorr) {
      maxCorr = corr;
      bestLag = lag;
    }
  }
  
  if (maxCorr > 0.01 && bestLag > 0) {
    return sampleRate / bestLag;
  }
  
  return null;
}

function drawChaseWithProgress(canvas, data, currentTime, history) {
  const W = canvas.offsetWidth || 600;
  const H = canvas.offsetHeight || 200;
  canvas.width = W * 2;
  canvas.height = H * 2;
  const ctx = canvas.getContext('2d');
  ctx.scale(2, 2);
  
  const { padding, chartW, chartH, minF0, maxF0, f0Range, maxTime, f0Interpolated, timePoints } = data;
  
  const gradient = ctx.createLinearGradient(0, 0, 0, H);
  gradient.addColorStop(0, '#0f172a');
  gradient.addColorStop(1, '#1e293b');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, W, H);
  
  const progressX = padding.left + (currentTime / maxTime) * chartW;
  
  ctx.fillStyle = 'rgba(139, 92, 246, 0.08)';
  ctx.fillRect(padding.left, padding.top, progressX - padding.left, chartH);
  
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 0.5;
  
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (chartH / 5) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(W - padding.right, y);
    ctx.stroke();
    
    const freq = maxF0 - (f0Range / 5) * i;
    ctx.fillStyle = '#64748b';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(Math.round(freq) + ' Hz', padding.left - 8, y + 3);
  }
  
  ctx.beginPath();
  ctx.strokeStyle = '#8b5cf6';
  ctx.lineWidth = 2;
  
  let started = false;
  for (let i = 0; i < f0Interpolated.length; i++) {
    const f0 = f0Interpolated[i];
    if (f0 === null) {
      if (started) {
        ctx.stroke();
        ctx.beginPath();
        started = false;
      }
      continue;
    }
    
    const t = timePoints[i];
    const x = padding.left + (t / maxTime) * chartW;
    const y = padding.top + chartH - ((f0 - minF0) / f0Range) * chartH;
    
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  if (started) ctx.stroke();
  
  if (history.length > 1) {
    ctx.beginPath();
    ctx.lineWidth = 3;
    
    for (let i = 0; i < history.length; i++) {
      const point = history[i];
      if (point.userF0 === null) continue;
      
      const x = padding.left + (point.time / maxTime) * chartW;
      const y = padding.top + chartH - ((point.userF0 - minF0) / f0Range) * chartH;
      
      let color;
      if (point.score >= 80) {
        color = '#10b981';
      } else if (point.score >= 50) {
        color = '#f59e0b';
      } else {
        color = '#ef4444';
      }
      
      if (i === 0 || (i > 0 && history[i - 1].score !== point.score)) {
        if (i > 0) ctx.stroke();
        ctx.beginPath();
        ctx.strokeStyle = color;
        if (point.score >= 80) {
          ctx.lineWidth = 4;
        } else {
          ctx.lineWidth = 2.5;
        }
        const prev = history[i - 1];
        if (prev && prev.userF0 !== null) {
          const px = padding.left + (prev.time / maxTime) * chartW;
          const py = padding.top + chartH - ((prev.userF0 - minF0) / f0Range) * chartH;
          ctx.moveTo(px, py);
        } else {
          ctx.moveTo(x, y);
        }
      }
      
      ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  
  ctx.strokeStyle = '#f8fafc';
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(progressX, padding.top);
  ctx.lineTo(progressX, H - padding.bottom);
  ctx.stroke();
  ctx.setLineDash([]);
  
  if (history.length > 0) {
    const lastPoint = history[history.length - 1];
    if (lastPoint.userF0 !== null) {
      const x = padding.left + (lastPoint.time / maxTime) * chartW;
      const y = padding.top + chartH - ((lastPoint.userF0 - minF0) / f0Range) * chartH;
      
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = lastPoint.score >= 80 ? '#10b981' : lastPoint.score >= 50 ? '#f59e0b' : '#ef4444';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
  
  ctx.fillStyle = '#e2e8f0';
  ctx.font = 'bold 11px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`进度: ${(currentTime / maxTime * 100).toFixed(0)}%`, W / 2, 18);
}

function stopChaseMode() {
  chaseModeActive = false;
  
  if (chaseAnimFrame) {
    cancelAnimationFrame(chaseAnimFrame);
    chaseAnimFrame = null;
  }
  
  if (chaseMediaStream) {
    chaseMediaStream.getTracks().forEach(t => t.stop());
    chaseMediaStream = null;
  }
  
  $('btn-start-chase').hidden = false;
  $('btn-stop-chase').hidden = true;
  
  if (chaseStats.totalFrames > 0) {
    const avgScore = chaseStats.frameScores.reduce((a, b) => a + b, 0) / chaseStats.frameScores.length;
    chaseStats.finalScore = Math.round(avgScore);
    
    $('chase-final-value').textContent = chaseStats.finalScore;
    $('chase-max-combo').textContent = chaseStats.maxCombo;
    $('chase-perfect-frames').textContent = Math.round(chaseStats.perfectFrames / chaseStats.totalFrames * 100) + '%';
    $('chase-good-frames').textContent = Math.round((chaseStats.perfectFrames + chaseStats.goodFrames) / chaseStats.totalFrames * 100) + '%';
    
    setVisible('chase-final-score', true);
    
    const scoreEl = $('chase-final-value');
    scoreEl.classList.remove('good', 'ok', 'needs-work');
    if (chaseStats.finalScore >= 80) scoreEl.classList.add('good');
    else if (chaseStats.finalScore >= 60) scoreEl.classList.add('ok');
    else scoreEl.classList.add('needs-work');
    
    showToast(`音高追逐完成！得分: ${chaseStats.finalScore}`, 'success');
  }
}

function resetChaseMode() {
  stopChaseMode();
  chaseHistory = [];
  chaseStats = {
    frameScores: [],
    combo: 0,
    maxCombo: 0,
    perfectFrames: 0,
    goodFrames: 0,
    totalFrames: 0,
    finalScore: 0
  };
  
  setVisible('chase-stats', false);
  setVisible('chase-final-score', false);
  setVisible('chase-overlay', true);
  $('chase-start-hint').textContent = '点击「开始追逐」准备';
  
  if (chaseF0Data) {
    drawChaseTrack($('canvas-pitch-chase'), chaseF0Data);
  }
  
  $('chase-realtime-score').textContent = '0';
  $('chase-combo').textContent = '0';
}


// ── Update Existing Functions for V2.0 ───────────────────────
const originalSetupPlayback = setupPlayback;

async function setupPlayback(url) {
  await originalSetupPlayback(url);
  
  setVisible('card-morph', true);
  setVisible('card-fingerprint', true);
  
  if (refCharacteristics) {
    drawMiniFingerprint($('canvas-morph-ref'), refCharacteristics, '#8b5cf6');
  }
  
  if (targetRefId) {
    $('btn-morph-synthesize').disabled = false;
    setVisible('morph-target-select', false);
  } else {
    $('btn-morph-synthesize').disabled = true;
    setVisible('morph-target-select', true);
  }
  
  setTimeout(() => loadAndDrawFingerprint(), 100);
}

const originalLoadAndDrawRadar = loadAndDrawRadar;

async function loadAndDrawRadar() {
  await originalLoadAndDrawRadar();
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
  
  setupLLMPanel();
  setupAccentMorphing();
  setupFingerprint();
  setupPitchChase();
  
  refreshHistoryChart();
  loadSavedLLMConfig();
});
