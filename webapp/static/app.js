/* 딥페이크 음성 탐지 데모 — 메인 프론트엔드 로직 */

const $ = (id) => document.getElementById(id);

const WARN_MSG = {
  near_silent: '입력이 거의 무음입니다 — 판정 신뢰도가 낮을 수 있습니다.',
  clipping: '입력에 클리핑(음량 포화)이 있습니다 — 판정에 영향을 줄 수 있습니다.',
  empty: '빈 오디오입니다.',
};

let models = [];
let history = [];
let recorder = null;
let recTimerId = null;
let liveDrawId = null;
let currentBlobUrl = null;

/* ── 초기화 ─────────────────────────────────────────── */

async function init() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    models = data.models;
    const dev = data.device === 'cuda' ? `GPU (${data.gpu_name || 'CUDA'})` : 'CPU';
    $('devicePill').textContent = `${dev} · 모델 ${models.length}개`;
    buildModelSelect(data.default_model);
    buildStreamSelect(data.stream_default_model);
  } catch (e) {
    $('devicePill').textContent = '서버 연결 실패';
  }
}

function buildStreamSelect(defaultId) {
  const sel = $('streamModelSelect');
  sel.innerHTML = '';
  const streamable = models.filter((m) => m.window_sec === 2);
  for (const m of streamable) {
    const opt = document.createElement('option');
    opt.value = m.id;
    const eer = m.t2_eer != null ? ` · T2 EER ${m.t2_eer}%` : '';
    opt.textContent = `${m.name}${eer}`;
    sel.appendChild(opt);
  }
  if (defaultId && streamable.some((m) => m.id === defaultId)) sel.value = defaultId;
  if (!streamable.length) {
    $('streamBtn').disabled = true;
    $('liveSub').textContent = '2초 윈도우 체크포인트가 없어 실시간 감시를 사용할 수 없습니다';
  }
}

function buildModelSelect(defaultId) {
  const sel = $('modelSelect');
  sel.innerHTML = '';
  const groups = {};
  for (const m of models) {
    if (!groups[m.group]) {
      groups[m.group] = document.createElement('optgroup');
      groups[m.group].label = m.group;
      sel.appendChild(groups[m.group]);
    }
    const opt = document.createElement('option');
    opt.value = m.id;
    const eer = m.t2_eer != null ? ` · T2 EER ${m.t2_eer}%` : '';
    opt.textContent = `${m.name} (${m.params}${eer})`;
    groups[m.group].appendChild(opt);
  }
  if (defaultId) sel.value = defaultId;
  updateModelDesc();
  sel.addEventListener('change', updateModelDesc);
}

function updateModelDesc() {
  const m = models.find((x) => x.id === $('modelSelect').value);
  $('modelDesc').textContent = m ? `${m.desc} · 윈도우 ${m.window_sec}초` : '';
}

/* ── 탭 ─────────────────────────────────────────────── */

$('tabMicBtn').addEventListener('click', () => switchTab('mic'));
$('tabFileBtn').addEventListener('click', () => switchTab('file'));
$('tabStreamBtn').addEventListener('click', () => switchTab('stream'));

function switchTab(tab) {
  $('tabMicBtn').classList.toggle('active', tab === 'mic');
  $('tabFileBtn').classList.toggle('active', tab === 'file');
  $('tabStreamBtn').classList.toggle('active', tab === 'stream');
  $('tabMic').hidden = tab !== 'mic';
  $('tabFile').hidden = tab !== 'file';
  $('tabStream').hidden = tab !== 'stream';
  if (tab === 'stream') drawStreamChart();  // 빈 차트라도 축·판정선 표시
}

/* ── 마이크 녹음 ────────────────────────────────────── */

$('recordBtn').addEventListener('click', async () => {
  if (recorder && recorder.recording) {
    await stopRecording();
  } else {
    await startRecording();
  }
});

async function startRecording() {
  try {
    recorder = new PCMRecorder();
    await recorder.start();
  } catch (e) {
    showError(
      e.name === 'NotAllowedError'
        ? '마이크 권한이 거부되었습니다. 브라우저 주소창의 권한 설정을 확인해주세요.'
        : `마이크를 열 수 없습니다: ${e.message}`
    );
    recorder = null;
    return;
  }
  $('recordBtn').classList.add('recording');
  $('recTimer').classList.add('on');
  recTimerId = setInterval(() => {
    const d = recorder.durationSec;
    $('recTimer').textContent = fmtTimer(d);
  }, 100);
  drawLiveWave();
}

async function stopRecording() {
  clearInterval(recTimerId);
  cancelAnimationFrame(liveDrawId);
  $('recordBtn').classList.remove('recording');
  $('recTimer').classList.remove('on');
  const { blob, durationSec } = await recorder.stop();
  recorder = null;
  clearLiveWave();
  if (durationSec < 0.5) {
    showError('녹음이 너무 짧습니다 (0.5초 이상 녹음해주세요).');
    return;
  }
  analyze(blob, 'mic_recording.wav', '🎤');
}

function drawLiveWave() {
  const canvas = $('liveWave');
  const ctx = canvas.getContext('2d');
  const analyser = recorder.analyser;
  const buf = new Float32Array(analyser.fftSize);
  const draw = () => {
    if (!recorder || !recorder.recording) return;
    liveDrawId = requestAnimationFrame(draw);
    analyser.getFloatTimeDomainData(buf);
    ctx.fillStyle = '#1c2129';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.strokeStyle = '#f85149';
    ctx.lineWidth = 2;
    const mid = canvas.height / 2;
    for (let i = 0; i < buf.length; i++) {
      const x = (i / buf.length) * canvas.width;
      const y = mid + buf[i] * mid * 1.6;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  draw();
}

function clearLiveWave() {
  const canvas = $('liveWave');
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#1c2129';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function fmtTimer(sec) {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${String(m).padStart(2, '0')}:${s < 10 ? '0' : ''}${s.toFixed(1)}`;
}

/* ── 파일 업로드 ────────────────────────────────────── */

const dz = $('dropzone');
dz.addEventListener('click', () => $('fileInput').click());
dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', (e) => {
  e.preventDefault();
  dz.classList.remove('drag');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
$('fileInput').addEventListener('change', (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
  e.target.value = '';
});

function handleFile(file) {
  if (file.size > 60 * 1024 * 1024) {
    showError('파일이 너무 큽니다 (최대 60MB).');
    return;
  }
  analyze(file, file.name, '📁');
}

/* ── 분석 요청 ──────────────────────────────────────── */

async function analyze(blob, filename, sourceIcon) {
  const compare = $('compareChk').checked;
  const modelId = $('modelSelect').value;

  $('statusCard').hidden = false;
  $('statusInner').classList.remove('error');
  $('statusInner').querySelector('.spinner').style.display = '';
  $('statusText').textContent = compare ? '6개 모델로 분석 중…' : '분석 중…';

  const fd = new FormData();
  fd.append('file', blob, filename);
  fd.append('model_id', modelId);
  fd.append('compare', compare);

  let data;
  try {
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    data = await res.json();
    if (!res.ok) throw new Error(data.detail || `서버 오류 (${res.status})`);
  } catch (e) {
    showError(`분석 실패: ${e.message}`);
    return;
  }

  $('statusCard').hidden = true;
  renderResult(data, blob, filename, sourceIcon);
}

function showError(msg) {
  $('statusCard').hidden = false;
  $('statusInner').classList.add('error');
  $('statusInner').querySelector('.spinner').style.display = 'none';
  $('statusText').textContent = msg;
}

/* ── 결과 렌더링 ────────────────────────────────────── */

function renderResult(data, blob, filename, sourceIcon) {
  const main = data.results[0];
  const audio = data.audio;

  // 경고
  const warns = (audio.warnings || []).map((w) => WARN_MSG[w]).filter(Boolean);
  if (audio.truncated) warns.push('5분 초과분은 잘라내고 앞부분만 분석했습니다.');
  $('warnBox').hidden = warns.length === 0;
  $('warnBox').textContent = warns.join(' ');

  // 판정 배너
  const pFake = main.prob_fake;
  const uncertain = pFake >= 0.4 && pFake <= 0.6;
  const banner = $('verdictBanner');
  banner.className = 'verdict ' + (uncertain ? 'uncertain' : main.verdict);
  if (uncertain) {
    $('verdictIcon').textContent = '⚠️';
    $('verdictLabel').textContent = '판별 불확실';
    $('verdictConf').textContent = `P(fake) ${(pFake * 100).toFixed(1)}% — 임계값(0.5) 부근이라 확신이 낮습니다`;
  } else if (main.verdict === 'fake') {
    $('verdictIcon').textContent = '🚨';
    $('verdictLabel').textContent = '딥페이크 의심 (FAKE)';
    $('verdictConf').textContent = `확신도 ${(pFake * 100).toFixed(1)}% · ${main.model_name}`;
  } else {
    $('verdictIcon').textContent = '✅';
    $('verdictLabel').textContent = '진짜 음성 판정 (REAL)';
    $('verdictConf').textContent = `확신도 ${((1 - pFake) * 100).toFixed(1)}% · ${main.model_name}`;
  }

  // 확률 바
  $('fakePart').style.width = `${(pFake * 100).toFixed(1)}%`;
  $('fakePct').textContent = `${(pFake * 100).toFixed(1)}%`;
  $('realPct').textContent = `${((1 - pFake) * 100).toFixed(1)}%`;

  // 메타 칩
  const chips = [
    `길이 <b>${audio.duration_sec}초</b>`,
    `원본 <b>${(audio.original_sr / 1000).toFixed(1)}kHz</b> → 16kHz`,
    `윈도우 <b>${main.num_windows}개</b> × ${main.window_sec}초`,
    `추론 <b>${main.inference_ms}ms</b> (${main.device.toUpperCase()})`,
    `모델 <b>${main.model_name}</b> · ${main.params}`,
  ];
  if (main.t2_eer != null) chips.push(`T2 EER <b>${main.t2_eer}%</b>`);
  $('metaChips').innerHTML = chips.map((c) => `<div class="chip">${c}</div>`).join('');

  // 오디오 플레이어
  if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
  currentBlobUrl = URL.createObjectURL(blob);
  $('player').src = currentBlobUrl;

  // 파형 + 구간
  $('segTitle').textContent = `— ${main.window_sec}초 윈도우 × ${main.num_windows}개, 50% overlap`;
  drawWaveform(audio.waveform, audio.duration_sec, main.windows);
  renderSegments(main.windows);

  // 스펙트로그램
  $('specImg').src = audio.spectrogram;
  $('specCaption').textContent = `가로: 0 – ${audio.duration_sec}초 · 세로: 0 – 8kHz (아래가 저주파)`;

  // 비교 테이블
  $('compareWrap').hidden = !data.compare;
  if (data.compare) renderCompare(data.results);

  $('resultCard').hidden = false;
  $('resultCard').scrollIntoView({ behavior: 'smooth', block: 'start' });

  addHistory(sourceIcon, filename, main, data.compare);
}

function drawWaveform(env, durationSec, windows) {
  const canvas = $('waveCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const heatH = 26;              // 하단 heat strip 높이
  const waveH = H - heatH - 8;
  ctx.fillStyle = '#1c2129';
  ctx.fillRect(0, 0, W, H);

  // 초 단위 heat strip: 해당 초를 덮는 윈도우들의 평균 P(fake)
  const secs = Math.max(1, Math.ceil(durationSec));
  for (let s = 0; s < secs; s++) {
    const c = s + 0.5;
    const cover = windows.filter((w) => w.start <= c && c < w.end);
    const p = cover.length
      ? cover.reduce((a, w) => a + w.prob_fake, 0) / cover.length
      : windows[windows.length - 1].prob_fake;
    const x0 = (s / durationSec) * W;
    const x1 = (Math.min(s + 1, durationSec) / durationSec) * W;
    ctx.fillStyle = heatColor(p);
    ctx.fillRect(x0, waveH + 8, x1 - x0, heatH);
  }

  // 파형
  const n = env.max.length;
  const mid = waveH / 2;
  ctx.fillStyle = '#4d9fff';
  for (let i = 0; i < n; i++) {
    const x = (i / n) * W;
    const w = Math.max(1, W / n);
    const y1 = mid - env.max[i] * mid * 0.92;
    const y2 = mid - env.min[i] * mid * 0.92;
    ctx.fillRect(x, y1, w, Math.max(1, y2 - y1));
  }

  // 시간 눈금
  ctx.fillStyle = '#9aa4b2';
  ctx.font = '20px sans-serif';
  const step = durationSec > 60 ? 10 : durationSec > 20 ? 5 : 1;
  for (let s = 0; s <= durationSec; s += step) {
    const x = (s / durationSec) * W;
    ctx.fillRect(x, waveH - 6, 1, 6);
    if (x < W - 30) ctx.fillText(`${s}s`, x + 3, waveH - 10);
  }
}

function heatColor(p) {
  // green(real) → red(fake)
  const r = Math.round(46 + (248 - 46) * p);
  const g = Math.round(160 + (81 - 160) * p);
  const b = Math.round(67 + (73 - 67) * p);
  return `rgba(${r},${g},${b},0.85)`;
}

function renderSegments(windows) {
  $('segList').innerHTML = windows
    .map((w) => {
      const cls = w.prob_fake >= 0.5 ? 'fake' : 'real';
      return `<span class="seg ${cls}">${w.start}–${w.end}s · ${(w.prob_fake * 100).toFixed(1)}%</span>`;
    })
    .join('');
}

function renderCompare(results) {
  const rows = results.map((r) => {
    const eer = r.t2_eer != null ? `${r.t2_eer}%` : '—';
    return `<tr>
      <td>${r.model_name}</td>
      <td>${r.params}</td>
      <td>${eer}</td>
      <td><span class="badge ${r.verdict}">${r.verdict.toUpperCase()}</span></td>
      <td>${(r.prob_fake * 100).toFixed(1)}%</td>
      <td><div class="mini-bar"><div style="width:${(r.prob_fake * 100).toFixed(1)}%"></div></div></td>
      <td>${r.inference_ms}ms</td>
    </tr>`;
  });
  $('compareBody').innerHTML = rows.join('');
}

/* ── 기록 ───────────────────────────────────────────── */

function addHistory(icon, filename, main, compared) {
  history.unshift({
    time: new Date().toLocaleTimeString('ko-KR', { hour12: false }),
    icon, filename,
    model: main.model_name + (compared ? ' 외' : ''),
    verdict: main.verdict,
    pFake: main.prob_fake,
  });
  if (history.length > 30) history.pop();
  $('historyCard').hidden = false;
  $('histList').innerHTML = history
    .map(
      (h) => `<div class="hist-item">
        <span class="t">${h.time}</span>
        <span class="src">${h.icon}</span>
        <span class="name">${h.filename} · ${h.model}</span>
        <span class="badge ${h.verdict}">${h.verdict.toUpperCase()} ${((h.verdict === 'fake' ? h.pFake : 1 - h.pFake) * 100).toFixed(0)}%</span>
      </div>`
    )
    .join('');
}

$('clearHist').addEventListener('click', () => {
  history = [];
  $('historyCard').hidden = true;
});

/* ── 실시간 감시 (WebSocket 스트리밍) ───────────────────── */

const CHART_SPAN_SEC = 60;   // 차트에 보여줄 최근 구간
const EMA_ALPHA = 0.35;      // 판정 스무딩 계수

let stream = null; // {ws, rec, sendBuf, sendLen, scores:[{t,p}], ema, startedAt, lastInferMs}

$('streamBtn').addEventListener('click', () => (stream ? stopStream() : startStream()));

async function startStream() {
  const modelId = $('streamModelSelect').value;
  if (!modelId) return;
  const btn = $('streamBtn');
  btn.disabled = true;
  btn.textContent = '⏳ 연결 중…';

  const rec = new PCMRecorder({ sampleRate: 16000, onChunk: onStreamChunk });
  try {
    await rec.start();
  } catch (e) {
    btn.disabled = false;
    btn.textContent = '▶ 감시 시작';
    setLiveState('idle', '마이크 오류',
      e.name === 'NotAllowedError' ? '마이크 권한이 거부되었습니다' : e.message);
    return;
  }

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
  ws.binaryType = 'arraybuffer';
  stream = { ws, rec, sendBuf: [], sendLen: 0, scores: [], ema: null, startedAt: null, lastInferMs: null };

  ws.onopen = () => ws.send(JSON.stringify({
    type: 'start', sample_rate: rec.sampleRate, model_id: modelId,
  }));
  ws.onmessage = (e) => handleStreamMsg(JSON.parse(e.data));
  ws.onerror = () => teardownStream('서버 연결 오류');
  ws.onclose = () => { if (stream) teardownStream(); };
}

function handleStreamMsg(msg) {
  if (!stream) return;
  if (msg.type === 'ready') {
    stream.startedAt = performance.now();
    $('streamBtn').disabled = false;
    $('streamBtn').textContent = '⏹ 감시 중지';
    $('streamBtn').classList.add('on');
    $('streamModelSelect').disabled = true;
    setLiveState('idle', '듣는 중…', `${msg.model_name} · ${msg.window_sec}초 윈도우, ${msg.hop_sec}초 간격 판정`);
    $('livePct').textContent = '—';
  } else if (msg.type === 'score') {
    stream.scores.push({ t: msg.t, p: msg.prob_fake });
    stream.lastInferMs = msg.inference_ms;
    stream.ema = stream.ema == null ? msg.prob_fake : EMA_ALPHA * msg.prob_fake + (1 - EMA_ALPHA) * stream.ema;
    updateLiveVerdict(msg);
    drawStreamChart();
    updateStreamStats();
  } else if (msg.type === 'summary') {
    showStreamSummary(msg);
  } else if (msg.type === 'error') {
    teardownStream(msg.message);
  }
}

function onStreamChunk(chunk) {
  if (!stream || stream.ws.readyState !== WebSocket.OPEN) return;
  stream.sendBuf.push(chunk);
  stream.sendLen += chunk.length;
  // 레벨 미터
  let sum = 0;
  for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
  const rms = Math.sqrt(sum / chunk.length);
  $('lvlBar').style.width = `${Math.min(100, rms * 320)}%`;
  // ~0.25초 단위로 묶어 전송 (프레임 오버헤드 감소)
  if (stream.sendLen >= stream.rec.sampleRate / 4) {
    const merged = new Float32Array(stream.sendLen);
    let off = 0;
    for (const c of stream.sendBuf) { merged.set(c, off); off += c.length; }
    stream.sendBuf = [];
    stream.sendLen = 0;
    stream.ws.send(merged.buffer);
  }
}

function updateLiveVerdict(msg) {
  const ema = stream.ema;
  const state = ema >= 0.6 ? 'fake' : ema <= 0.4 ? 'real' : 'uncertain';
  const label = state === 'fake' ? '🚨 딥페이크 의심' : state === 'real' ? '✅ 진짜 음성' : '⚠️ 판별 불확실';
  setLiveState(state, label,
    `순간 P(fake) ${(msg.prob_fake * 100).toFixed(0)}% · 추론 ${msg.inference_ms}ms · ${msg.t.toFixed(1)}초 경과`);
  $('livePct').textContent = `${(ema * 100).toFixed(0)}%`;
}

function setLiveState(state, label, sub) {
  const ind = $('liveIndicator');
  ind.className = `live-indicator ${state}`;
  $('liveIcon').textContent = state === 'fake' ? '🚨' : state === 'real' ? '✅' : state === 'uncertain' ? '⚠️' : '📡';
  $('liveLabel').textContent = label;
  $('liveSub').textContent = sub;
}

function updateStreamStats() {
  const ps = stream.scores.map((s) => s.p);
  const mean = ps.reduce((a, b) => a + b, 0) / ps.length;
  const chips = [
    `경과 <b>${stream.scores[stream.scores.length - 1].t.toFixed(1)}초</b>`,
    `판정 <b>${ps.length}회</b>`,
    `EMA <b>${(stream.ema * 100).toFixed(1)}%</b>`,
    `평균 <b>${(mean * 100).toFixed(1)}%</b>`,
    `최대 <b>${(Math.max(...ps) * 100).toFixed(1)}%</b>`,
  ];
  $('streamStats').innerHTML = chips.map((c) => `<div class="chip">${c}</div>`).join('');
}

function drawStreamChart() {
  const canvas = $('streamChart');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const padB = 34, padT = 10;
  const plotH = H - padB - padT;
  ctx.fillStyle = '#1c2129';
  ctx.fillRect(0, 0, W, H);

  const yOf = (p) => padT + (1 - p) * plotH;

  // 위험 영역(>0.5) tint + 판정선
  ctx.fillStyle = 'rgba(248,81,73,0.06)';
  ctx.fillRect(0, padT, W, yOf(0.5) - padT);
  ctx.strokeStyle = 'rgba(210,153,34,0.7)';
  ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.moveTo(0, yOf(0.5));
  ctx.lineTo(W, yOf(0.5));
  ctx.stroke();
  ctx.setLineDash([]);

  // y축 라벨
  ctx.fillStyle = '#9aa4b2';
  ctx.font = '20px sans-serif';
  ctx.fillText('FAKE 1.0', 8, yOf(1) + 20);
  ctx.fillText('0.5', 8, yOf(0.5) - 8);
  ctx.fillText('REAL 0.0', 8, yOf(0) - 6);

  const scores = stream ? stream.scores : [];
  if (!scores.length) return;

  const tEnd = Math.max(scores[scores.length - 1].t, CHART_SPAN_SEC * 0.2);
  const tStart = Math.max(0, tEnd - CHART_SPAN_SEC);
  const xOf = (t) => ((t - tStart) / (tEnd - tStart || 1)) * (W - 20) + 10;
  const visible = scores.filter((s) => s.t >= tStart);

  // 순간값 라인 (점수별 색)
  for (let i = 1; i < visible.length; i++) {
    const a = visible[i - 1], b = visible[i];
    ctx.strokeStyle = heatColor((a.p + b.p) / 2);
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(xOf(a.t), yOf(a.p));
    ctx.lineTo(xOf(b.t), yOf(b.p));
    ctx.stroke();
  }
  // 포인트
  for (const s of visible) {
    ctx.fillStyle = heatColor(s.p);
    ctx.beginPath();
    ctx.arc(xOf(s.t), yOf(s.p), 4, 0, Math.PI * 2);
    ctx.fill();
  }

  // x축 시간 눈금
  ctx.fillStyle = '#9aa4b2';
  const span = tEnd - tStart;
  const step = span > 40 ? 10 : span > 15 ? 5 : 2;
  for (let t = Math.ceil(tStart / step) * step; t <= tEnd; t += step) {
    ctx.fillText(`${t}s`, xOf(t) - 10, H - 8);
  }
}

function stopStream() {
  if (!stream) return;
  $('streamBtn').disabled = true;
  $('streamBtn').textContent = '⏳ 종료 중…';
  stream.rec.stop().catch(() => {});
  if (stream.ws.readyState === WebSocket.OPEN) {
    stream.ws.send(JSON.stringify({ type: 'stop' }));
    // summary 수신 → showStreamSummary → teardown. 2초 내 미수신 시 강제 종료.
    setTimeout(() => { if (stream) teardownStream(); }, 2000);
  } else {
    teardownStream();
  }
}

function showStreamSummary(msg) {
  const verdict = msg.mean != null && msg.mean >= 0.5 ? 'fake' : 'real';
  if (msg.num_scores > 0) {
    addHistory('📡', `실시간 감시 (${msg.duration_sec}초)`, {
      model_name: $('streamModelSelect').selectedOptions[0]?.textContent || '2초 모델',
      verdict, prob_fake: msg.mean,
    }, false);
    setLiveState('idle', '감시 종료',
      `${msg.duration_sec}초 · 판정 ${msg.num_scores}회 · 평균 P(fake) ${(msg.mean * 100).toFixed(1)}% · 최대 ${(msg.max * 100).toFixed(1)}%`);
  } else {
    setLiveState('idle', '감시 종료', '판정된 윈도우가 없습니다 (2초 이상 말해야 첫 판정이 나옵니다)');
  }
  teardownStream(null, true);
}

function teardownStream(errMsg, keepLabel = false) {
  if (!stream) return;
  const s = stream;
  stream = null;
  s.rec.stop().catch(() => {});
  try { s.ws.close(); } catch (e) { /* already closed */ }
  $('streamBtn').disabled = false;
  $('streamBtn').textContent = '▶ 감시 시작';
  $('streamBtn').classList.remove('on');
  $('streamModelSelect').disabled = false;
  $('lvlBar').style.width = '0%';
  $('livePct').textContent = '—';
  if (errMsg) setLiveState('idle', '연결 종료', errMsg);
  else if (!keepLabel) setLiveState('idle', '대기 중', '감시를 시작하면 말하는 동안 0.5초마다 판정합니다');
}

init();
