/**
 * 마이크 PCM 녹음기.
 *
 * MediaRecorder(webm/opus) 대신 AudioWorklet으로 raw PCM을 캡처해
 * 무손실 WAV(16-bit PCM)로 인코딩한다. opus 압축 아티팩트가
 * 탐지 모델 입력을 오염시키지 않도록 하기 위함.
 * echoCancellation/noiseSuppression/autoGainControl도 끈다
 * (브라우저 DSP가 스푸핑 아티팩트를 뭉갤 수 있음).
 */

const WORKLET_CODE = `
class PCMCapture extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch && ch.length) this.port.postMessage(new Float32Array(ch));
    return true;
  }
}
registerProcessor('pcm-capture', PCMCapture);
`;

class PCMRecorder {
  /**
   * @param {Object} [opts]
   * @param {number} [opts.sampleRate] AudioContext 목표 샘플레이트 (예: 16000).
   *   브라우저가 거부하면 native rate로 폴백 — 실제값은 this.sampleRate 확인.
   * @param {(chunk: Float32Array) => void} [opts.onChunk] 스트리밍 콜백.
   *   지정하면 chunks에 쌓지 않고 콜백으로만 전달한다.
   */
  constructor(opts = {}) {
    this.opts = opts;
    this.stream = null;
    this.ctx = null;
    this.node = null;
    this.analyser = null;
    this.chunks = [];
    this.sampleRate = 0;
    this.recording = false;
  }

  async start() {
    this.chunks = [];
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    try {
      this.ctx = this.opts.sampleRate
        ? new AudioContext({ sampleRate: this.opts.sampleRate })
        : new AudioContext();
    } catch (e) {
      this.ctx = new AudioContext(); // 목표 샘플레이트 미지원 브라우저 폴백
    }
    this.sampleRate = this.ctx.sampleRate;

    const blobUrl = URL.createObjectURL(
      new Blob([WORKLET_CODE], { type: 'application/javascript' })
    );
    await this.ctx.audioWorklet.addModule(blobUrl);
    URL.revokeObjectURL(blobUrl);

    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, 'pcm-capture');
    this.node.port.onmessage = (e) => {
      if (!this.recording) return;
      if (this.opts.onChunk) this.opts.onChunk(e.data);
      else this.chunks.push(e.data);
    };
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    src.connect(this.node);
    src.connect(this.analyser);
    this.recording = true;
  }

  /** 녹음 종료 → WAV Blob 반환 */
  async stop() {
    this.recording = false;
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.ctx) await this.ctx.close().catch(() => {});

    const total = this.chunks.reduce((n, c) => n + c.length, 0);
    const pcm = new Float32Array(total);
    let off = 0;
    for (const c of this.chunks) {
      pcm.set(c, off);
      off += c.length;
    }
    this.chunks = [];
    return {
      blob: encodeWav(pcm, this.sampleRate),
      durationSec: total / this.sampleRate,
    };
  }

  get durationSec() {
    const total = this.chunks.reduce((n, c) => n + c.length, 0);
    return this.sampleRate ? total / this.sampleRate : 0;
  }
}

/** Float32 PCM → 16-bit PCM WAV Blob */
function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (o, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);        // fmt chunk size
  view.setUint16(20, 1, true);         // PCM
  view.setUint16(22, 1, true);         // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);  // byte rate
  view.setUint16(32, 2, true);         // block align
  view.setUint16(34, 16, true);        // bits per sample
  writeStr(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}

window.PCMRecorder = PCMRecorder;
