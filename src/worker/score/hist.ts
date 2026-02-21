const H_BINS = 16;
const S_BINS = 4;
const V_BINS = 4;
const TOTAL_BINS = H_BINS * S_BINS * V_BINS;

function rgbToHsv(r: number, g: number, b: number): { h: number; s: number; v: number } {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;

  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;

  let h = 0;
  if (delta !== 0) {
    if (max === rn) {
      h = 60 * (((gn - bn) / delta) % 6);
    } else if (max === gn) {
      h = 60 * ((bn - rn) / delta + 2);
    } else {
      h = 60 * ((rn - gn) / delta + 4);
    }
  }
  if (h < 0) {
    h += 360;
  }

  const s = max === 0 ? 0 : delta / max;
  const v = max;

  return { h, s, v };
}

function clampIndex(value: number, maxExclusive: number): number {
  return Math.max(0, Math.min(maxExclusive - 1, value));
}

export function buildHsvHistogram(rgb: Uint8Array): Float64Array {
  if (rgb.length % 3 !== 0) {
    throw new Error("rgb buffer must have 3 channels");
  }

  const hist = new Float64Array(TOTAL_BINS);
  const pixelCount = rgb.length / 3;

  for (let i = 0; i < rgb.length; i += 3) {
    const { h, s, v } = rgbToHsv(rgb[i], rgb[i + 1], rgb[i + 2]);
    const hBin = clampIndex(Math.floor((h / 360) * H_BINS), H_BINS);
    const sBin = clampIndex(Math.floor(s * S_BINS), S_BINS);
    const vBin = clampIndex(Math.floor(v * V_BINS), V_BINS);
    const idx = hBin * S_BINS * V_BINS + sBin * V_BINS + vBin;
    hist[idx] += 1;
  }

  if (pixelCount === 0) {
    return hist;
  }

  for (let i = 0; i < hist.length; i += 1) {
    hist[i] /= pixelCount;
  }

  return hist;
}

export function histogramIntersection(a: Float64Array, b: Float64Array): number {
  if (a.length !== b.length) {
    throw new Error("histogram size mismatch");
  }

  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += Math.min(a[i], b[i]);
  }
  return Math.max(0, Math.min(1, sum));
}
