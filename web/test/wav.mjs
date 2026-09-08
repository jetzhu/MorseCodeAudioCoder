/**
 * Minimal WAV reader for the Node tests.
 *
 * Parses RIFF/WAVE files with PCM samples (8, 16, 24 or 32-bit integers) or
 * IEEE float samples (32 or 64-bit), including the WAVE_FORMAT_EXTENSIBLE
 * wrapper, and returns channel 0 as `Float32Array` in -1..1.  The scaling
 * matches `morse.pipeline.load_wav`: integers are divided by their full scale
 * (32768 for 16-bit, 2**31 for 24- and 32-bit; 8-bit is unsigned and centred
 * on 128), floats are passed through, and the result is stored as float32.
 *
 * Extra chunks (LIST, fact, ...) are skipped.  Nothing is resampled.
 *
 * @module wav
 */

import { readFileSync } from "node:fs";

const WAVE_FORMAT_PCM = 1;
const WAVE_FORMAT_IEEE_FLOAT = 3;
const WAVE_FORMAT_EXTENSIBLE = 0xfffe;

/**
 * @typedef {object} WavData
 * @property {number} sampleRate frames per second
 * @property {number} channels number of interleaved channels in the file
 * @property {number} bitsPerSample container width of one sample
 * @property {"pcm"|"float"} format sample encoding
 * @property {number} frames number of sample frames
 * @property {Float32Array} samples channel 0, scaled to -1..1
 */

/**
 * Four ASCII bytes at `offset` as a string.
 * @param {DataView} view
 * @param {number} offset
 * @returns {string}
 */
function tag(view, offset) {
  return String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3),
  );
}

/**
 * Parse a WAV file held in memory.
 *
 * @param {ArrayBuffer|ArrayBufferView} bytes the whole file
 * @returns {WavData}
 * @throws {Error} on a malformed header or an unsupported sample format
 */
export function parseWav(bytes) {
  const view =
    bytes instanceof ArrayBuffer
      ? new DataView(bytes)
      : new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.byteLength < 12 || tag(view, 0) !== "RIFF" || tag(view, 8) !== "WAVE") {
    throw new Error("not a RIFF/WAVE file");
  }

  let fmt = null;
  let dataOffset = -1;
  let dataSize = 0;
  let offset = 12;
  while (offset + 8 <= view.byteLength) {
    const id = tag(view, offset);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === "fmt ") {
      let audioFormat = view.getUint16(body, true);
      if (audioFormat === WAVE_FORMAT_EXTENSIBLE && size >= 26) {
        audioFormat = view.getUint16(body + 24, true); // first two bytes of the sub-format GUID
      }
      fmt = {
        audioFormat,
        channels: view.getUint16(body + 2, true),
        sampleRate: view.getUint32(body + 4, true),
        blockAlign: view.getUint16(body + 12, true),
        bitsPerSample: view.getUint16(body + 14, true),
      };
    } else if (id === "data") {
      dataOffset = body;
      dataSize = Math.min(size, view.byteLength - body); // tolerate a truncated file
      break; // chunks after the data are irrelevant here
    }
    offset = body + size + (size & 1); // chunks are word-aligned
  }
  if (fmt === null) throw new Error("WAV file has no fmt chunk");
  if (dataOffset < 0) throw new Error("WAV file has no data chunk");
  if (fmt.channels < 1 || fmt.blockAlign < 1) throw new Error("WAV fmt chunk is invalid");

  const { audioFormat, channels, sampleRate, blockAlign, bitsPerSample } = fmt;
  const frames = Math.floor(dataSize / blockAlign);
  const samples = new Float32Array(frames);
  const isFloat = audioFormat === WAVE_FORMAT_IEEE_FLOAT;
  if (!isFloat && audioFormat !== WAVE_FORMAT_PCM) {
    throw new Error("unsupported WAV audio format " + audioFormat);
  }

  const key = (isFloat ? "float" : "pcm") + bitsPerSample;
  switch (key) {
    case "pcm8":
      for (let i = 0, o = dataOffset; i < frames; i++, o += blockAlign) {
        samples[i] = (view.getUint8(o) - 128) / 128;
      }
      break;
    case "pcm16":
      for (let i = 0, o = dataOffset; i < frames; i++, o += blockAlign) {
        samples[i] = view.getInt16(o, true) / 32768;
      }
      break;
    case "pcm24":
      for (let i = 0, o = dataOffset; i < frames; i++, o += blockAlign) {
        // Little-endian 24-bit into the top of a signed 32-bit word, like scipy.
        const v = (view.getUint8(o) << 8) | (view.getUint8(o + 1) << 16) | (view.getUint8(o + 2) << 24);
        samples[i] = v / 2147483648;
      }
      break;
    case "pcm32":
      for (let i = 0, o = dataOffset; i < frames; i++, o += blockAlign) {
        samples[i] = view.getInt32(o, true) / 2147483648;
      }
      break;
    case "float32":
      for (let i = 0, o = dataOffset; i < frames; i++, o += blockAlign) {
        samples[i] = view.getFloat32(o, true);
      }
      break;
    case "float64":
      for (let i = 0, o = dataOffset; i < frames; i++, o += blockAlign) {
        samples[i] = view.getFloat64(o, true);
      }
      break;
    default:
      throw new Error("unsupported WAV sample size " + bitsPerSample + "-bit " + (isFloat ? "float" : "PCM"));
  }

  return {
    sampleRate,
    channels,
    bitsPerSample,
    format: isFloat ? "float" : "pcm",
    frames,
    samples,
  };
}

/**
 * Read and parse a WAV file from disk.
 *
 * @param {string|URL} path
 * @returns {WavData}
 */
export function readWav(path) {
  return parseWav(readFileSync(path));
}
