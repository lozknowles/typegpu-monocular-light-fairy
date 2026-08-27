import { d, tgpu } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { DepthCameraSession } from './camera-session.ts';
import { SourceChoice, SourceChooser } from './chooser.ts';
import { parseDepthBundle } from './inference/bundle.ts';
import { DepthInferencePlan } from './inference/depthart.ts';
import { setupLightInput } from './light-input.ts';
import {
  MODEL_REVISION,
  fetchModel,
  modelLabel,
  type ModelSize,
  type ModelVariant,
} from './model-store.ts';
import {
  DepthRelightingRenderer,
  defaultRelightingSettings,
  type PixelDiagnostics,
  type RenderTiming,
} from './renderer.ts';

const CAMERA_FRAME_RATE = 60;
const LIVE_TIMING_SAMPLE_COUNT = 5;
const FAIRY_MAX_ROLL_DEGREES = 60;
const FAIRY_MAX_PITCH_DEGREES = 39;
const DEMO_IMAGE_URL = '/demo.jpg';
const FACING_MODES = ['front', 'back'] as const;
const VIEW_OUTPUT_LABELS = [
  'Relit image output',
  'Camera image output',
  'Relative disparity image output',
  'Normals image output',
] as const;
const URL_OPTIONS = new URLSearchParams(location.search);
const AUTORUN = URL_OPTIONS.get('autorun') === '1';
const BENCHMARK_SAMPLE_COUNT = URL_OPTIONS.get('benchmark') === '1' ? 7 : 5;
const REQUESTED_TIMING_MODE = URL_OPTIONS.get('timing');
const TIMING_MODE =
  REQUESTED_TIMING_MODE === 'gpu' || REQUESTED_TIMING_MODE === 'wall'
    ? REQUESTED_TIMING_MODE
    : 'auto';
const ADAPTER_POWER_PREFERENCE =
  URL_OPTIONS.get('adapter') === 'low-power' ? 'low-power' : 'high-performance';

interface AdapterDiagnostics {
  readonly architecture: string;
  readonly description: string;
  readonly device: string;
  readonly powerPreference: GPUPowerPreference;
  readonly requestAttempts: number;
  readonly vendor: string;
}

interface TimingSummary {
  readonly wallMs: number;
  readonly gpuComputeMs?: number;
  readonly gpuFrameMs?: number;
  readonly gpuRenderMs?: number;
}

interface WarmTimingSummary extends TimingSummary {
  readonly samples: number;
}

interface QualificationRecord {
  readonly schemaVersion: 1;
  status: 'pending' | 'complete' | 'failed';
  capturedAt?: string;
  browser: {
    readonly userAgent: string;
    readonly brands?: readonly string[];
    readonly mobile?: boolean;
    readonly platform: string;
  };
  secureContext: boolean;
  webgpu: boolean;
  adapter?: AdapterDiagnostics;
  features: {
    shaderF16: boolean;
    timestampQuery: boolean;
    timestampQueryEnabled: boolean;
    timestampQueryNote?: string;
  };
  model?: {
    readonly bundle: string;
    readonly bytes: number;
    readonly label: string;
    readonly precision: ModelVariant['precision'];
    readonly revision: string;
    readonly sha256: string;
  };
  timings?: {
    modelLoadAndVerifyMs: number;
    modelParseMs: number;
    pipelineCompileMs: number;
    cold: TimingSummary;
    warm: WarmTimingSummary;
    source: 'gpu-timestamp-query' | 'wall-clock-approximation';
    gpuTimestampNote?: string;
  };
  frames: {
    dropped: number;
    fps?: number;
  };
  fairyPose: {
    headingRadians: number;
    pitchNormalized: number;
    rollNormalized: number;
  };
  outputs?: Record<'camera' | 'normals' | 'relativeDisparity' | 'relit', PixelDiagnostics>;
  error?: string;
}

declare global {
  interface Window {
    __TYPEGPU_QUALIFICATION__?: QualificationRecord;
  }
}

type NavigatorWithUAData = Navigator & {
  readonly userAgentData?: {
    readonly brands: readonly { readonly brand: string; readonly version: string }[];
    readonly mobile: boolean;
    readonly platform: string;
  };
};

type AdapterWithInfo = GPUAdapter & {
  readonly info?: GPUAdapterInfo;
  requestAdapterInfo?: () => Promise<GPUAdapterInfo>;
};

function element<T extends Element>(selector: string): T {
  const match = document.querySelector(selector);
  if (!match) {
    throw new Error(`Required element is missing: ${selector}`);
  }
  return match as T;
}

const canvas = element<HTMLCanvasElement>('canvas');
const video = element<HTMLVideoElement>('video');
const status = element<HTMLDivElement>('.status');
const statusMessage = element<HTMLParagraphElement>('.status-message');
const unsupported = element<HTMLDivElement>('.unsupported');
const unsupportedMessage = element<HTMLParagraphElement>('.unsupported-message');
const qualificationBadge = element<HTMLSpanElement>('#qualification-badge');
const listenerController = new AbortController();

const uaData = (navigator as NavigatorWithUAData).userAgentData;
const qualification: QualificationRecord = {
  schemaVersion: 1,
  status: 'pending',
  browser: {
    userAgent: navigator.userAgent,
    brands: uaData?.brands.map(({ brand, version }) => `${brand} ${version}`),
    mobile: uaData?.mobile,
    platform: uaData?.platform ?? navigator.platform,
  },
  secureContext: globalThis.isSecureContext,
  webgpu: navigator.gpu !== undefined,
  features: { shaderF16: false, timestampQuery: false, timestampQueryEnabled: false },
  frames: { dropped: 0 },
  fairyPose: {
    headingRadians: defaultRelightingSettings.fairyHeading,
    pitchNormalized: defaultRelightingSettings.fairyPitch,
    rollNormalized: defaultRelightingSettings.fairyBank,
  },
};
window.__TYPEGPU_QUALIFICATION__ = qualification;

let root: TgpuRoot | undefined;
let plan: DepthInferencePlan | undefined;
let renderer: DepthRelightingRenderer | undefined;
let chooser: SourceChooser | undefined;
let disposed = false;
let deviceLost = false;
let currentBundle: string | undefined;
let demoImage: ImageBitmap | undefined;
let staticLoopGeneration = 0;
let depthDirty = true;
let liveRenderGeneration = 0;
let liveRenderInFlight = false;
let liveTimingSamples: RenderTiming[] = [];
let fairyEnabled = true;

function setDiagnostic(id: string, text: string): void {
  element<HTMLElement>(`#${id}`).textContent = text;
}

function browserLabel(): string {
  if (uaData?.brands.length) {
    return `${uaData.brands.map(({ brand, version }) => `${brand} ${version}`).join(', ')} · ${uaData.platform}`;
  }
  return navigator.userAgent;
}

setDiagnostic('diag-browser', browserLabel());
setDiagnostic(
  'diag-secure-context',
  globalThis.isSecureContext ? 'Yes — camera APIs permitted' : 'No — camera APIs blocked',
);
setDiagnostic('diag-model-revision', MODEL_REVISION);

function syncQualification(): void {
  window.__TYPEGPU_QUALIFICATION__ = qualification;
}

function setQualificationState(state: QualificationRecord['status'], error?: string): void {
  qualification.status = state;
  qualification.capturedAt = new Date().toISOString();
  qualification.error = error;
  document.documentElement.dataset.qualification = state;
  qualificationBadge.dataset.state = state;
  qualificationBadge.textContent =
    state === 'complete' ? 'Qualified' : state === 'failed' ? 'Failed' : 'Pending';
  syncQualification();
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function setStatus(tone: 'busy' | 'error', message: string): void {
  status.dataset.tone = tone;
  status.hidden = false;
  statusMessage.textContent = message;
}

function clearTransientStatus(): void {
  if (status.dataset.tone === 'busy') {
    status.hidden = true;
  }
}

function showUnsupported(message: string): void {
  status.hidden = true;
  element<HTMLDivElement>('.chooser').hidden = true;
  unsupportedMessage.textContent = message;
  unsupported.hidden = false;
  setDiagnostic('diag-webgpu', 'Unavailable');
  setQualificationState('failed', message);
}

function median(values: readonly number[]): number {
  if (values.length === 0) {
    throw new Error('A median requires at least one timing sample.');
  }
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)] ?? sorted[0] ?? 0;
}

function compactTiming(timing: RenderTiming): TimingSummary {
  return {
    wallMs: timing.wallMs,
    gpuComputeMs: timing.gpuComputeMs,
    gpuFrameMs: timing.gpuFrameMs,
    gpuRenderMs: timing.gpuRenderMs,
  };
}

function formatTiming(timing: TimingSummary): string {
  if (timing.gpuComputeMs !== undefined) {
    return `${timing.gpuComputeMs.toFixed(2)} ms GPU compute · ${timing.wallMs.toFixed(2)} ms queue wall`;
  }
  return `${timing.wallMs.toFixed(2)} ms queue wall`;
}

class FrameMeter {
  #dropped = 0;
  #frames = 0;
  #lastFrameAt: number | undefined;
  #lastPresentedFrames: number | undefined;
  #mode: 'camera' | 'static' = 'static';
  #windowStartedAt = performance.now();

  reset(mode: 'camera' | 'static' = this.#mode): void {
    this.#dropped = 0;
    this.#frames = 0;
    this.#lastFrameAt = undefined;
    this.#lastPresentedFrames = undefined;
    this.#mode = mode;
    this.#windowStartedAt = performance.now();
    qualification.frames = { dropped: 0 };
    setDiagnostic('diag-fps', 'Measuring…');
    setDiagnostic('diag-dropped', '0');
    syncQualification();
  }

  noteStatic(now: number): void {
    if (document.hidden) {
      return;
    }
    this.#frames += 1;
    if (this.#lastFrameAt !== undefined) {
      const expectedFrames = Math.round((now - this.#lastFrameAt) / (1000 / CAMERA_FRAME_RATE));
      this.#dropped += Math.max(0, expectedFrames - 1);
    }
    this.#lastFrameAt = now;
    this.#publish(now);
  }

  noteCameraSource(metadata?: VideoFrameCallbackMetadata): void {
    if (document.hidden) {
      return;
    }
    if (
      metadata?.presentedFrames !== undefined &&
      this.#lastPresentedFrames !== undefined &&
      metadata.presentedFrames > this.#lastPresentedFrames + 1
    ) {
      this.#dropped += metadata.presentedFrames - this.#lastPresentedFrames - 1;
    }
    this.#lastPresentedFrames = metadata?.presentedFrames ?? this.#lastPresentedFrames;
  }

  noteCameraDropped(): void {
    if (!document.hidden) {
      this.#dropped += 1;
    }
  }

  noteCameraRendered(now: number): void {
    if (document.hidden) {
      return;
    }
    this.#frames += 1;
    this.#lastFrameAt = now;
    this.#publish(now);
  }

  #publish(now: number): void {
    const elapsed = now - this.#windowStartedAt;
    if (elapsed >= 1000) {
      const fps = (this.#frames * 1000) / elapsed;
      qualification.frames = { dropped: this.#dropped, fps };
      setDiagnostic('diag-fps', `${fps.toFixed(1)} FPS`);
      setDiagnostic('diag-dropped', String(this.#dropped));
      this.#frames = 0;
      this.#windowStartedAt = now;
      syncQualification();
    }
  }
}

const frameMeter = new FrameMeter();

function stopLiveBenchmark(): void {
  liveRenderGeneration += 1;
  liveRenderInFlight = false;
  liveTimingSamples = [];
}

function beginLiveBenchmark(): void {
  stopLiveBenchmark();
  frameMeter.reset('camera');
  setDiagnostic('diag-cold', 'Measuring first live inference…');
  setDiagnostic('diag-warm', `Measuring ${LIVE_TIMING_SAMPLE_COUNT - 1} warmed live inferences…`);
  setDiagnostic('diag-timing', 'Measuring…');
  setQualificationState('pending');
}

function failLiveBenchmark(generation: number, error: unknown): void {
  if (generation !== liveRenderGeneration || disposed || deviceLost) {
    return;
  }
  const message = `Camera rendering stopped: ${errorMessage(error)}`;
  setStatus('error', message);
  setQualificationState('failed', message);
}

function recordLiveTiming(generation: number, timing: RenderTiming): void {
  if (generation !== liveRenderGeneration || disposed || deviceLost) {
    return;
  }
  frameMeter.noteCameraRendered(performance.now());
  clearTransientStatus();
  if (liveTimingSamples.length >= LIVE_TIMING_SAMPLE_COUNT) {
    return;
  }
  liveTimingSamples.push(timing);
  if (liveTimingSamples.length === LIVE_TIMING_SAMPLE_COUNT) {
    recordTimingMeasurements(liveTimingSamples, 'live camera');
    setQualificationState('complete');
  }
}

const light = setupLightInput(
  canvas,
  (update) => {
    renderer?.update(update);
    qualification.fairyPose = {
      headingRadians: update.fairyHeading ?? qualification.fairyPose.headingRadians,
      pitchNormalized: update.fairyPitch ?? qualification.fairyPose.pitchNormalized,
      rollNormalized: update.fairyBank ?? qualification.fairyPose.rollNormalized,
    };
    setDiagnostic(
      'diag-fairy-pose',
      `Roll ${(qualification.fairyPose.rollNormalized * FAIRY_MAX_ROLL_DEGREES).toFixed(0)}° · ` +
        `pitch ${(qualification.fairyPose.pitchNormalized * FAIRY_MAX_PITCH_DEGREES).toFixed(0)}°`,
    );
  },
  listenerController.signal,
);

const camera = new DepthCameraSession(
  video,
  {
    onFrame: (frame, metadata) => {
      const activeRenderer = renderer;
      if (!activeRenderer || disposed || deviceLost) {
        return;
      }
      frameMeter.noteCameraSource(metadata);
      if (liveRenderInFlight) {
        frameMeter.noteCameraDropped();
        return;
      }
      const generation = liveRenderGeneration;
      liveRenderInFlight = true;
      light.flightTick();
      activeRenderer.update({ fairyTime: performance.now() / 1000 });
      let measurement: Promise<RenderTiming> | undefined;
      try {
        measurement = activeRenderer.render(frame, { captureTiming: true });
      } catch (error) {
        liveRenderInFlight = false;
        failLiveBenchmark(generation, error);
        return;
      }
      if (!measurement) {
        liveRenderInFlight = false;
        failLiveBenchmark(generation, new Error('The live renderer returned no timing result.'));
        return;
      }
      void measurement
        .then((timing) => recordLiveTiming(generation, timing))
        .catch((error: unknown) => failLiveBenchmark(generation, error))
        .finally(() => {
          if (generation === liveRenderGeneration) {
            liveRenderInFlight = false;
          }
        });
    },
    onError: (error) => {
      if (!disposed && !deviceLost) {
        stopLiveBenchmark();
        setStatus('error', `Camera stopped: ${errorMessage(error)}`);
      }
    },
    onEnded: () => {
      if (!disposed && !deviceLost) {
        stopLiveBenchmark();
        setStatus('error', 'The camera stream ended.');
      }
    },
  },
  { frameRate: CAMERA_FRAME_RATE, facingMode: 'user' },
);

document.addEventListener(
  'visibilitychange',
  () => {
    if (!document.hidden) {
      frameMeter.reset(camera.active ? 'camera' : 'static');
    }
  },
  { signal: listenerController.signal },
);

function stopStaticLoop(): void {
  staticLoopGeneration += 1;
}

function nextAnimationFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

async function renderMeasuredFrame(
  bitmap: ImageBitmap,
  options?: { capturePixels?: boolean; skipDepth?: boolean },
): Promise<RenderTiming> {
  const activeRenderer = renderer;
  if (!activeRenderer) {
    throw new Error('The renderer is not ready.');
  }
  await nextAnimationFrame();
  activeRenderer.update({ fairyEnabled, fairyTime: performance.now() / 1000 });
  const source = new VideoFrame(bitmap, { timestamp: performance.now() * 1000 });
  try {
    const measurement = activeRenderer.render(
      { source, uvTransform: d.mat2x2f.identity(), swapAxes: false },
      {
        capturePixels: options?.capturePixels,
        captureTiming: true,
        skipDepth: options?.skipDepth ?? false,
      },
    );
    if (!measurement) {
      throw new Error('The renderer did not return a timing measurement.');
    }
    return await measurement;
  } finally {
    source.close();
  }
}

function recordTimingMeasurements(
  measurements: readonly RenderTiming[],
  benchmarkLabel: string,
): void {
  const coldMeasurement = measurements[0];
  const warmMeasurements = measurements.slice(1);
  if (!coldMeasurement || warmMeasurements.length === 0 || !qualification.timings) {
    throw new Error(`The ${benchmarkLabel} benchmark did not produce enough timing samples.`);
  }
  if (
    deviceLost ||
    measurements.some((measurement) => measurement.gpuTimingError?.includes('[Device] is lost'))
  ) {
    throw new Error(`The WebGPU device was lost during the ${benchmarkLabel} benchmark.`);
  }

  const gpuComputeSamples = warmMeasurements.flatMap((sample) =>
    sample.gpuComputeMs === undefined ? [] : [sample.gpuComputeMs],
  );
  const gpuFrameSamples = warmMeasurements.flatMap((sample) =>
    sample.gpuFrameMs === undefined ? [] : [sample.gpuFrameMs],
  );
  const gpuRenderSamples = warmMeasurements.flatMap((sample) =>
    sample.gpuRenderMs === undefined ? [] : [sample.gpuRenderMs],
  );
  const cold = compactTiming(coldMeasurement);
  const warm: WarmTimingSummary = {
    samples: warmMeasurements.length,
    wallMs: median(warmMeasurements.map((sample) => sample.wallMs)),
    gpuComputeMs: gpuComputeSamples.length ? median(gpuComputeSamples) : undefined,
    gpuFrameMs: gpuFrameSamples.length ? median(gpuFrameSamples) : undefined,
    gpuRenderMs: gpuRenderSamples.length ? median(gpuRenderSamples) : undefined,
  };
  const hasGpuTiming = cold.gpuComputeMs !== undefined && warm.gpuComputeMs !== undefined;
  qualification.timings = {
    ...qualification.timings,
    cold,
    warm,
    source: hasGpuTiming ? 'gpu-timestamp-query' : 'wall-clock-approximation',
    gpuTimestampNote: hasGpuTiming
      ? 'True GPU timestamp query; browser privacy quantisation may apply.'
      : coldMeasurement.gpuTimingError,
  };
  setDiagnostic('diag-cold', formatTiming(cold));
  setDiagnostic('diag-warm', `${formatTiming(warm)} · median of ${warm.samples}`);
  setDiagnostic(
    'diag-timing',
    hasGpuTiming
      ? 'True GPU timestamp query (privacy quantisation may apply)'
      : `Wall-clock approximation${coldMeasurement.gpuTimingError ? ` · ${coldMeasurement.gpuTimingError}` : ''}`,
  );
}

async function benchmarkStaticImage(bitmap: ImageBitmap): Promise<void> {
  setStatus('busy', `Benchmarking cold and warmed inference (${BENCHMARK_SAMPLE_COUNT} runs)…`);
  const measurements: RenderTiming[] = [];
  for (let index = 0; index < BENCHMARK_SAMPLE_COUNT; index += 1) {
    measurements.push(await renderMeasuredFrame(bitmap));
  }
  recordTimingMeasurements(measurements, 'static');

  const outputModes = [
    ['relit', 0],
    ['camera', 1],
    ['relativeDisparity', 2],
    ['normals', 3],
  ] as const;
  const outputs = {} as NonNullable<QualificationRecord['outputs']>;
  for (const [label, mode] of outputModes) {
    renderer?.update({ mode });
    const capture = await renderMeasuredFrame(bitmap, { capturePixels: true, skipDepth: true });
    if (!capture.outputPixels) {
      throw new Error(`The ${label} view did not return pixel-readback evidence.`);
    }
    outputs[label] = capture.outputPixels;
    if (capture.outputPixels.nonBlackPixelRatio < 0.01) {
      throw new Error(`The ${label} view pixel readback was effectively black.`);
    }
  }
  renderer?.update({ mode: 0 });
  qualification.outputs = outputs;
  setQualificationState('complete');
}

function startStaticLoop(bitmap: ImageBitmap): void {
  const generation = ++staticLoopGeneration;
  depthDirty = false;
  frameMeter.reset('static');
  const step = (): void => {
    if (generation !== staticLoopGeneration || disposed || deviceLost) {
      return;
    }
    const activeRenderer = renderer;
    if (activeRenderer) {
      const now = performance.now();
      frameMeter.noteStatic(now);
      light.flightTick();
      activeRenderer.update({ fairyTime: now / 1000 });
      const source = new VideoFrame(bitmap, { timestamp: now * 1000 });
      try {
        void activeRenderer.render(
          { source, uvTransform: d.mat2x2f.identity(), swapAxes: false },
          { skipDepth: !depthDirty },
        );
        depthDirty = false;
        clearTransientStatus();
      } catch (error) {
        if (generation === staticLoopGeneration && !disposed && !deviceLost) {
          setStatus('error', `Rendering stopped: ${errorMessage(error)}`);
        }
        return;
      } finally {
        source.close();
      }
    }
    if (generation === staticLoopGeneration) {
      requestAnimationFrame(step);
    }
  };
  requestAnimationFrame(step);
}

async function setFacing(facing: (typeof FACING_MODES)[number]): Promise<void> {
  camera.facingMode = facing === 'front' ? 'user' : 'environment';
  if (chooser?.source !== SourceChoice.CAMERA) {
    return;
  }
  renderer?.update({ mirror: camera.facingMode === 'user' });
  if (!camera.active) {
    return;
  }
  stopLiveBenchmark();
  camera.stop();
  beginLiveBenchmark();
  try {
    await camera.start();
    renderer?.resetHistory();
    depthDirty = true;
  } catch (error) {
    const message = `Could not switch camera: ${errorMessage(error)}`;
    setStatus('error', message);
    setQualificationState('failed', message);
  }
}

async function loadDemoImage(): Promise<ImageBitmap> {
  if (!demoImage) {
    const response = await fetch(DEMO_IMAGE_URL, { signal: listenerController.signal });
    if (!response.ok) {
      throw new Error(`Demo photo download failed (${response.status}).`);
    }
    demoImage = await createImageBitmap(await response.blob());
  }
  return demoImage;
}

async function startSource(source: SourceChoice, uploadedImage?: ImageBitmap): Promise<void> {
  stopStaticLoop();
  stopLiveBenchmark();
  camera.stop();
  if (source === SourceChoice.CAMERA) {
    beginLiveBenchmark();
    setStatus('busy', 'Waiting for camera permission…');
    renderer?.update({ mirror: camera.facingMode === 'user' });
    await camera.start();
    renderer?.resetHistory();
    depthDirty = true;
    return;
  }

  frameMeter.reset('static');
  setQualificationState('pending');
  setStatus('busy', 'Preparing the local photograph…');
  const bitmap =
    source === SourceChoice.UPLOAD && uploadedImage ? uploadedImage : await loadDemoImage();
  renderer?.update({ mirror: false });
  renderer?.resetHistory();
  await benchmarkStaticImage(bitmap);
  startStaticLoop(bitmap);
}

async function attachBundle(bytes: ArrayBuffer): Promise<{ parseMs: number; compileMs: number }> {
  const activeRoot = root;
  if (!activeRoot || disposed || deviceLost) {
    throw new Error('The WebGPU device is not available.');
  }
  const parseStartedAt = performance.now();
  const bundle = parseDepthBundle(bytes);
  const parseMs = performance.now() - parseStartedAt;
  setStatus('busy', `Compiling ${bundle.model} pipelines…`);
  const compileStartedAt = performance.now();
  const nextPlan = new DepthInferencePlan(activeRoot, bundle);
  try {
    await nextPlan.initAsync();
    if (disposed || deviceLost) {
      nextPlan.destroy();
      throw new Error('The WebGPU device was lost during compilation.');
    }
    if (!renderer) {
      const nextRenderer = new DepthRelightingRenderer(activeRoot, canvas);
      await nextRenderer.initAsync();
      renderer = nextRenderer;
    }
  } catch (error) {
    nextPlan.destroy();
    throw error;
  }
  const compileMs = performance.now() - compileStartedAt;
  renderer.attach(nextPlan);
  plan?.destroy();
  plan = nextPlan;
  renderer.update({
    fairyEnabled,
    lightPosition: light.lightPosition,
    lightZ: light.lightZ,
    fairyHeading: light.fairyHeading,
    fairyBank: light.fairyBank,
    fairyPitch: light.fairyPitch,
  });
  renderer.resetHistory();
  depthDirty = true;
  return { parseMs, compileMs };
}

function recordModel(variant: ModelVariant, label: string): void {
  qualification.model = {
    bundle: variant.bundle,
    bytes: variant.bytes,
    label,
    precision: variant.precision,
    revision: MODEL_REVISION,
    sha256: variant.sha256,
  };
  const precision = variant.precision === 'mixed-fp16' ? 'FP16 balanced' : 'FP32 fallback';
  setDiagnostic(
    'diag-model',
    `${precision} · ${variant.bytes.toLocaleString()} bytes · ${variant.bundle}.depthart`,
  );
  syncQualification();
}

async function loadModel(size: ModelSize): Promise<boolean> {
  const variant = chooser?.variant(size);
  if (!variant) {
    setStatus('error', `The ${size} model is unavailable on this device.`);
    return false;
  }
  const label = modelLabel(size, variant);
  recordModel(variant, label);
  try {
    setStatus('busy', `Loading and verifying ${label}…`);
    const modelStartedAt = performance.now();
    const bytes = await fetchModel(variant, listenerController.signal);
    const modelLoadAndVerifyMs = performance.now() - modelStartedAt;
    setDiagnostic(
      'diag-model-load',
      `${modelLoadAndVerifyMs.toFixed(2)} ms · SHA-256 verified`,
    );
    const { parseMs, compileMs } = await attachBundle(bytes);
    setDiagnostic('diag-pipeline', `${compileMs.toFixed(2)} ms · parse ${parseMs.toFixed(2)} ms`);
    qualification.timings = {
      modelLoadAndVerifyMs,
      modelParseMs: parseMs,
      pipelineCompileMs: compileMs,
      cold: { wallMs: 0 },
      warm: { wallMs: 0, samples: 0 },
      source: 'wall-clock-approximation',
    };
    currentBundle = variant.bundle;
    syncQualification();
    return true;
  } catch (error) {
    if (!disposed) {
      const message = `Could not load ${label}: ${errorMessage(error)}`;
      setStatus('error', message);
      if (AUTORUN) {
        setQualificationState('failed', message);
      }
    }
    return false;
  }
}

function showChooser(errorText?: string): void {
  stopStaticLoop();
  stopLiveBenchmark();
  camera.stop();
  status.hidden = true;
  chooser?.show(errorText);
}

async function start(): Promise<void> {
  if (!chooser) {
    return;
  }
  chooser.hide();
  if (chooser.variant(chooser.model)?.bundle !== currentBundle || !plan) {
    const loaded = await loadModel(chooser.model);
    if (!loaded || disposed || deviceLost) {
      if (!disposed && !deviceLost && !AUTORUN) {
        showChooser(statusMessage.textContent ?? 'Could not load the model.');
      }
      return;
    }
  }
  try {
    await startSource(chooser.source, chooser.uploadedImage);
  } catch (error) {
    if (!disposed && !deviceLost) {
      const message = `Could not start: ${errorMessage(error)}`;
      if (AUTORUN) {
        setStatus('error', message);
        setQualificationState('failed', message);
      } else {
        showChooser(message);
      }
    }
  }
}

async function requestPreferredAdapter(): Promise<{
  adapter: GPUAdapter | null;
  attempts: number;
}> {
  const retryDelaysMs = [0, 250, 750] as const;
  for (const [index, delayMs] of retryDelaysMs.entries()) {
    if (delayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: ADAPTER_POWER_PREFERENCE,
    });
    if (adapter) {
      return { adapter, attempts: index + 1 };
    }
  }
  return { adapter: null, attempts: retryDelaysMs.length };
}

async function readAdapterInfo(
  adapter: GPUAdapter,
  requestAttempts: number,
): Promise<AdapterDiagnostics> {
  const enhancedAdapter = adapter as AdapterWithInfo;
  const info = enhancedAdapter.info ?? (await enhancedAdapter.requestAdapterInfo?.());
  return {
    architecture: info?.architecture ?? '',
    description: info?.description ?? '',
    device: info?.device ?? '',
    powerPreference: ADAPTER_POWER_PREFERENCE,
    requestAttempts,
    vendor: info?.vendor ?? '',
  };
}

function formatAdapter(info: AdapterDiagnostics): string {
  const parts = [info.description, info.vendor, info.architecture, info.device].filter(
    (value, index, values) => value && values.indexOf(value) === index,
  );
  const identity = parts.length ? parts.join(' · ') : 'Adapter identity not exposed by browser';
  return `${identity} · ${info.powerPreference} request`;
}

async function initialize(): Promise<void> {
  setStatus('busy', 'Initializing WebGPU…');
  if (!navigator.gpu) {
    showUnsupported('navigator.gpu is not exposed by this browser.');
    return;
  }
  try {
    const { adapter, attempts } = await requestPreferredAdapter();
    if (!adapter) {
      showUnsupported('The browser could not obtain a WebGPU adapter.');
      return;
    }
    const adapterDiagnostics = await readAdapterInfo(adapter, attempts);
    qualification.adapter = adapterDiagnostics;
    setDiagnostic('diag-adapter', formatAdapter(adapterDiagnostics));

    const timestampQueryAvailable = adapter.features.has('timestamp-query');
    const pascalStabilityGuard = adapterDiagnostics.architecture.toLowerCase() === 'pascal';
    const timestampQueryEnabled =
      timestampQueryAvailable &&
      (TIMING_MODE === 'gpu' || (TIMING_MODE === 'auto' && !pascalStabilityGuard));
    const timestampQueryNote = !timestampQueryAvailable
      ? 'Unavailable — queue wall clock used'
      : timestampQueryEnabled
        ? 'Available — true GPU pass timestamps enabled'
        : TIMING_MODE === 'wall'
          ? 'Available — disabled by timing=wall; queue wall clock used'
          : 'Available — disabled by the Pascal stability guard after a reproduced device-loss; queue wall clock used';
    const requestedFeatures: GPUFeatureName[] = [];
    if (adapter.features.has('shader-f16')) {
      requestedFeatures.push('shader-f16');
    }
    if (timestampQueryEnabled) {
      requestedFeatures.push('timestamp-query');
    }
    const device = await adapter.requestDevice({ requiredFeatures: requestedFeatures });
    const nextRoot = tgpu.initFromDevice({ device });
    if (disposed) {
      nextRoot.destroy();
      return;
    }
    root = nextRoot;
    qualification.webgpu = true;
    qualification.features = {
      shaderF16: device.features.has('shader-f16'),
      timestampQuery: timestampQueryAvailable,
      timestampQueryEnabled: device.features.has('timestamp-query'),
      timestampQueryNote,
    };
    setDiagnostic('diag-webgpu', 'Available');
    setDiagnostic(
      'diag-f16',
      qualification.features.shaderF16 ? 'Available — FP16 small model selected' : 'Unavailable — FP32 fallback selected',
    );
    setDiagnostic('diag-timestamp', timestampQueryNote);
    syncQualification();

    chooser = new SourceChooser(
      device.features.has('shader-f16'),
      () => void start(),
      listenerController.signal,
    );
    void device.lost.then((info) => {
      if (disposed) {
        return;
      }
      deviceLost = true;
      stopStaticLoop();
      stopLiveBenchmark();
      camera.stop();
      const message = `GPU device lost: ${info.message || info.reason}`;
      setStatus('error', message);
      setQualificationState('failed', message);
    });
    showChooser();
    if (AUTORUN) {
      chooser.source = SourceChoice.DEMO;
      await start();
    }
  } catch (error) {
    if (!disposed) {
      showUnsupported(`WebGPU initialisation failed: ${errorMessage(error)}`);
    }
  }
}

function bindRange(
  inputId: string,
  outputId: string,
  update: (value: number) => void,
): void {
  const input = element<HTMLInputElement>(`#${inputId}`);
  const output = element<HTMLOutputElement>(`#${outputId}`);
  const apply = (): void => {
    const value = Number.parseFloat(input.value);
    output.value = value.toFixed(2);
    update(value);
  };
  input.addEventListener('input', apply, { signal: listenerController.signal });
  apply();
}

bindRange('intensity', 'intensity-output', (intensity) => renderer?.update({ intensity }));
bindRange('ambient', 'ambient-output', (exposure) => renderer?.update({ exposure }));
bindRange('relief', 'relief-output', (relief) => renderer?.update({ relief }));
bindRange('shadow', 'shadow-output', (shadow) => renderer?.update({ shadow }));
bindRange('reflection', 'reflection-output', (specular) => renderer?.update({ specular }));
bindRange('occlusion', 'occlusion-output', (occlusion) => renderer?.update({ occlusion }));

element<HTMLSelectElement>('#view-select').addEventListener(
  'change',
  (event) => {
    const mode = Number.parseInt((event.currentTarget as HTMLSelectElement).value, 10);
    renderer?.update({ mode });
    canvas.setAttribute('aria-label', VIEW_OUTPUT_LABELS[mode] ?? VIEW_OUTPUT_LABELS[0]);
  },
  { signal: listenerController.signal },
);

element<HTMLInputElement>('#fairy-light').addEventListener(
  'change',
  (event) => {
    fairyEnabled = (event.currentTarget as HTMLInputElement).checked;
    renderer?.update({ fairyEnabled });
  },
  { signal: listenerController.signal },
);

element<HTMLSelectElement>('#camera-select').addEventListener(
  'change',
  (event) => {
    const facing = (event.currentTarget as HTMLSelectElement).value as (typeof FACING_MODES)[number];
    void setFacing(facing);
  },
  { signal: listenerController.signal },
);

element<HTMLInputElement>('#light-color').addEventListener(
  'input',
  (event) => {
    const hex = (event.currentTarget as HTMLInputElement).value.slice(1);
    const packed = Number.parseInt(hex, 16);
    renderer?.update({
      lightColor: [((packed >> 16) & 0xff) / 255, ((packed >> 8) & 0xff) / 255, (packed & 0xff) / 255],
    });
  },
  { signal: listenerController.signal },
);

element<HTMLButtonElement>('#source-button').addEventListener('click', () => showChooser(), {
  signal: listenerController.signal,
});

function cleanup(): void {
  if (disposed) {
    return;
  }
  disposed = true;
  stopStaticLoop();
  stopLiveBenchmark();
  listenerController.abort();
  chooser?.destroy();
  demoImage?.close();
  camera.destroy();
  renderer?.destroy();
  plan?.destroy();
  root?.destroy();
}

window.addEventListener('pagehide', cleanup, { once: true });

renderer?.update(defaultRelightingSettings);
void initialize();
