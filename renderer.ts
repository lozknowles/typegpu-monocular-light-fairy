import { common, d } from 'typegpu';
import type {
  SampledFlag,
  StorageFlag,
  TgpuBindGroup,
  TgpuBuffer,
  TgpuComputePipeline,
  TgpuRenderPipeline,
  TgpuRoot,
  TgpuSampler,
  TgpuQuerySet,
  TgpuTexture,
  UniformFlag,
} from 'typegpu';
import type { DepthCameraFrame } from './camera-session.ts';
import { DepthDisparityRangeEstimator } from './inference/disparity-range.ts';
import type { DepthInferencePlan } from './inference/depthart.ts';
import {
  DEPTH_WORKGROUP_SIZE,
  DepthParams,
  RelightMode,
  RelightParams,
  SURFACE_FAR_Z,
  SURFACE_WORKGROUP_SIZE,
  depthPrepareKernel,
  depthPrepareLayout,
  rangeStabilityLayout,
  relightFragment,
  relightFrameLayout,
  relightLayout,
  stabilizeRangeKernel,
  surfaceKernel,
  surfaceLayout,
} from './shaders.ts';

const MAX_CANVAS_SIDE = 1024;
const MAX_PIXEL_RATIO = 2;

const LIGHT_Z_CLEARANCE = 0.04;
export const LIGHT_Z_MIN = SURFACE_FAR_Z + LIGHT_Z_CLEARANCE;
export const LIGHT_Z_MAX = 1.65;

type SurfaceTexture = TgpuTexture<{
  size: readonly [number, number];
  format: 'rgba16float';
}> &
  StorageFlag &
  SampledFlag;

interface RelightAttachment {
  readonly depthWorkgroups: number;
  readonly fieldWorkgroups: readonly [number, number];
  readonly disparity: TgpuBuffer<d.WgslArray<d.Vec4f>> & StorageFlag;
  readonly history: TgpuBuffer<d.WgslArray<d.F32>> & StorageFlag;
  readonly surface: SurfaceTexture;
  readonly depthBindGroup: TgpuBindGroup<typeof depthPrepareLayout.entries>;
  readonly surfaceBindGroup: TgpuBindGroup<typeof surfaceLayout.entries>;
  readonly relightBindGroup: TgpuBindGroup<typeof relightLayout.entries>;
}

interface RelightingState {
  readonly lightPosition: readonly [number, number];
  readonly lightZ: number;
  readonly fairyEnabled: boolean;
  readonly fairyTime: number;
  readonly fairyHeading: number;
  readonly fairyBank: number;
  readonly fairyPitch: number;
  readonly mirror: boolean;
  readonly demoMirrorStudy: boolean;
  readonly lightColor: readonly [number, number, number];
  readonly exposure: number;
  readonly intensity: number;
  readonly relief: number;
  readonly specular: number;
  readonly shadow: number;
  readonly occlusion: number;
  readonly mode: number;
}

type RelightingSettings = Partial<RelightingState>;

export interface RenderTiming {
  readonly wallMs: number;
  readonly gpuComputeMs?: number;
  readonly gpuRenderMs?: number;
  readonly gpuFrameMs?: number;
  readonly gpuTimingError?: string;
  readonly outputPixels?: PixelDiagnostics;
}

export interface PixelDiagnostics {
  readonly format: GPUTextureFormat;
  readonly height: number;
  readonly meanChannels: readonly [number, number, number, number];
  readonly nonBlackPixelRatio: number;
  readonly nonBlackPixels: number;
  readonly sha256: string;
  readonly totalPixels: number;
  readonly width: number;
}

interface PixelReadback {
  readonly buffer: GPUBuffer;
  readonly bytesPerRow: number;
  readonly format: GPUTextureFormat;
  readonly height: number;
  readonly width: number;
}

export const defaultRelightingSettings: RelightingState = {
  lightPosition: [0.34, 0.34],
  lightZ: 0.42,
  fairyEnabled: true,
  fairyTime: 0,
  fairyHeading: -0.4,
  fairyBank: 0,
  fairyPitch: 0,
  mirror: true,
  demoMirrorStudy: false,
  lightColor: [1, 0.72, 0.46],
  exposure: 0.5,
  intensity: 3,
  relief: 0.85,
  specular: 0.55,
  shadow: 0.7,
  occlusion: 0.55,
  mode: RelightMode.RELIT,
};

export class DepthRelightingRenderer {
  readonly #root: TgpuRoot;
  readonly #canvas: HTMLCanvasElement;
  readonly #context: GPUCanvasContext;
  readonly #canvasFormat: GPUTextureFormat;
  readonly #rangeEstimator: DepthDisparityRangeEstimator;
  readonly #frameRange: TgpuBuffer<d.Vec2f> & StorageFlag;
  readonly #stableRange: TgpuBuffer<d.Vec2f> & StorageFlag;
  readonly #depthParams: TgpuBuffer<typeof DepthParams> & UniformFlag;
  readonly #relightParams: TgpuBuffer<typeof RelightParams> & UniformFlag;
  readonly #sampler: TgpuSampler;
  readonly #rangeBindGroup: TgpuBindGroup<typeof rangeStabilityLayout.entries>;
  readonly #stabilizePipeline: TgpuComputePipeline;
  readonly #depthPipeline: TgpuComputePipeline;
  readonly #surfacePipeline: TgpuComputePipeline;
  readonly #relightPipeline: TgpuRenderPipeline<d.Vec4f>;
  readonly #timingQuerySet: TgpuQuerySet<'timestamp'> | undefined;
  #plan: DepthInferencePlan | undefined;
  #attachment: RelightAttachment | undefined;
  #uvTransform = d.mat2x2f.identity();
  #swapAxes = false;
  #firstFrame = true;
  #settings: RelightingState = defaultRelightingSettings;

  constructor(root: TgpuRoot, canvas: HTMLCanvasElement) {
    this.#root = root;
    this.#canvas = canvas;
    this.#canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    this.#context = root.configureContext({
      canvas,
      alphaMode: 'opaque',
      format: this.#canvasFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    this.#rangeEstimator = new DepthDisparityRangeEstimator(root);
    this.#frameRange = root.createBuffer(d.vec2f, d.vec2f(0, 1)).$usage('storage');
    this.#stableRange = root.createBuffer(d.vec2f, d.vec2f(0, 1)).$usage('storage');
    this.#depthParams = root
      .createBuffer(DepthParams, { outputSize: d.vec2u(1), reset: 1 })
      .$usage('uniform');
    this.#relightParams = root.createBuffer(RelightParams).$usage('uniform');
    this.#sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.#rangeBindGroup = root.createBindGroup(rangeStabilityLayout, {
      params: this.#depthParams,
      frameRange: this.#frameRange,
      stableRange: this.#stableRange,
    });
    this.#stabilizePipeline = root.createComputePipeline({ compute: stabilizeRangeKernel });
    this.#depthPipeline = root.createComputePipeline({ compute: depthPrepareKernel });
    this.#surfacePipeline = root.createComputePipeline({ compute: surfaceKernel });
    this.#relightPipeline = root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: relightFragment,
      targets: { format: this.#canvasFormat },
    });
    this.#timingQuerySet = root.enabledFeatures.has('timestamp-query')
      ? root.createQuerySet('timestamp', 4)
      : undefined;
    this.#writeRelightParams();
  }

  async initAsync(): Promise<void> {
    await Promise.all([
      this.#rangeEstimator.initAsync(),
      this.#stabilizePipeline.initAsync(),
      this.#depthPipeline.initAsync(),
      this.#surfacePipeline.initAsync(),
      this.#relightPipeline.initAsync(),
    ]);
  }

  attach(plan: DepthInferencePlan): void {
    this.detach();
    const [width, height] = plan.outputSize;
    const pixelCount = width * height;
    const disparity = this.#root
      .createBuffer(d.arrayOf(d.vec4f, pixelCount), plan.outputBuffer)
      .$usage('storage');
    const history = this.#root.createBuffer(d.arrayOf(d.f32, pixelCount)).$usage('storage');
    const surface: SurfaceTexture = this.#root
      .createTexture({ size: [width, height], format: 'rgba16float' })
      .$usage('storage', 'sampled');

    this.#plan = plan;
    this.#attachment = {
      depthWorkgroups: Math.ceil(pixelCount / DEPTH_WORKGROUP_SIZE),
      fieldWorkgroups: [
        Math.ceil(width / SURFACE_WORKGROUP_SIZE),
        Math.ceil(height / SURFACE_WORKGROUP_SIZE),
      ],
      disparity,
      history,
      surface,
      depthBindGroup: this.#root.createBindGroup(depthPrepareLayout, {
        params: this.#depthParams,
        disparity,
        stableRange: this.#stableRange,
        history,
      }),
      surfaceBindGroup: this.#root.createBindGroup(surfaceLayout, {
        params: this.#depthParams,
        depth: history,
        surface: surface.createView(d.textureStorage2d('rgba16float', 'write-only')),
      }),
      relightBindGroup: this.#root.createBindGroup(relightLayout, {
        params: this.#relightParams,
        surface: surface.createView(),
        sampler: this.#sampler,
      }),
    };
    this.#rangeEstimator.attach(disparity, this.#frameRange, pixelCount);
    this.#depthParams.write({ outputSize: d.vec2u(width, height), reset: 1 });
    this.#firstFrame = true;
  }

  detach(): void {
    this.#rangeEstimator.detach();
    const attachment = this.#attachment;
    if (attachment) {
      attachment.disparity.destroy();
      attachment.history.destroy();
      attachment.surface.destroy();
    }
    this.#attachment = undefined;
    this.#plan = undefined;
  }

  update(settings: RelightingSettings): void {
    this.#settings = {
      ...this.#settings,
      ...settings,
      lightPosition: [...(settings.lightPosition ?? this.#settings.lightPosition)],
      lightColor: [...(settings.lightColor ?? this.#settings.lightColor)],
    };
  }

  resetHistory(): void {
    this.#firstFrame = true;
  }

  render(
    frame: DepthCameraFrame,
    options?: { skipDepth?: boolean; captureTiming?: boolean; capturePixels?: boolean },
  ): Promise<RenderTiming> | undefined {
    const plan = this.#plan;
    const attachment = this.#attachment;
    if (!plan || !attachment) {
      throw new Error('No depth inference plan is attached to the relighting renderer.');
    }
    const updateDepth = !options?.skipDepth || this.#firstFrame;
    const wallStart = performance.now();
    const timingQuerySet =
      options?.captureTiming && this.#timingQuerySet?.available
        ? this.#timingQuerySet
        : undefined;

    this.#syncCanvasSize();
    this.#uvTransform = frame.uvTransform;
    this.#swapAxes = frame.swapAxes;
    this.#writeRelightParams();
    if (updateDepth) {
      this.#depthParams.patch({ reset: this.#firstFrame ? 1 : 0 });
    }

    const encoder = this.#root['~unstable'].createCommandEncoder();
    const externalFrame = this.#root.device.importExternalTexture({ source: frame.source });
    if (updateDepth) {
      const pass = encoder.beginComputePass({
        timestampWrites: timingQuerySet
          ? {
              querySet: timingQuerySet,
              beginningOfPassWriteIndex: 0,
              endOfPassWriteIndex: 1,
            }
          : undefined,
      });
      plan.encodeFrame(pass, externalFrame, {
        uvTransform: frame.uvTransform,
        mirrorX: this.#settings.mirror,
        swapAxes: frame.swapAxes,
      });
      this.#rangeEstimator.encode(pass);
      this.#stabilizePipeline.with(pass).with(this.#rangeBindGroup).dispatchWorkgroups(1);
      this.#depthPipeline
        .with(pass)
        .with(attachment.depthBindGroup)
        .dispatchWorkgroups(attachment.depthWorkgroups);
      const [fieldX, fieldY] = attachment.fieldWorkgroups;
      this.#surfacePipeline
        .with(pass)
        .with(attachment.surfaceBindGroup)
        .dispatchWorkgroups(fieldX, fieldY);
      pass.end();
    }

    const outputTexture = this.#context.getCurrentTexture();
    const pass = encoder.beginRenderPass({
      colorAttachments: { view: outputTexture.createView() },
      timestampWrites: timingQuerySet
        ? {
            querySet: timingQuerySet,
            beginningOfPassWriteIndex: 2,
            endOfPassWriteIndex: 3,
          }
        : undefined,
    });
    this.#relightPipeline
      .with(pass)
      .with(attachment.relightBindGroup)
      .with(this.#root.createBindGroup(relightFrameLayout, { frame: externalFrame }))
      .draw(3);
    pass.end();
    let pixelReadback: PixelReadback | undefined;
    if (options?.capturePixels) {
      const width = outputTexture.width;
      const height = outputTexture.height;
      const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
      const buffer = this.#root.device.createBuffer({
        size: bytesPerRow * height,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.#root.unwrap(encoder).copyTextureToBuffer(
        { texture: outputTexture },
        { buffer, bytesPerRow, rowsPerImage: height },
        { width, height, depthOrArrayLayers: 1 },
      );
      pixelReadback = { buffer, bytesPerRow, format: this.#canvasFormat, height, width };
    }
    encoder.submit();
    this.#firstFrame = false;
    if (options?.captureTiming || pixelReadback) {
      return this.#finishTiming(wallStart, updateDepth, timingQuerySet, pixelReadback);
    }
    return undefined;
  }

  destroy(): void {
    this.detach();
    this.#rangeEstimator.destroy();
    this.#frameRange.destroy();
    this.#stableRange.destroy();
    this.#depthParams.destroy();
    this.#relightParams.destroy();
    this.#timingQuerySet?.destroy();
    this.#context.unconfigure();
  }

  async #finishTiming(
    wallStart: number,
    updateDepth: boolean,
    timingQuerySet: TgpuQuerySet<'timestamp'> | undefined,
    pixelReadback: PixelReadback | undefined,
  ): Promise<RenderTiming> {
    await this.#root.device.queue.onSubmittedWorkDone();
    const wallMs = performance.now() - wallStart;
    const outputPixels = pixelReadback
      ? await this.#readPixelDiagnostics(pixelReadback)
      : undefined;
    if (!timingQuerySet) {
      return { wallMs, outputPixels };
    }
    try {
      timingQuerySet.resolve();
      const timestamps = await timingQuerySet.read();
      const computeStart = timestamps[0];
      const computeEnd = timestamps[1];
      const renderStart = timestamps[2];
      const renderEnd = timestamps[3];
      if (
        computeStart === undefined ||
        computeEnd === undefined ||
        renderStart === undefined ||
        renderEnd === undefined
      ) {
        return { wallMs, gpuTimingError: 'Incomplete timestamp query result.', outputPixels };
      }
      return {
        wallMs,
        gpuComputeMs: updateDepth ? Number(computeEnd - computeStart) / 1_000_000 : undefined,
        gpuRenderMs: Number(renderEnd - renderStart) / 1_000_000,
        gpuFrameMs:
          Number(renderEnd - (updateDepth ? computeStart : renderStart)) / 1_000_000,
        outputPixels,
      };
    } catch (error) {
      return {
        wallMs,
        gpuTimingError: error instanceof Error ? error.message : String(error),
        outputPixels,
      };
    }
  }

  async #readPixelDiagnostics(readback: PixelReadback): Promise<PixelDiagnostics> {
    const { buffer, bytesPerRow, format, height, width } = readback;
    try {
      await buffer.mapAsync(GPUMapMode.READ);
      const mapped = new Uint8Array(buffer.getMappedRange());
      const packed = new Uint8Array(width * height * 4);
      for (let row = 0; row < height; row += 1) {
        const sourceOffset = row * bytesPerRow;
        const targetOffset = row * width * 4;
        packed.set(mapped.subarray(sourceOffset, sourceOffset + width * 4), targetOffset);
      }

      const channelSums = [0, 0, 0, 0];
      let nonBlackPixels = 0;
      for (let offset = 0; offset < packed.length; offset += 4) {
        const channel0 = packed[offset] ?? 0;
        const channel1 = packed[offset + 1] ?? 0;
        const channel2 = packed[offset + 2] ?? 0;
        const channel3 = packed[offset + 3] ?? 0;
        channelSums[0] += channel0;
        channelSums[1] += channel1;
        channelSums[2] += channel2;
        channelSums[3] += channel3;
        if (channel0 > 2 || channel1 > 2 || channel2 > 2) {
          nonBlackPixels += 1;
        }
      }
      const totalPixels = width * height;
      const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', packed));
      return {
        format,
        height,
        meanChannels: channelSums.map((sum) => sum / totalPixels) as [
          number,
          number,
          number,
          number,
        ],
        nonBlackPixelRatio: nonBlackPixels / totalPixels,
        nonBlackPixels,
        sha256: [...digest].map((value) => value.toString(16).padStart(2, '0')).join(''),
        totalPixels,
        width,
      };
    } finally {
      if (buffer.mapState === 'mapped') {
        buffer.unmap();
      }
      buffer.destroy();
    }
  }

  #syncCanvasSize(): void {
    const displayWidth = this.#canvas.clientWidth;
    if (displayWidth <= 0) {
      return;
    }
    const ratio = Math.min(globalThis.devicePixelRatio || 1, MAX_PIXEL_RATIO);
    const side = Math.min(MAX_CANVAS_SIDE, Math.max(1, Math.round(displayWidth * ratio)));
    if (this.#canvas.width !== side || this.#canvas.height !== side) {
      this.#canvas.width = side;
      this.#canvas.height = side;
    }
  }

  #writeRelightParams(): void {
    this.#relightParams.write({
      uvTransform: this.#uvTransform,
      lightColor: d.vec4f(...this.#settings.lightColor, 1),
      lightPosition: d.vec2f(...this.#settings.lightPosition),
      lightZ: this.#settings.lightZ,
      fairyTime: this.#settings.fairyTime,
      fairyHeading: this.#settings.fairyHeading,
      fairyBank: this.#settings.fairyBank,
      fairyPitch: this.#settings.fairyPitch,
      exposure: this.#settings.exposure,
      intensity: this.#settings.intensity,
      relief: this.#settings.relief,
      specular: this.#settings.specular,
      shadow: this.#settings.shadow,
      occlusion: this.#settings.occlusion,
      swapAxes: this.#swapAxes ? 1 : 0,
      mirror: this.#settings.mirror ? 1 : 0,
      demoMirrorStudy: this.#settings.demoMirrorStudy ? 1 : 0,
      mode: this.#settings.mode,
      fairyEnabled: this.#settings.fairyEnabled ? 1 : 0,
    });
  }
}
