import { d, std, tgpu } from 'typegpu';

export const DEPTH_WORKGROUP_SIZE = 64;
export const SURFACE_WORKGROUP_SIZE = 8;

const RING_OFFSETS = [-1, 0, 1] as const;

export const RelightMode = {
  RELIT: 0,
  CAMERA: 1,
  DEPTH: 2,
  NORMALS: 3,
} as const;

const RANGE_BLEND = 0.12;
const TEMPORAL_ALPHA = 0.32;
const MOTION_ALPHA = 0.8;
const MOTION_LOW = 0.02;
const MOTION_HIGH = 0.09;

const GRADIENT_RADIUS = 7;
const GRADIENT_BACK = -GRADIENT_RADIUS;
const GRADIENT_LIMIT = 0.009;
const GRADIENT_NOISE = 0.0003;
const GRADIENT_NOISE_ENERGY = GRADIENT_NOISE ** 2;
const OCCLUSION_RADII = [3, 9] as const;
const OCCLUSION_TAPS = OCCLUSION_RADII.length * (RING_OFFSETS.length ** 2 - 1);
const OCCLUSION_SCALE = 0.07;
const OCCLUSION_RANGE = 0.25;
const OCCLUSION_FLOOR = 0.012;

const NEAR_Z = 0;
/** Depth of the furthest surface the relit scene can hold */
export const SURFACE_FAR_Z = -0.7;
const LIGHT_RADIUS = 0.85;
const LIGHT_WRAP = 0.25;
const RELIEF_SCALE = 200;
const SLOPE_COMPRESSION = 0.55;
const SPECULAR_POWER = 36;
const SPECULAR_F0 = 0.06;
const GAMMA = 2.2;
const WHITE_POINT = 2.6;
const LUMINANCE_WEIGHTS = d.vec3f(0.2126, 0.7152, 0.0722);
const HIGHLIGHT_BLEACH = 2;
const AMBIENT_FILL = d.vec3f(0.78, 0.86, 1);
const DITHER_STEP = 1 / 255;

const BULB_WORLD_RADIUS = 0.05;
const BULB_CAMERA_Z = 2;
const BULB_REFERENCE_Z = 0.42;
const BULB_CORE = 8;
const BULB_LIMB = 0.28;
const BULB_EDGE = 0.75;
const BULB_EDGE_FLOOR = 0.004;
const BULB_EDGE_LIMIT = 0.3;
const BULB_HALO = 1.6;
const BULB_HALO_SPAN = 1.2;
const BULB_VEIL = 0.12;
const BULB_VEIL_SPAN = 4;
const BULB_ONSET = 0.6;
const BULB_OCCLUSION_SOFTNESS = 0.02;
const BULB_SOURCE_SOFTNESS = 0.08;
const BULB_SAMPLE_SPREAD = 0.6;
const BULB_SAMPLES = RING_OFFSETS.length ** 2;

const FAIRY_WING_SPEED = 16;
const FAIRY_WING_CASTER_PLANE_OFFSET = 0.14;
const FAIRY_WING_CASTER_STRENGTH = 0.28;
const FAIRY_BODY_SURFACE = 1.05;
const FAIRY_WING_SURFACE = 0.46;
const FIREFLY_LAMP_EMISSION = 8.4;
const FIREFLY_GLOW = 1.15;
const FIREFLY_TRAIL_GLOW = 0.24;
const FIREFLY_LIGHT_RADIUS = 0.64;
const FAIRY_REFLECTION_POWER = 18;
const FAIRY_REFLECTION_GAIN = 1.1;
const FAIRY_MAX_ROLL = 1.05;
const FAIRY_MAX_PITCH = 0.68;
const FAIRY_BANK_CANT = 0.2;
const FAIRY_POSE_DEPTH_GAIN = 2.2;

const DEMO_MIRROR_LEFT = 0.635;
const DEMO_MIRROR_TOP = 0.018;
const DEMO_MIRROR_RIGHT = 0.998;
const DEMO_MIRROR_BOTTOM = 0.725;
const DEMO_MIRROR_EDGE = 0.012;
const DEMO_MIRROR_GLASS_WARP = 0.0016;
const DEMO_MIRROR_REFLECTION_GAIN = 0.72;
const DEMO_MIRROR_SHADOW_GAIN = 0.34;

const SHADOW_FAR_Z = -1.25;
const SHADOW_STEPS = 32;
const SHADOW_SPAN = 0.3;
const SHADOW_BASELINE = 0.005;
const SHADOW_BIAS = 0.014;
const SHADOW_SLOPE_BIAS = 0.02;
const SHADOW_THICKNESS = 0.7;
const SHADOW_THICKNESS_GROWTH = 2.6;
const SHADOW_SOFTNESS = 0.089;
const SHADOW_GAIN = 2.5;
/** How far above the light plane an occluder may rise before it stops casting */
const SHADOW_FRONT_FADE = 0.2;

export const DepthParams = d.struct({
  outputSize: d.vec2u,
  reset: d.u32,
});

export const RelightParams = d.struct({
  uvTransform: d.mat2x2f,
  lightColor: d.vec4f,
  lightPosition: d.vec2f,
  lightZ: d.f32,
  fairyTime: d.f32,
  fairyHeading: d.f32,
  fairyBank: d.f32,
  fairyPitch: d.f32,
  exposure: d.f32,
  intensity: d.f32,
  relief: d.f32,
  specular: d.f32,
  shadow: d.f32,
  occlusion: d.f32,
  swapAxes: d.u32,
  mirror: d.u32,
  demoMirrorStudy: d.u32,
  mode: d.u32,
  fairyEnabled: d.u32,
});

export const rangeStabilityLayout = tgpu.bindGroupLayout({
  params: { uniform: DepthParams },
  frameRange: { storage: d.vec2f, access: 'readonly' },
  stableRange: { storage: d.vec2f, access: 'mutable' },
});

export const depthPrepareLayout = tgpu.bindGroupLayout({
  params: { uniform: DepthParams },
  disparity: { storage: d.arrayOf(d.vec4f), access: 'readonly' },
  stableRange: { storage: d.vec2f, access: 'readonly' },
  history: { storage: d.arrayOf(d.f32), access: 'mutable' },
});

export const surfaceLayout = tgpu.bindGroupLayout({
  params: { uniform: DepthParams },
  depth: { storage: d.arrayOf(d.f32), access: 'readonly' },
  surface: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
});

export const relightLayout = tgpu.bindGroupLayout({
  params: { uniform: RelightParams },
  surface: { texture: d.texture2d() },
  sampler: { sampler: 'filtering' },
});

export const relightFrameLayout = tgpu.bindGroupLayout({
  frame: { externalTexture: d.textureExternal() },
});

export const stabilizeRangeKernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu';
  const low = rangeStabilityLayout.$.frameRange.x;
  const high = std.max(rangeStabilityLayout.$.frameRange.y, low + 0.001);
  if (rangeStabilityLayout.$.params.reset !== 0) {
    rangeStabilityLayout.$.stableRange = d.vec2f(low, high);
    return;
  }

  const previousLow = rangeStabilityLayout.$.stableRange.x;
  const previousHigh = rangeStabilityLayout.$.stableRange.y;
  rangeStabilityLayout.$.stableRange = d.vec2f(
    std.mix(previousLow, low, RANGE_BLEND),
    std.mix(previousHigh, high, RANGE_BLEND),
  );
});

export const depthPrepareKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [DEPTH_WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const width = depthPrepareLayout.$.params.outputSize.x;
  const index = gid.x;
  if (index >= width * depthPrepareLayout.$.params.outputSize.y) {
    return;
  }

  const low = depthPrepareLayout.$.stableRange.x;
  const span = std.max(depthPrepareLayout.$.stableRange.y - low, 0.001);
  const disparity = depthPrepareLayout.$.disparity[index].x;
  let normalized = d.f32(0);
  if (disparity === disparity) {
    normalized = std.saturate((disparity - low) / span);
  }

  let filtered = d.f32(normalized);
  if (depthPrepareLayout.$.params.reset === 0) {
    const previous = depthPrepareLayout.$.history[index];
    const motion = std.smoothstep(MOTION_LOW, MOTION_HIGH, std.abs(normalized - previous));
    filtered = std.mix(previous, normalized, std.mix(TEMPORAL_ALPHA, MOTION_ALPHA, motion));
  }

  depthPrepareLayout.$.history[index] = filtered;
});

function texelIndex(coord: d.v2i, size: d.v2i): number {
  'use gpu';
  const clamped = std.clamp(coord, d.vec2i(0), size - 1);
  return d.u32(clamped.y) * d.u32(size.x) + d.u32(clamped.x);
}

function depthTexelAt(coord: d.v2i, size: d.v2i): number {
  'use gpu';
  return surfaceLayout.$.depth[texelIndex(coord, size)];
}

function gentlerDelta(backward: number, forward: number): number {
  'use gpu';
  const back = std.abs(backward);
  const front = std.abs(forward);
  return (backward * front + forward * back) / std.max(back + front, 0.000000001);
}

function surfaceSlope(gradient: d.v2f): d.v2f {
  'use gpu';
  const steepness = std.max(std.length(gradient), 0.000000001);
  const shrunk = std.sqrt(std.max(steepness * steepness - GRADIENT_NOISE_ENERGY, 0));
  const ceiling = GRADIENT_LIMIT * std.tanh(shrunk / GRADIENT_LIMIT);
  return gradient * (ceiling / steepness);
}

/** Derives the surface slope and a height-field occlusion term from the depth field */
export const surfaceKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [SURFACE_WORKGROUP_SIZE, SURFACE_WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const size = d.vec2i(surfaceLayout.$.params.outputSize);
  const coord = d.vec2i(gid.xy);
  if (coord.x >= size.x || coord.y >= size.y) {
    return;
  }

  const center = depthTexelAt(coord, size);
  const left = depthTexelAt(coord + d.vec2i(GRADIENT_BACK, 0), size);
  const right = depthTexelAt(coord + d.vec2i(GRADIENT_RADIUS, 0), size);
  const up = depthTexelAt(coord + d.vec2i(0, GRADIENT_BACK), size);
  const down = depthTexelAt(coord + d.vec2i(0, GRADIENT_RADIUS), size);
  const gradient = surfaceSlope(
    d.vec2f(gentlerDelta(center - left, right - center), gentlerDelta(center - up, down - center)) /
      d.f32(GRADIENT_RADIUS),
  );

  let occlusion = d.f32(0);
  for (const radius of tgpu.unroll(OCCLUSION_RADII)) {
    for (const stepY of tgpu.unroll(RING_OFFSETS)) {
      for (const stepX of tgpu.unroll(RING_OFFSETS)) {
        if (stepX !== 0 || stepY !== 0) {
          const neighbor = depthTexelAt(coord + d.vec2i(stepX * radius, stepY * radius), size);
          const difference = neighbor - center;
          const contact = 1 - std.saturate(std.abs(difference) / OCCLUSION_RANGE);
          const cleared = std.max(difference - OCCLUSION_FLOOR, 0);
          occlusion += std.saturate(cleared / OCCLUSION_SCALE) * contact;
        }
      }
    }
  }

  std.textureStore(
    surfaceLayout.$.surface,
    d.vec2u(gid.xy),
    d.vec4f(gradient, 1 - std.saturate(occlusion / d.f32(OCCLUSION_TAPS)), center),
  );
});

function surfaceZ(depth: number): number {
  'use gpu';
  return std.mix(d.f32(SURFACE_FAR_Z), d.f32(NEAR_Z), depth);
}

function shadowZ(depth: number): number {
  'use gpu';
  return std.mix(d.f32(SHADOW_FAR_Z), d.f32(NEAR_Z), depth);
}

function depthAt(uv: d.v2f): number {
  'use gpu';
  return std.textureSampleLevel(relightLayout.$.surface, relightLayout.$.sampler, uv, 0).w;
}

function cameraUvAt(uv: d.v2f): d.v2f {
  'use gpu';
  let sourceSize = d.vec2f(std.textureDimensions(relightFrameLayout.$.frame));
  if (relightLayout.$.params.swapAxes !== 0) {
    sourceSize = d.vec2f(sourceSize.yx);
  }
  let framed = d.vec2f(uv);
  if (relightLayout.$.params.mirror !== 0) {
    framed = d.vec2f(1 - uv.x, uv.y);
  }
  const side = std.min(sourceSize.x, sourceSize.y);
  const sourcePixel = (sourceSize - side) * 0.5 + framed * side - 0.5;
  const clamped = std.clamp(sourcePixel, d.vec2f(0), sourceSize - 1);
  const sourceUv = (clamped + 0.5) / sourceSize;
  return relightLayout.$.params.uvTransform * (sourceUv - d.vec2f(0.5)) + d.vec2f(0.5);
}

function demoMirrorMask(uv: d.v2f): number {
  'use gpu';
  const minimum = d.vec2f(DEMO_MIRROR_LEFT, DEMO_MIRROR_TOP);
  const maximum = d.vec2f(DEMO_MIRROR_RIGHT, DEMO_MIRROR_BOTTOM);
  const inset = std.min(
    std.min(uv.x - minimum.x, maximum.x - uv.x),
    std.min(uv.y - minimum.y, maximum.y - uv.y),
  );
  return std.smoothstep(d.f32(0), d.f32(DEMO_MIRROR_EDGE), inset);
}

function demoMirrorVirtualUv(uv: d.v2f): d.v2f {
  'use gpu';
  const minimum = d.vec2f(DEMO_MIRROR_LEFT, DEMO_MIRROR_TOP);
  const size = d.vec2f(
    DEMO_MIRROR_RIGHT - DEMO_MIRROR_LEFT,
    DEMO_MIRROR_BOTTOM - DEMO_MIRROR_TOP,
  );
  const local = (uv - minimum) / size;
  return d.vec2f(1 - local.x, local.y);
}

function demoMirrorReflectedLampUv(): d.v2f {
  'use gpu';
  const minimum = d.vec2f(DEMO_MIRROR_LEFT, DEMO_MIRROR_TOP);
  const size = d.vec2f(
    DEMO_MIRROR_RIGHT - DEMO_MIRROR_LEFT,
    DEMO_MIRROR_BOTTOM - DEMO_MIRROR_TOP,
  );
  const lamp = fairyLampUv();
  return minimum + d.vec2f(1 - lamp.x, lamp.y) * size;
}

/** A deliberately small screen-space glass displacement, not physical refraction. */
function demoMirrorWarpedUv(uv: d.v2f): d.v2f {
  'use gpu';
  const delta = uv - demoMirrorReflectedLampUv();
  const distance = std.max(std.length(delta), d.f32(0.0001));
  const radial = delta / distance;
  const wave =
    std.sin(distance * 58 - relightLayout.$.params.fairyTime * 2.6) *
    std.exp(0 - distance * 8);
  const shimmer = d.vec2f(
    std.sin(uv.y * 19 + relightLayout.$.params.fairyTime * 0.31),
    std.cos(uv.x * 17 - relightLayout.$.params.fairyTime * 0.27),
  );
  const offset =
    radial * (DEMO_MIRROR_GLASS_WARP * (0.55 + wave * 0.45)) +
    shimmer * (DEMO_MIRROR_GLASS_WARP * 0.22);
  return uv + offset * demoMirrorMask(uv);
}

function dither(uv: d.v2f): number {
  'use gpu';
  const point = uv * 1024;
  return std.fract(52.9829189 * std.fract(0.06711056 * point.x + 0.00583715 * point.y));
}

function shadowFactor(
  origin: d.v3f,
  lightDirection: d.v3f,
  reach: number,
  jitter: number,
  lightZ: number,
): number {
  'use gpu';
  const stride = reach / d.f32(SHADOW_STEPS);
  const baselineTravel = reach * (SHADOW_BASELINE / SHADOW_SPAN);
  const trailProbe = origin - lightDirection * baselineTravel;
  const receiverRise = std.max(
    origin.z - shadowZ(depthAt(trailProbe.xy + d.vec2f(0.5))) - baselineTravel * lightDirection.z,
    d.f32(0),
  );
  const risePerTravel = receiverRise / baselineTravel;

  let occlusion = d.f32(0);
  for (const step of std.range(SHADOW_STEPS)) {
    const travel = (d.f32(step) + jitter) * stride;
    const probe = origin + lightDirection * travel;
    const sampleZ = shadowZ(depthAt(probe.xy + d.vec2f(0.5)));
    const difference = sampleZ - probe.z;
    const bias = SHADOW_BIAS + travel * (SHADOW_SLOPE_BIAS + risePerTravel);
    const thickness = SHADOW_THICKNESS * (1 + (travel / SHADOW_SPAN) * SHADOW_THICKNESS_GROWTH);
    if (difference > bias && difference < thickness) {
      const behindLight =
        1 - std.saturate((sampleZ - lightZ) / SHADOW_FRONT_FADE);
      occlusion += std.saturate((difference - bias) / SHADOW_SOFTNESS) * behindLight;
    }
  }
  return 1 - std.saturate((occlusion / d.f32(SHADOW_STEPS)) * SHADOW_GAIN);
}

function depthRamp(value: number): d.v3f {
  'use gpu';
  const cold = d.vec3f(0.03, 0.02, 0.12);
  const middle = d.vec3f(0.11, 0.45, 0.94);
  const warm = d.vec3f(0.85, 0.36, 0.96);
  const hot = d.vec3f(0.97, 0.97, 0.87);
  if (value < 0.4) {
    return std.mix(cold, middle, value / 0.4);
  }
  if (value < 0.75) {
    return std.mix(middle, warm, (value - 0.4) / 0.35);
  }
  return std.mix(warm, hot, (value - 0.75) / 0.25);
}

function bulbRadius(): number {
  'use gpu';
  return (
    BULB_WORLD_RADIUS *
    ((BULB_CAMERA_Z - BULB_REFERENCE_Z) / (BULB_CAMERA_Z - relightLayout.$.params.lightZ))
  );
}

function bulbExposure(radius: number, center: d.v2f): number {
  'use gpu';
  let open = d.f32(0);
  for (const stepY of tgpu.unroll(RING_OFFSETS)) {
    for (const stepX of tgpu.unroll(RING_OFFSETS)) {
      const probe = center + d.vec2f(stepX, stepY) * (radius * BULB_SAMPLE_SPREAD);
      open += std.smoothstep(
        d.f32(0),
        BULB_SOURCE_SOFTNESS,
        relightLayout.$.params.lightZ - surfaceZ(depthAt(probe)),
      );
    }
  }
  return open / d.f32(BULB_SAMPLES);
}

function bulbSurface(uv: d.v2f, tint: d.v3f, depth: number): d.v4f {
  'use gpu';
  const radius = bulbRadius();
  const spread = std.length(uv - relightLayout.$.params.lightPosition) / radius;
  const limb = std.saturate(spread);
  const dome = std.sqrt(std.max(1 - limb * limb, d.f32(0)));
  const facing = dome * dome;
  const front = relightLayout.$.params.lightZ + BULB_WORLD_RADIUS * dome;
  const solid = std.smoothstep(d.f32(0), BULB_OCCLUSION_SOFTNESS, front - surfaceZ(depth));
  const edge = std.clamp(std.fwidth(spread) * BULB_EDGE, BULB_EDGE_FLOOR, BULB_EDGE_LIMIT);
  const coverage = (1 - std.smoothstep(1 - edge, 1 + edge, spread)) * solid;
  const hue = std.mix(tint, d.vec3f(1), facing * facing);
  return d.vec4f(hue * (BULB_CORE * std.mix(d.f32(BULB_LIMB), d.f32(1), facing)), coverage);
}

function bulbGlow(uv: d.v2f, tint: d.v3f): d.v3f {
  'use gpu';
  const radius = bulbRadius();
  const radii = std.length(uv - relightLayout.$.params.lightPosition) / radius;
  const halo = std.exp(0 - radii / BULB_HALO_SPAN);
  const veil = std.exp(0 - radii / BULB_VEIL_SPAN);
  return tint *
    ((halo * BULB_HALO + veil * BULB_VEIL) *
      bulbExposure(radius, relightLayout.$.params.lightPosition));
}

function bulbPresence(): number {
  'use gpu';
  return std.saturate(relightLayout.$.params.intensity / BULB_ONSET);
}

function fairyForward(): d.v2f {
  'use gpu';
  const displayHeading =
    relightLayout.$.params.fairyHeading +
    relightLayout.$.params.fairyBank * FAIRY_BANK_CANT;
  return d.vec2f(
    std.cos(displayHeading),
    std.sin(displayHeading),
  );
}

/** Projects the camera-facing local fairy plane after longitudinal roll and lateral pitch. */
function fairyProjectedOffset(localOffset: d.v2f): d.v2f {
  'use gpu';
  const roll = relightLayout.$.params.fairyBank * FAIRY_MAX_ROLL;
  const pitch = relightLayout.$.params.fairyPitch * FAIRY_MAX_PITCH;
  const sinRoll = std.sin(roll);
  const cosRoll = std.max(std.cos(roll), d.f32(0.42));
  const sinPitch = std.sin(pitch);
  const cosPitch = std.max(std.cos(pitch), d.f32(0.56));
  return d.vec2f(
    localOffset.x * cosRoll,
    localOffset.y * cosPitch + localOffset.x * sinRoll * sinPitch,
  );
}

function fairyUnprojectedOffset(projectedOffset: d.v2f): d.v2f {
  'use gpu';
  const roll = relightLayout.$.params.fairyBank * FAIRY_MAX_ROLL;
  const pitch = relightLayout.$.params.fairyPitch * FAIRY_MAX_PITCH;
  const sinRoll = std.sin(roll);
  const cosRoll = std.max(std.cos(roll), d.f32(0.42));
  const sinPitch = std.sin(pitch);
  const cosPitch = std.max(std.cos(pitch), d.f32(0.56));
  const localX = projectedOffset.x / cosRoll;
  return d.vec2f(
    localX,
    (projectedOffset.y - localX * sinRoll * sinPitch) / cosPitch,
  );
}

function fairyLocalDepth(localUv: d.v2f): number {
  'use gpu';
  const localOffset = localUv - relightLayout.$.params.lightPosition;
  const roll = relightLayout.$.params.fairyBank * FAIRY_MAX_ROLL;
  const pitch = relightLayout.$.params.fairyPitch * FAIRY_MAX_PITCH;
  return (
    (localOffset.y * std.sin(pitch) -
      localOffset.x * std.sin(roll) * std.cos(pitch)) *
    FAIRY_POSE_DEPTH_GAIN
  );
}

function fairyProjectedUv(localUv: d.v2f): d.v2f {
  'use gpu';
  const center = relightLayout.$.params.lightPosition;
  const forward = fairyForward();
  const right = d.vec2f(0 - forward.y, forward.x);
  const down = d.vec2f(0) - forward;
  const projected = fairyProjectedOffset(localUv - center);
  return center + right * projected.x + down * projected.y;
}

function fairyLocalUv(uv: d.v2f): d.v2f {
  'use gpu';
  const center = relightLayout.$.params.lightPosition;
  const forward = fairyForward();
  const right = d.vec2f(0 - forward.y, forward.x);
  const down = d.vec2f(0) - forward;
  const offset = uv - center;
  return center +
    fairyUnprojectedOffset(d.vec2f(std.dot(offset, right), std.dot(offset, down)));
}

function fairyLampUv(): d.v2f {
  'use gpu';
  const radius = bulbRadius() * 1.15;
  return fairyProjectedUv(
    relightLayout.$.params.lightPosition + d.vec2f(0, radius * 0.22),
  );
}

function fairyLampZ(): number {
  'use gpu';
  const radius = bulbRadius() * 1.15;
  return (
    relightLayout.$.params.lightZ +
    fairyLocalDepth(relightLayout.$.params.lightPosition + d.vec2f(0, radius * 0.22))
  );
}

function fairyWingBeat(): number {
  'use gpu';
  const time = relightLayout.$.params.fairyTime;
  const flutter = std.sin(time * FAIRY_WING_SPEED);
  const glideCycle = 0.5 + std.sin(time * 0.47) * 0.5;
  const glide = std.smoothstep(d.f32(0.72), d.f32(0.94), glideCycle);
  return std.mix(flutter, d.f32(0.82), glide * 0.78);
}

function fireflyPulse(): number {
  'use gpu';
  const time = relightLayout.$.params.fairyTime;
  const breath = 0.5 + std.sin(time * 2.05 + std.sin(time * 0.63) * 0.45) * 0.5;
  const flicker = 0.5 + std.sin(time * 5.7) * 0.5;
  return (
    0.76 + std.smoothstep(d.f32(0.18), d.f32(0.88), breath) * 0.18 + flicker * 0.06
  );
}

function fairyProjectedWingShadow(position: d.v3f, lightPosition: d.v3f): number {
  'use gpu';
  const basePlaneZ = relightLayout.$.params.lightZ - FAIRY_WING_CASTER_PLANE_OFFSET;
  const initialTravel = std.clamp(
    (basePlaneZ - position.z) / std.max(lightPosition.z - position.z, d.f32(0.0001)),
    d.f32(0),
    d.f32(1),
  );
  const initialProbe = fairyLocalUv(
    position.xy + (lightPosition.xy - position.xy) * initialTravel + d.vec2f(0.5),
  );
  const posedPlaneZ = basePlaneZ + fairyLocalDepth(initialProbe);
  const posedTravel = std.clamp(
    (posedPlaneZ - position.z) / std.max(lightPosition.z - position.z, d.f32(0.0001)),
    d.f32(0),
    d.f32(1),
  );
  const probe = fairyLocalUv(
    position.xy + (lightPosition.xy - position.xy) * posedTravel + d.vec2f(0.5),
  );
  const center = relightLayout.$.params.lightPosition;
  const radius = bulbRadius() * 1.15;
  const beat = fairyWingBeat();
  const bank = relightLayout.$.params.fairyBank;
  const wingOpen = 0.34 + std.abs(beat) * 0.66;
  const upperSpread = radius * (0.38 + wingOpen * 0.42);
  const lowerSpread = radius * (0.3 + wingOpen * 0.34);
  const upperY = center.y - radius * 0.16 + beat * radius * 0.07;
  const lowerY = center.y + radius * 0.2 - beat * radius * 0.045;
  const upperRadii = d.vec2f(
    radius * (0.22 + wingOpen * 0.54),
    radius * (0.52 - wingOpen * 0.14),
  );
  const lowerRadii = d.vec2f(
    radius * (0.16 + wingOpen * 0.36),
    radius * (0.31 - wingOpen * 0.07),
  );
  const upperLeft = ellipseMask(
    probe,
    d.vec2f(center.x - upperSpread, upperY + bank * radius * 0.04),
    upperRadii * d.vec2f(1 - bank * 0.07, 1 - bank * 0.04),
  );
  const upperRight = ellipseMask(
    probe,
    d.vec2f(center.x + upperSpread, upperY - bank * radius * 0.04),
    upperRadii * d.vec2f(1 + bank * 0.07, 1 + bank * 0.04),
  );
  const lowerLeft = ellipseMask(
    probe,
    d.vec2f(center.x - lowerSpread, lowerY + bank * radius * 0.03),
    lowerRadii * d.vec2f(1 - bank * 0.05, 1 - bank * 0.03),
  );
  const lowerRight = ellipseMask(
    probe,
    d.vec2f(center.x + lowerSpread, lowerY - bank * radius * 0.03),
    lowerRadii * d.vec2f(1 + bank * 0.05, 1 + bank * 0.03),
  );
  return std.saturate(
    (upperLeft + upperRight) * 0.46 + (lowerLeft + lowerRight) * 0.3,
  ) * (0.78 + std.abs(beat) * 0.22);
}

function shadowReach(shadowToLight: d.v3f): number {
  'use gpu';
  const shadowDistance = std.max(std.length(shadowToLight), d.f32(0.0001));
  return shadowDistance *
    (SHADOW_SPAN / std.max(std.length(shadowToLight.xy), d.f32(SHADOW_SPAN)));
}

function directLight(
  position: d.v3f,
  normal: d.v3f,
  albedo: d.v3f,
  occlusion: number,
  lightPosition: d.v3f,
  tint: d.v3f,
  radius: number,
  intensity: number,
  shadow: number,
): d.v3f {
  'use gpu';
  const toLight = lightPosition - position;
  const distance = std.max(std.length(toLight), d.f32(0.0001));
  const lightDirection = toLight / distance;
  const spread = distance / radius;
  const falloff = 1 / (1 + spread * spread);
  const wrapped = std.saturate((std.dot(normal, lightDirection) + LIGHT_WRAP) / (1 + LIGHT_WRAP));
  const lambert = wrapped * wrapped;
  const halfDirection = std.normalize(lightDirection + d.vec3f(0, 0, 1));
  const lobe = std.pow(std.saturate(std.dot(normal, halfDirection)), d.f32(SPECULAR_POWER));
  const grazing = std.pow(1 - std.saturate(normal.z), d.f32(5));
  const highlight = lobe * (SPECULAR_F0 + (1 - SPECULAR_F0) * grazing);
  let contribution = albedo * tint * (lambert * falloff * shadow * intensity);
  contribution +=
    tint *
    (highlight *
      falloff *
      shadow *
      occlusion *
      relightLayout.$.params.specular *
      intensity);
  return contribution;
}

function reflectiveSurfaceCue(cameraColor: d.v3f): number {
  'use gpu';
  const luminance = std.dot(cameraColor, LUMINANCE_WEIGHTS);
  const brightest = std.max(cameraColor.x, std.max(cameraColor.y, cameraColor.z));
  const darkest = std.min(cameraColor.x, std.min(cameraColor.y, cameraColor.z));
  const neutrality =
    1 - std.saturate((brightest - darkest) / std.max(brightest, d.f32(0.02)));
  const brightCue = std.smoothstep(d.f32(0.22), d.f32(0.88), luminance);
  const detailCue = std.smoothstep(d.f32(0.008), d.f32(0.09), std.fwidth(luminance));
  return brightCue * (0.42 + neutrality * 0.58) * (0.45 + detailCue * 0.75);
}

function fairyReflection(
  position: d.v3f,
  normal: d.v3f,
  cameraColor: d.v3f,
  lightPosition: d.v3f,
  tint: d.v3f,
  radius: number,
  intensity: number,
): d.v3f {
  'use gpu';
  const toLight = lightPosition - position;
  const distance = std.max(std.length(toLight), d.f32(0.0001));
  const lightDirection = toLight / distance;
  const halfDirection = std.normalize(lightDirection + d.vec3f(0, 0, 1));
  const glint = std.pow(
    std.saturate(std.dot(normal, halfDirection)),
    d.f32(FAIRY_REFLECTION_POWER),
  );
  const spread = distance / radius;
  const falloff = 1 / (1 + spread * spread);
  return (
    tint *
    (glint *
      falloff *
      reflectiveSurfaceCue(cameraColor) *
      relightLayout.$.params.specular *
      intensity *
      FAIRY_REFLECTION_GAIN)
  );
}

function ellipseMask(uv: d.v2f, center: d.v2f, radii: d.v2f): number {
  'use gpu';
  const distance = std.length((uv - center) / radii);
  const edge = std.clamp(std.fwidth(distance) * 0.7, d.f32(0.015), d.f32(0.22));
  return 1 - std.smoothstep(1 - edge, 1 + edge, distance);
}

function segmentMask(uv: d.v2f, start: d.v2f, end: d.v2f, width: number): number {
  'use gpu';
  const span = end - start;
  const along = std.clamp(
    std.dot(uv - start, span) / std.max(std.dot(span, span), d.f32(0.0000001)),
    d.f32(0),
    d.f32(1),
  );
  const distance = std.length(uv - (start + span * along));
  const edge = std.clamp(std.fwidth(distance) * 0.8, d.f32(0.0004), width * 0.7);
  return 1 - std.smoothstep(width - edge, width + edge, distance);
}

function fairySurface(uv: d.v2f, tint: d.v3f, depth: number): d.v4f {
  'use gpu';
  const radius = bulbRadius() * 1.15;
  const center = relightLayout.$.params.lightPosition;
  const localUv = fairyLocalUv(uv);
  const beat = fairyWingBeat();
  const bank = relightLayout.$.params.fairyBank;
  const wingOpen = 0.34 + std.abs(beat) * 0.66;
  const upperSpread = radius * (0.38 + wingOpen * 0.42);
  const lowerSpread = radius * (0.3 + wingOpen * 0.34);
  const upperY = center.y - radius * 0.16 + beat * radius * 0.07;
  const lowerY = center.y + radius * 0.2 - beat * radius * 0.045;
  const upperRadii = d.vec2f(
    radius * (0.22 + wingOpen * 0.54),
    radius * (0.52 - wingOpen * 0.14),
  );
  const lowerRadii = d.vec2f(
    radius * (0.16 + wingOpen * 0.36),
    radius * (0.31 - wingOpen * 0.07),
  );
  const upperLeftCenter = d.vec2f(
    center.x - upperSpread,
    upperY + bank * radius * 0.04,
  );
  const upperRightCenter = d.vec2f(
    center.x + upperSpread,
    upperY - bank * radius * 0.04,
  );
  const upperLeftRadii = upperRadii * d.vec2f(1 - bank * 0.07, 1 - bank * 0.04);
  const upperRightRadii = upperRadii * d.vec2f(1 + bank * 0.07, 1 + bank * 0.04);
  const lowerLeftCenter = d.vec2f(
    center.x - lowerSpread,
    lowerY + bank * radius * 0.03,
  );
  const lowerRightCenter = d.vec2f(
    center.x + lowerSpread,
    lowerY - bank * radius * 0.03,
  );
  const lowerLeftRadii = lowerRadii * d.vec2f(1 - bank * 0.05, 1 - bank * 0.03);
  const lowerRightRadii = lowerRadii * d.vec2f(1 + bank * 0.05, 1 + bank * 0.03);
  const upperLeft = ellipseMask(localUv, upperLeftCenter, upperLeftRadii);
  const upperRight = ellipseMask(localUv, upperRightCenter, upperRightRadii);
  const lowerLeft = ellipseMask(localUv, lowerLeftCenter, lowerLeftRadii);
  const lowerRight = ellipseMask(localUv, lowerRightCenter, lowerRightRadii);
  const wings = std.saturate(upperLeft + upperRight + lowerLeft + lowerRight);
  const wingVeins = std.saturate(
    wings -
      0.74 *
        (ellipseMask(localUv, upperLeftCenter, upperLeftRadii * d.vec2f(0.58, 0.54)) +
          ellipseMask(localUv, upperRightCenter, upperRightRadii * d.vec2f(0.58, 0.54))),
  );
  const head = ellipseMask(
    localUv,
    center - d.vec2f(0, radius * 0.43),
    d.vec2f(radius * 0.17, radius * 0.18),
  );
  const hair = ellipseMask(
    localUv,
    center + d.vec2f(radius * 0.14, 0 - radius * 0.52),
    d.vec2f(radius * 0.12, radius * 0.11),
  );
  const torso = segmentMask(
    localUv,
    center - d.vec2f(0, radius * 0.24),
    center + d.vec2f(0, radius * 0.2),
    radius * 0.1,
  );
  const skirt = ellipseMask(
    localUv,
    center + d.vec2f(0, radius * 0.2),
    d.vec2f(radius * 0.24, radius * 0.16),
  );
  const arms = std.saturate(
    segmentMask(
      localUv,
      center - d.vec2f(0, radius * 0.12),
      center + d.vec2f(0 - radius * 0.38, radius * 0.02),
      radius * 0.055,
    ) +
      segmentMask(
        localUv,
        center - d.vec2f(0, radius * 0.1),
        center + d.vec2f(radius * 0.42, 0 - radius * 0.02),
        radius * 0.055,
      ),
  );
  const legs = std.saturate(
    segmentMask(
      localUv,
      center + d.vec2f(0 - radius * 0.08, radius * 0.28),
      center + d.vec2f(0 - radius * 0.2, radius * 0.68),
      radius * 0.05,
    ) +
      segmentMask(
        localUv,
        center + d.vec2f(radius * 0.08, radius * 0.28),
        center + d.vec2f(radius * 0.18, radius * 0.64),
        radius * 0.05,
      ),
  );
  const lampMask = ellipseMask(
    localUv,
    center + d.vec2f(0, radius * 0.22),
    d.vec2f(radius * 0.105, radius * 0.135),
  );
  const bodyMask = std.saturate(head + hair + torso + skirt + arms + legs);
  const wingMask = std.saturate(wings * 0.42 + wingVeins * 0.38) * (1 - bodyMask * 0.74);
  const visibility = std.smoothstep(
    d.f32(0),
    BULB_OCCLUSION_SOFTNESS,
    relightLayout.$.params.lightZ +
      fairyLocalDepth(localUv) +
      radius * 0.35 -
      surfaceZ(depth),
  );
  const coverage = std.saturate(bodyMask + wingMask * 0.42) * visibility;
  const bodyTint = std.mix(d.vec3f(1, 0.4, 0.1), tint, d.f32(0.34));
  const wingTint = std.mix(
    d.vec3f(0.22, 0.84, 1),
    d.vec3f(1, 0.42, 0.86),
    d.f32(0.24 + wingOpen * 0.16),
  );
  const lampTint = std.mix(tint, d.vec3f(0.78, 1, 0.2), d.f32(0.38));
  const emission =
    bodyTint * (FAIRY_BODY_SURFACE * bodyMask) +
    wingTint * (FAIRY_WING_SURFACE * (0.72 + wingOpen * 0.28) * wingMask) +
    lampTint * (FIREFLY_LAMP_EMISSION * fireflyPulse() * lampMask);
  const surfaceWeight = std.max(bodyMask + wingMask + lampMask, d.f32(0.001));
  return d.vec4f(emission / surfaceWeight, coverage);
}

function fairyGlow(uv: d.v2f, tint: d.v3f): d.v3f {
  'use gpu';
  const radius = bulbRadius();
  const center = fairyLampUv();
  const forward = fairyForward();
  const coreHalo = std.exp(0 - std.length(uv - center) / (radius * 0.72));
  const softVeil = std.exp(0 - std.length(uv - center) / (radius * 1.8));
  const firstSpark = center - forward * (radius * 1.55);
  const secondSpark = center - forward * (radius * 2.55);
  const trail =
    std.exp(0 - std.length(uv - firstSpark) / (radius * 0.28)) +
    0.62 * std.exp(0 - std.length(uv - secondSpark) / (radius * 0.2));
  const exposure = bulbExposure(radius, center);
  const lampTint = std.mix(tint, d.vec3f(0.78, 1, 0.2), d.f32(0.38));
  return lampTint *
    ((coreHalo * FIREFLY_GLOW + softVeil * 0.14) * fireflyPulse() +
      trail * FIREFLY_TRAIL_GLOW) *
    exposure;
}

function demoMirrorFairyComposite(uv: d.v2f, lit: d.v3f, tint: d.v3f): d.v3f {
  'use gpu';
  const mask = demoMirrorMask(uv);
  const virtualUv = demoMirrorVirtualUv(uv);
  const reflected = fairySurface(virtualUv, tint, d.f32(0));
  const shadowCaster = fairySurface(
    virtualUv + d.vec2f(0.09, -0.04),
    tint,
    d.f32(0),
  );
  const presence = bulbPresence();
  const reflectionStrength =
    mask *
    presence *
    std.saturate(relightLayout.$.params.specular * DEMO_MIRROR_REFLECTION_GAIN);
  const shadowStrength =
    mask *
    shadowCaster.w *
    relightLayout.$.params.shadow *
    DEMO_MIRROR_SHADOW_GAIN;
  let result = d.vec3f(lit) * (1 - shadowStrength);
  result = std.mix(
    result,
    reflected.xyz * (0.58 + fireflyPulse() * 0.18),
    reflected.w * reflectionStrength,
  );
  result +=
    fairyGlow(virtualUv, tint) *
    (mask * presence * relightLayout.$.params.specular * 0.2);
  return result;
}

function compress(value: number): number {
  'use gpu';
  return (value * (value / (WHITE_POINT * WHITE_POINT) + 1)) / (value + 1);
}

function tonemap(color: d.v3f): d.v3f {
  'use gpu';
  const luminance = std.max(std.dot(color, LUMINANCE_WEIGHTS), d.f32(0.0001));
  const mapped = compress(luminance);
  const shoulder = color / d.vec3f(WHITE_POINT * WHITE_POINT) + d.vec3f(1);
  const perChannel = (color * shoulder) / (color + d.vec3f(1));
  const bleach = std.pow(std.saturate(mapped), d.f32(HIGHLIGHT_BLEACH));
  return std.saturate(std.mix(color * (mapped / luminance), perChannel, bleach));
}

export const relightFragment = tgpu.fragmentFn({
  in: { uv: d.vec2f },
  out: d.vec4f,
})(({ uv }) => {
  'use gpu';
  const cameraColor = std.saturate(
    std.textureSampleBaseClampToEdge(
      relightFrameLayout.$.frame,
      relightLayout.$.sampler,
      cameraUvAt(uv),
    ).rgb,
  );
  if (relightLayout.$.params.mode === RelightMode.CAMERA) {
    return d.vec4f(cameraColor, 1);
  }

  const surface = std.textureSampleLevel(relightLayout.$.surface, relightLayout.$.sampler, uv, 0);
  if (relightLayout.$.params.mode === RelightMode.DEPTH) {
    return d.vec4f(depthRamp(std.saturate(surface.w)), 1);
  }

  const slope = surface.xy * (relightLayout.$.params.relief * RELIEF_SCALE);
  const tilt = d.vec2f(0) - slope / (1 + std.length(slope) * SLOPE_COMPRESSION);
  const normal = std.normalize(d.vec3f(tilt, 1));
  if (relightLayout.$.params.mode === RelightMode.NORMALS) {
    return d.vec4f(normal * 0.5 + 0.5, 1);
  }

  const centered = uv - d.vec2f(0.5);
  const noise = dither(uv);
  const position = d.vec3f(centered, surfaceZ(surface.w));
  let lightUv = d.vec2f(relightLayout.$.params.lightPosition);
  let lightZ = d.f32(relightLayout.$.params.lightZ);
  if (relightLayout.$.params.fairyEnabled !== 0) {
    lightUv = fairyLampUv();
    lightZ = fairyLampZ();
  }
  const lightPosition = d.vec3f(
    lightUv - d.vec2f(0.5),
    lightZ,
  );
  const shadowOrigin = d.vec3f(centered, shadowZ(surface.w));
  let mainShadow = d.f32(1);
  if (relightLayout.$.params.shadow > 0) {
    const shadowToLight = lightPosition - shadowOrigin;
    const shadowDistance = std.max(std.length(shadowToLight), 0.0001);
    const traced = shadowFactor(
      shadowOrigin,
      shadowToLight / shadowDistance,
      shadowReach(shadowToLight),
      noise,
      lightPosition.z,
    );
    mainShadow = std.mix(d.f32(1), traced, relightLayout.$.params.shadow);
  }
  const occlusion = std.mix(d.f32(1), surface.z, relightLayout.$.params.occlusion);

  let sceneColor = d.vec3f(cameraColor);
  if (
    relightLayout.$.params.demoMirrorStudy !== 0 &&
    relightLayout.$.params.fairyEnabled !== 0
  ) {
    const displacedMirrorColor = std.saturate(
      std.textureSampleBaseClampToEdge(
        relightFrameLayout.$.frame,
        relightLayout.$.sampler,
        cameraUvAt(demoMirrorWarpedUv(uv)),
      ).rgb,
    );
    sceneColor = std.mix(
      sceneColor,
      displacedMirrorColor * d.vec3f(0.985, 1, 1.018),
      demoMirrorMask(uv) * 0.32,
    );
  }

  const albedo = std.pow(sceneColor, d.vec3f(GAMMA));
  const tint = d.vec3f(relightLayout.$.params.lightColor.rgb);
  let lit = albedo * AMBIENT_FILL * (relightLayout.$.params.exposure * occlusion);
  const presence = bulbPresence();
  if (relightLayout.$.params.fairyEnabled !== 0) {
    const pulse = fireflyPulse();
    const projectedWings = fairyProjectedWingShadow(position, lightPosition);
    const wingSilhouette =
      1 -
      std.saturate(
        projectedWings *
          FAIRY_WING_CASTER_STRENGTH *
          relightLayout.$.params.shadow,
      );
    const lampTint = std.mix(tint, d.vec3f(0.78, 1, 0.2), d.f32(0.38));
    lit += directLight(
      position,
      normal,
      albedo,
      occlusion,
      lightPosition,
      lampTint,
      FIREFLY_LIGHT_RADIUS,
      relightLayout.$.params.intensity * pulse,
      mainShadow * wingSilhouette,
    );
    lit += fairyReflection(
      position,
      normal,
      sceneColor,
      lightPosition,
      lampTint,
      FIREFLY_LIGHT_RADIUS,
      relightLayout.$.params.intensity * 0.9 * pulse,
    );
    const fairy = fairySurface(uv, tint, surface.w);
    lit = std.mix(lit, fairy.xyz * presence, fairy.w * presence);
    lit += fairyGlow(uv, tint) * presence;
    if (relightLayout.$.params.demoMirrorStudy !== 0) {
      lit = demoMirrorFairyComposite(uv, lit, tint);
    }
  } else {
    lit += directLight(
      position,
      normal,
      albedo,
      occlusion,
      lightPosition,
      tint,
      LIGHT_RADIUS,
      relightLayout.$.params.intensity,
      mainShadow,
    );
    const bulb = bulbSurface(uv, tint, surface.w);
    lit = std.mix(lit, bulb.xyz * presence, bulb.w * presence);
    lit += bulbGlow(uv, tint) * presence;
  }
  const display = std.pow(tonemap(lit), d.vec3f(1 / GAMMA));
  return d.vec4f(display + (noise - 0.5) * DITHER_STEP, 1);
});
