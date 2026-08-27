import { LIGHT_Z_MAX, LIGHT_Z_MIN, defaultRelightingSettings } from './renderer.ts';

const FLIGHT_PHASE_RATE = 0.00036;
const FLIGHT_START_PHASE = 2.35;
const FLIGHT_SAMPLE_MS = 70;
const FLIGHT_BANK_GAIN = 5.5;
const FLIGHT_PITCH_GAIN = 2.4;
const FLIGHT_RESUME_MS = 900;
const WHEEL_STEP_LIMIT = 60;
const WHEEL_SENSITIVITY = 0.0015;
const PINCH_SENSITIVITY = 0.004;
const LIGHT_GRAB_RADIUS = 0.08;
const TAP_SLOP = 0.012;

const LightControl = {
  FLIGHT: 'flight',
  PINNED: 'pinned',
} as const;
type LightControl = (typeof LightControl)[keyof typeof LightControl];

type Gesture =
  | { readonly kind: 'none' }
  | { readonly kind: 'press'; readonly grabbed: boolean; readonly x: number; readonly y: number }
  | { readonly kind: 'drag' }
  | { readonly kind: 'pinch'; span: number };

interface LightUpdate {
  lightPosition?: [number, number];
  lightZ?: number;
  fairyHeading?: number;
  fairyBank?: number;
  fairyPitch?: number;
}

interface LightInput {
  readonly lightPosition: [number, number];
  readonly lightZ: number;
  readonly fairyHeading: number;
  readonly fairyBank: number;
  readonly fairyPitch: number;
  /** Advances autonomous swooping flight; call once per rendered frame */
  flightTick(): void;
}

interface FlightSample {
  readonly x: number;
  readonly y: number;
  readonly zOffset: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function smoothstep(value: number): number {
  const clamped = clamp(value, 0, 1);
  return clamped * clamped * (3 - 2 * clamped);
}

function wrapAngle(angle: number): number {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

function mixAngle(from: number, to: number, amount: number): number {
  return from + wrapAngle(to - from) * amount;
}

/** A wide, non-repeating harmonic path with climbs, dives, and speed variation. */
function sampleFlight(now: number): FlightSample {
  const phase = FLIGHT_START_PHASE + now * FLIGHT_PHASE_RATE;
  return {
    x: 0.5 + Math.cos(phase) * 0.29 + Math.cos(phase * 2.2 + 0.9) * 0.075,
    y: 0.46 + Math.sin(phase * 1.58 + 0.4) * 0.16 + Math.sin(phase * 0.61 - 0.8) * 0.065,
    zOffset: Math.sin(phase * 0.72 + 0.4) * 0.11 + Math.cos(phase * 1.83) * 0.035,
  };
}

function headingBetween(from: FlightSample, to: FlightSample): number {
  return Math.atan2(to.y - from.y, to.x - from.x);
}

/** Tap or drag pins the light; releasing it resumes autonomous banked flight. */
export function setupLightInput(
  canvas: HTMLCanvasElement,
  onChange: (update: LightUpdate) => void,
  signal: AbortSignal,
): LightInput {
  const pointers = new Map<number, { x: number; y: number }>();
  let gesture: Gesture = { kind: 'none' };
  let control: LightControl = LightControl.FLIGHT;
  let lightPosition: [number, number] = [...defaultRelightingSettings.lightPosition];
  let lightZ = defaultRelightingSettings.lightZ;
  let lightZAnchor = lightZ;
  let fairyHeading = defaultRelightingSettings.fairyHeading;
  let fairyBank = defaultRelightingSettings.fairyBank;
  let fairyPitch = defaultRelightingSettings.fairyPitch;
  let flightResumeAt = performance.now();
  let flightResumePosition: [number, number] = [...lightPosition];
  let flightResumeZ = lightZ;
  let flightResumeHeading = fairyHeading;
  let flightResumeBank = fairyBank;
  let flightResumePitch = fairyPitch;

  function placeLight(x: number, y: number): void {
    const next: [number, number] = [clamp(x, 0, 1), clamp(y, 0, 1)];
    const distance = Math.hypot(next[0] - lightPosition[0], next[1] - lightPosition[1]);
    if (distance > 0.001) {
      fairyHeading = Math.atan2(next[1] - lightPosition[1], next[0] - lightPosition[0]);
    }
    fairyBank = 0;
    fairyPitch = 0;
    lightPosition = next;
    onChange({ lightPosition, fairyHeading, fairyBank, fairyPitch });
  }

  function pushLight(amount: number): void {
    lightZ = clamp(lightZ + amount, LIGHT_Z_MIN, LIGHT_Z_MAX);
    lightZAnchor = clamp(lightZAnchor + amount, LIGHT_Z_MIN, LIGHT_Z_MAX);
    onChange({ lightZ });
  }

  function resumeFlight(): void {
    control = LightControl.FLIGHT;
    flightResumeAt = performance.now();
    flightResumePosition = [...lightPosition];
    flightResumeZ = lightZ;
    flightResumeHeading = fairyHeading;
    flightResumeBank = fairyBank;
    flightResumePitch = fairyPitch;
  }

  function pinLight(pinned: boolean): void {
    if (pinned) {
      control = LightControl.PINNED;
    } else {
      resumeFlight();
    }
  }

  function canvasFraction(event: PointerEvent): { x: number; y: number } | undefined {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return undefined;
    }
    return {
      x: (event.clientX - rect.left) / rect.width,
      y: (event.clientY - rect.top) / rect.height,
    };
  }

  function overLight(point: { x: number; y: number }): boolean {
    return Math.hypot(point.x - lightPosition[0], point.y - lightPosition[1]) <= LIGHT_GRAB_RADIUS;
  }

  function pinchSpan(): number {
    const [first, second] = [...pointers.values()];
    if (!first || !second) {
      return 0;
    }
    return Math.hypot(first.x - second.x, first.y - second.y);
  }

  /** A touch defers placement to the drag or the release, so a pinch's first finger cannot fling the light */
  function beginGesture(event: PointerEvent): void {
    canvas.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.size >= 2) {
      gesture = { kind: 'pinch', span: pinchSpan() };
      return;
    }
    const point = canvasFraction(event);
    if (!point) {
      return;
    }
    const grabbed = overLight(point);
    gesture = { kind: 'press', grabbed, x: point.x, y: point.y };
    if (!grabbed && event.pointerType !== 'touch') {
      placeLight(point.x, point.y);
      pinLight(true);
    }
  }

  function continueGesture(event: PointerEvent): void {
    if (pointers.has(event.pointerId)) {
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    }
    if (gesture.kind === 'pinch') {
      const span = pinchSpan();
      if (gesture.span > 0) {
        pushLight((span - gesture.span) * PINCH_SENSITIVITY);
      }
      gesture.span = span;
      return;
    }

    const point = canvasFraction(event);
    if (!point) {
      return;
    }
    switch (gesture.kind) {
      case 'press':
        if (Math.hypot(point.x - gesture.x, point.y - gesture.y) > TAP_SLOP) {
          gesture = { kind: 'drag' };
          pinLight(true);
          placeLight(point.x, point.y);
        }
        break;
      case 'drag':
        placeLight(point.x, point.y);
        break;
      case 'none':
        canvas.style.cursor = overLight(point) ? 'grab' : 'crosshair';
        break;
    }
  }

  function endGesture(event: PointerEvent): void {
    pointers.delete(event.pointerId);
    if (gesture.kind === 'pinch') {
      gesture.span = pinchSpan();
    }
    if (pointers.size > 0) {
      return;
    }
    if (gesture.kind === 'press' && event.type === 'pointerup') {
      if (gesture.grabbed) {
        pinLight(control !== LightControl.PINNED);
      } else if (event.pointerType === 'touch') {
        placeLight(gesture.x, gesture.y);
        pinLight(true);
      }
    }
    gesture = { kind: 'none' };
  }

  function enterCanvas(): void {
    canvas.style.cursor = 'crosshair';
  }

  function leaveCanvas(): void {
    canvas.style.cursor = 'crosshair';
  }

  function pushLightFromWheel(event: WheelEvent): void {
    event.preventDefault();
    let delta = event.deltaY;
    if (event.deltaMode === WheelEvent.DOM_DELTA_LINE) {
      delta *= 16;
    } else if (event.deltaMode === WheelEvent.DOM_DELTA_PAGE) {
      delta *= canvas.clientHeight;
    }
    delta = Math.sign(delta) * Math.min(Math.abs(delta), WHEEL_STEP_LIMIT);
    pushLight(delta * WHEEL_SENSITIVITY);
  }

  canvas.addEventListener('pointerdown', beginGesture, { signal });
  canvas.addEventListener('pointermove', continueGesture, { signal });
  canvas.addEventListener('pointerup', endGesture, { signal });
  canvas.addEventListener('pointercancel', endGesture, { signal });
  canvas.addEventListener('pointerenter', enterCanvas, { signal });
  canvas.addEventListener('pointerleave', leaveCanvas, { signal });
  canvas.addEventListener('wheel', pushLightFromWheel, { passive: false, signal });

  return {
    get lightPosition() {
      return lightPosition;
    },
    get lightZ() {
      return lightZ;
    },
    get fairyHeading() {
      return fairyHeading;
    },
    get fairyBank() {
      return fairyBank;
    },
    get fairyPitch() {
      return fairyPitch;
    },
    flightTick() {
      if (control !== LightControl.FLIGHT) {
        return;
      }
      const now = performance.now();
      const previous = sampleFlight(now - FLIGHT_SAMPLE_MS);
      const target = sampleFlight(now);
      const next = sampleFlight(now + FLIGHT_SAMPLE_MS);
      const targetHeading = headingBetween(previous, next);
      const targetBank = clamp(
        wrapAngle(headingBetween(target, next) - headingBetween(previous, target)) *
          FLIGHT_BANK_GAIN,
        -1,
        1,
      );
      const screenTravel = Math.max(
        Math.hypot(next.x - previous.x, next.y - previous.y),
        0.0001,
      );
      // Local forward is -Y in the shader, so a negative pitch leads a move towards the camera.
      const targetPitch = clamp(
        (0 - (next.zOffset - previous.zOffset) / screenTravel) * FLIGHT_PITCH_GAIN,
        -1,
        1,
      );
      const targetZ = clamp(lightZAnchor + target.zOffset, LIGHT_Z_MIN, LIGHT_Z_MAX);
      const blend = smoothstep((now - flightResumeAt) / FLIGHT_RESUME_MS);
      lightPosition = [
        flightResumePosition[0] + (target.x - flightResumePosition[0]) * blend,
        flightResumePosition[1] + (target.y - flightResumePosition[1]) * blend,
      ];
      lightZ = flightResumeZ + (targetZ - flightResumeZ) * blend;
      fairyHeading = mixAngle(flightResumeHeading, targetHeading, blend);
      fairyBank = flightResumeBank + (targetBank - flightResumeBank) * blend;
      fairyPitch = flightResumePitch + (targetPitch - flightResumePitch) * blend;
      onChange({ lightPosition, lightZ, fairyHeading, fairyBank, fairyPitch });
    },
  };
}
