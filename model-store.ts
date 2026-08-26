export const MODEL_SIZES = ['small'] as const;
export type ModelSize = (typeof MODEL_SIZES)[number];

export interface ModelVariant {
  readonly bundle: string;
  readonly bytes: number;
  readonly megabytes: number;
  readonly precision: 'mixed-fp16' | 'fp32';
  readonly sha256: string;
}

const MODEL_VARIANTS: Record<
  ModelSize,
  { readonly fp16: ModelVariant; readonly f32?: ModelVariant }
> = {
  small: {
    fp16: {
      bundle: 'depthart-relative-s-448-balanced',
      bytes: 13_662_992,
      megabytes: 13,
      precision: 'mixed-fp16',
      sha256: 'e6d7b65bd2888771790d3cc3ad827133f0b014f05010347b6fc6fc891ff9e19c',
    },
    f32: {
      bundle: 'depthart-relative-s-448-f32',
      bytes: 23_994_512,
      megabytes: 23,
      precision: 'fp32',
      sha256: 'adc5352f2fc83d1fd7e740ed32b8a0bd7862cef463a430d23d6071990e822aef',
    },
  },
};

export const RECOMMENDED_MODEL: ModelSize = 'small';

export const MODEL_REVISION = '913a7c13ddfbd48549279555d1db98172e8e5e0d';
const MODEL_HOST = '/models';
const MODEL_CACHE = 'depthart-models';
const CACHE_OPT_OUT_KEY = 'depthart-cache-disabled';

export function modelVariant(size: ModelSize, hasShaderF16: boolean): ModelVariant | undefined {
  const variants = MODEL_VARIANTS[size];
  return hasShaderF16 ? variants.fp16 : variants.f32;
}

export function modelLabel(size: ModelSize, variant: ModelVariant): string {
  const precision = variant.precision === 'mixed-fp16' ? 'FP16' : 'FP32';
  return `${size} ${precision} · ${variant.megabytes} MB`;
}

export function modelUrl(variant: ModelVariant): string {
  return `${MODEL_HOST}/${variant.bundle}.depthart`;
}

async function verifyModel(bytes: ArrayBuffer, variant: ModelVariant): Promise<void> {
  if (bytes.byteLength !== variant.bytes) {
    throw new Error(`Model size mismatch: expected ${variant.bytes}, received ${bytes.byteLength}.`);
  }
  if (!globalThis.crypto?.subtle) {
    throw new Error('Web Crypto is unavailable, so the pinned model checksum cannot be verified.');
  }
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  const actual = Array.from(digest, (byte) => byte.toString(16).padStart(2, '0')).join('');
  if (actual !== variant.sha256) {
    throw new Error(`Model checksum mismatch: expected ${variant.sha256}, received ${actual}.`);
  }
}

export function cachingEnabled(): boolean {
  try {
    return localStorage.getItem(CACHE_OPT_OUT_KEY) === null;
  } catch {
    return true;
  }
}

export function setCachingEnabled(enabled: boolean): void {
  try {
    if (enabled) {
      localStorage.removeItem(CACHE_OPT_OUT_KEY);
    } else {
      localStorage.setItem(CACHE_OPT_OUT_KEY, '1');
    }
  } catch {
    // private-mode storage denial is fine to ignore
  }
}

export async function fetchModel(variant: ModelVariant, signal: AbortSignal): Promise<ArrayBuffer> {
  const url = modelUrl(variant);
  let cache: Cache | undefined;
  try {
    cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(url);
    if (hit) {
      const bytes = await hit.arrayBuffer();
      try {
        await verifyModel(bytes, variant);
        return bytes;
      } catch {
        await cache.delete(url);
      }
    }
  } catch {
    cache = undefined;
  }
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`Model download failed (${response.status}).`);
  }
  const bytes = await response.arrayBuffer();
  await verifyModel(bytes, variant);
  if (cache && cachingEnabled()) {
    await cache
      .put(
        url,
        new Response(bytes.slice(0), {
          headers: { 'Content-Type': 'application/octet-stream' },
          status: 200,
        }),
      )
      .catch(() => undefined);
  }
  return bytes;
}

export async function isModelCached(variant: ModelVariant): Promise<boolean> {
  try {
    const cache = await caches.open(MODEL_CACHE);
    return (await cache.match(modelUrl(variant))) !== undefined;
  } catch {
    return false;
  }
}

export async function clearDownloads(): Promise<void> {
  try {
    await caches.delete(MODEL_CACHE);
  } catch {
    // nothing to clear if Cache Storage is unavailable
  }
}
