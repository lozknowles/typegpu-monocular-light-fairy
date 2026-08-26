import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = 'reczkok/depthart-typegpu';
const revision = '913a7c13ddfbd48549279555d1db98172e8e5e0d';
const modelRoot = fileURLToPath(new URL('../models/', import.meta.url));
const pinnedFiles = [
  {
    name: 'depthart-relative-s-448-balanced.depthart',
    bytes: 13_662_992,
    sha256: 'e6d7b65bd2888771790d3cc3ad827133f0b014f05010347b6fc6fc891ff9e19c',
  },
  {
    name: 'depthart-relative-s-448-f32.depthart',
    bytes: 23_994_512,
    sha256: 'adc5352f2fc83d1fd7e740ed32b8a0bd7862cef463a430d23d6071990e822aef',
  },
  {
    name: 'LICENSE',
    bytes: 11_358,
    sha256: 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
  },
  {
    name: 'NOTICE',
    bytes: 2_800,
    sha256: '2b0528b970b69f7cbc63d30f6c4e6cb68db54f1fdb7ec87205316fe9d7f8dfa3',
  },
  {
    name: 'README.md',
    bytes: 2_600,
    sha256: '5d8fd51d1a28b55128a4e521019e70ca2e52068dfa11479e0c04b02745200856',
  },
];

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

async function verify(path, pinned) {
  const metadata = await stat(path);
  const bytes = await readFile(path);
  const digest = sha256(bytes);
  if (metadata.size !== pinned.bytes || digest !== pinned.sha256) {
    throw new Error(
      `Pinned artifact mismatch for ${pinned.name}: ${metadata.size} bytes, ${digest}`,
    );
  }
}

await mkdir(modelRoot, { recursive: true });

for (const pinned of pinnedFiles) {
  const destination = resolve(modelRoot, pinned.name);
  try {
    await verify(destination, pinned);
    process.stdout.write(`Verified cached ${pinned.name}\n`);
    continue;
  } catch (error) {
    if (error?.code !== 'ENOENT') {
      throw error;
    }
  }

  const url =
    `https://huggingface.co/${repository}/resolve/${revision}/` +
    `${encodeURIComponent(pinned.name)}?download=true`;
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok) {
    throw new Error(`Download failed for ${pinned.name}: HTTP ${response.status}`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength !== pinned.bytes || sha256(bytes) !== pinned.sha256) {
    throw new Error(`Downloaded artifact failed its pin for ${pinned.name}`);
  }

  const partial = `${destination}.part`;
  await rm(partial, { force: true });
  await writeFile(partial, bytes, { flag: 'wx' });
  await rename(partial, destination);
  await verify(destination, pinned);
  process.stdout.write(`Downloaded and verified ${pinned.name}\n`);
}

process.stdout.write(`Model revision ${revision} is complete and verified.\n`);
