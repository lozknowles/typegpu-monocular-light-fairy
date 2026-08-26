import { createReadStream } from 'node:fs';
import { copyFile, mkdir, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const sourceRoot = fileURLToPath(new URL('./models/', import.meta.url));
const destinationRoot = fileURLToPath(new URL('./dist/models/', import.meta.url));
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

async function sha256(filePath) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(filePath)) {
    hash.update(chunk);
  }
  return hash.digest('hex');
}

await mkdir(destinationRoot, { recursive: true });
for (const pinned of pinnedFiles) {
  const source = resolve(sourceRoot, pinned.name);
  const sourceStat = await stat(source);
  const sourceHash = await sha256(source);
  if (sourceStat.size !== pinned.bytes || sourceHash !== pinned.sha256) {
    throw new Error(
      `Pinned model artifact mismatch for ${pinned.name}: ${sourceStat.size} bytes, ${sourceHash}`,
    );
  }
  const destination = resolve(destinationRoot, pinned.name);
  await copyFile(source, destination);
  const destinationHash = await sha256(destination);
  if (destinationHash !== pinned.sha256) {
    throw new Error(`Staged model artifact mismatch for ${pinned.name}: ${destinationHash}`);
  }
}

process.stdout.write(`Staged and verified ${pinnedFiles.length} pinned model artifacts.\n`);
