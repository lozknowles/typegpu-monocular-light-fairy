import { createReadStream, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:https';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const appRoot = dirname(fileURLToPath(import.meta.url));
const documentRoot = resolve(appRoot, 'dist');
const host = process.env.TYPEGPU_HOST ?? '127.0.0.1';
const port = Number.parseInt(process.env.TYPEGPU_PORT ?? '9443', 10);
const certPath = resolve(
  process.env.TYPEGPU_CERT ?? resolve(appRoot, 'certs', 'preview-server.crt'),
);
const keyPath = resolve(
  process.env.TYPEGPU_KEY ?? resolve(appRoot, 'certs', 'preview-server.key'),
);

const MIME_TYPES = new Map([
  ['.css', 'text/css; charset=utf-8'],
  ['.depthart', 'application/octet-stream'],
  ['.html', 'text/html; charset=utf-8'],
  ['.jpg', 'image/jpeg'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.map', 'application/json; charset=utf-8'],
  ['.pem', 'application/x-pem-file'],
  ['.crt', 'application/x-x509-ca-cert'],
  ['.png', 'image/png'],
  ['.svg', 'image/svg+xml'],
  ['.txt', 'text/plain; charset=utf-8'],
]);

function resolveRequestPath(rawUrl) {
  const pathname = decodeURIComponent(new URL(rawUrl ?? '/', 'https://preview.invalid').pathname);
  const relativePath = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '');
  const candidate = resolve(documentRoot, relativePath);
  if (candidate !== documentRoot && !candidate.startsWith(`${documentRoot}${sep}`)) {
    return undefined;
  }
  return candidate;
}

function setSecurityHeaders(response) {
  response.setHeader(
    'Content-Security-Policy',
    "default-src 'self'; base-uri 'none'; connect-src 'self'; form-action 'none'; frame-ancestors 'none'; img-src 'self' blob: data:; media-src 'self' blob:; object-src 'none'; script-src 'self'; style-src 'self'; worker-src 'self' blob:",
  );
  response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  response.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  response.setHeader('Permissions-Policy', 'camera=(self), microphone=(), geolocation=()');
  response.setHeader('Referrer-Policy', 'no-referrer');
  response.setHeader('X-Content-Type-Options', 'nosniff');
}

const server = createServer(
  { cert: readFileSync(certPath), key: readFileSync(keyPath), minVersion: 'TLSv1.2' },
  (request, response) => {
    setSecurityHeaders(response);
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      response.writeHead(405, { Allow: 'GET, HEAD' });
      response.end('Method not allowed.');
      return;
    }

    const filePath = resolveRequestPath(request.url);
    if (!filePath) {
      response.writeHead(400);
      response.end('Invalid path.');
      return;
    }

    let fileStat;
    try {
      fileStat = statSync(filePath);
    } catch {
      response.writeHead(404);
      response.end('Not found.');
      return;
    }
    if (!fileStat.isFile()) {
      response.writeHead(404);
      response.end('Not found.');
      return;
    }

    const extension = extname(filePath).toLowerCase();
    response.setHeader('Content-Type', MIME_TYPES.get(extension) ?? 'application/octet-stream');
    response.setHeader('Content-Length', fileStat.size);
    response.setHeader(
      'Cache-Control',
      extension === '.html' ? 'no-store' : 'private, max-age=31536000, immutable',
    );
    response.writeHead(200);
    if (request.method === 'HEAD') {
      response.end();
      return;
    }
    createReadStream(filePath).pipe(response);
  },
);

server.listen(port, host, () => {
  process.stdout.write(`TypeGPU preview listening on https://${host}:${port}/\n`);
});

function shutdown() {
  server.close(() => process.exit(0));
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
