import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';

const directories = [
  join('dist', 'client', '_next', 'static', 'chunks'),
  join('dist', 'client', '_next', 'static', 'css'),
];
let gzipBytes = 0;
for (const directory of directories) {
  for (const file of readdirSync(directory)) {
    if (file.endsWith('.js') || file.endsWith('.css')) {
      gzipBytes += gzipSync(readFileSync(join(directory, file))).length;
    }
  }
}
const limit = 350 * 1024;
console.log(`Client JS + CSS gzip: ${(gzipBytes / 1024).toFixed(1)} KiB / 350.0 KiB`);
if (gzipBytes > limit) process.exitCode = 1;
