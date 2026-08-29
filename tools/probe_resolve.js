// Emulate the WGSL resolveLeafNode (walk-up + terminal descent) in JS against
// the page's live adaptiveMetadata, and compare with leafForParticle for a
// sample of live particle positions. Finds where the directory path diverges.
const { chromium } = require('playwright');
const fs = require('fs');
const EXE = [
    'C:/Users/Y/AppData/Local/ms-playwright/chromium-1234/chrome-win64/chrome.exe',
    ...fs.readdirSync(process.env.LOCALAPPDATA + '/ms-playwright', { withFileTypes: true })
        .filter((d) => d.isDirectory() && d.name.startsWith('chromium-') && !d.name.includes('headless'))
        .map((d) => `${process.env.LOCALAPPDATA}/ms-playwright/${d.name}/chrome-win64/chrome.exe`),
].filter((p) => fs.existsSync(p));

const N = process.argv[2] || '500000';
const RUN_SEC = parseInt(process.argv[3] || '15', 10);

(async () => {
    const browser = await chromium.launch({
        headless: true, executablePath: EXE[0],
        args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-sandbox',
               '--use-gl=angle', '--use-angle=d3d11', '--disable-background-timer-throttling', '--disable-renderer-backgrounding'],
    });
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    await page.goto(`http://localhost:8123/index.html?preset=120k&scenario=galaxy&n=${N}`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
        const el = document.getElementById('engineBadge');
        return el && el.innerText && !el.innerText.includes('Detecting');
    }, { timeout: 30000 });
    await page.evaluate(() => {
        document.getElementById('selectFmmMode').value = 'adaptive';
        document.getElementById('selectFmmMode').dispatchEvent(new Event('change'));
    });
    await page.waitForTimeout(RUN_SEC * 1000);

    const out = await page.evaluate(() => {
        const md = adaptiveMetadata;
        const H = md.nodeHashPacked;
        const EMPTY = 0xFFFFFFFF;
        function h32(k) {
            k ^= 0x9E3779B9; k ^= k >>> 16; k = Math.imul(k, 0x7FEB352D);
            k ^= k >>> 15; k = Math.imul(k, 0x846CA68B); k ^= k >>> 16; return k >>> 0;
        }
        const alpha = H[0], beta = H[1], bOff = H[2], bSize = H[3], cOff = H[4], cNum = H[5], cSlots = H[6], bAtt = H[7], totalSize = H[9];
        const geomSize = 11 + 3 * alpha + bAtt + 2;
        const keysAt = (i) => H[geomSize + i];
        const valsAt = (i) => H[geomSize + totalSize + i];
        function probe(key) {
            for (let i = 0; i < alpha; i++) {
                const aCount = H[11 + alpha + i], slab = H[11 + i], salt = H[11 + 2 * alpha + i];
                const base = slab + (h32(key ^ salt) % aCount) * beta;
                for (let s = 0; s < beta; s++) { const k = keysAt(base + s); if (k === key) return valsAt(base + s); if (k === EMPTY) return -1; }
            }
            if (bSize > 0) for (let t = 0; t < bAtt; t++) { const p = bOff + (h32(key ^ H[11 + 3 * alpha + t]) % bSize); const k = keysAt(p); if (k === key) return valsAt(p); if (k === EMPTY) return -1; }
            if (cNum > 0) {
                const b1 = h32(key ^ H[11 + 3 * alpha + bAtt]) % cNum, b2 = h32(key ^ H[11 + 3 * alpha + bAtt + 1]) % cNum;
                for (let t = 0; t < cSlots; t++) {
                    const pa = cOff + b1 * cSlots + t, ka = keysAt(pa); if (ka === key) return valsAt(pa); if (ka === EMPTY) return -1;
                    const pb = cOff + b2 * cSlots + t, kb = keysAt(pb); if (kb === key) return valsAt(pb); if (kb === EMPTY) return -1;
                }
            }
            return -1;
        }
        const isTerm = (i) => (md.nodeMeta[i * 2 + 1] & 1) !== 0;
        // Pure root-descent leaf (what the builder's stack walk computes).
        function descendLeaf(px, py) {
            let node = 0;
            while (!isTerm(node)) {
                const cs = [md.nodeChildren[node * 4], md.nodeChildren[node * 4 + 1], md.nodeChildren[node * 4 + 2], md.nodeChildren[node * 4 + 3]];
                const cx = md.nodeCenterSize[node * 4], cy = md.nodeCenterSize[node * 4 + 1];
                let child = cs[(px >= cx ? 1 : 0) + (py >= cy ? 2 : 0)];
                if (child === EMPTY) { child = cs.find((v) => v !== EMPTY); if (child === undefined) return node; }
                node = child;
            }
            return node;
        }
        function resolveLeaf(px, py) {
            let lvl = md.numLevels - 1;
            let ix = Math.min((Math.max(0, Math.min(0.999999, px)) * (1 << lvl)) | 0, (1 << lvl) - 1);
            let iy = Math.min((Math.max(0, Math.min(0.999999, py)) * (1 << lvl)) | 0, (1 << lvl) - 1);
            let n = -1;
            for (;;) {
                n = probe((lvl << 20) | (iy << 10) | ix);
                if (n !== -1) break;
                if (lvl === 0) return -1;
                lvl--; ix >>= 1; iy >>= 1;
            }
            let node = n, steps = 0;
            while (steps++ < 12) {
                if (isTerm(node)) return node;
                const cs = [md.nodeChildren[node * 4], md.nodeChildren[node * 4 + 1], md.nodeChildren[node * 4 + 2], md.nodeChildren[node * 4 + 3]];
                const cx = md.nodeCenterSize[node * 4], cy = md.nodeCenterSize[node * 4 + 1];
                let child = cs[(px >= cx ? 1 : 0) + (py >= cy ? 2 : 0)];
                if (child === EMPTY) { child = cs.find((v) => v !== EMPTY); if (child === undefined) return node; }
                node = child;
            }
            return node;
        }
        // Read live positions from the current input buffer.
        const need = Math.ceil((N * 16) / 256) * 256;
        const buf = webgpuDevice.createBuffer({ size: need, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const src = webgpuPingPong === 0 ? webgpuPosVelBufferA : webgpuPosVelBufferB;
        const enc = webgpuDevice.createCommandEncoder();
        enc.copyBufferToBuffer(src, 0, buf, 0, N * 16);
        webgpuDevice.queue.submit([enc.finish()]);
        return buf.mapAsync(GPUMapMode.READ).then(() => {
            const f = new Float32Array(buf.getMappedRange());
            const stride = Math.max(1, N >> 9);
            let total = 0, mismatch = 0, nonTerminal = 0, notContaining = 0, invalid = 0;
            let descentAgrees = 0, descentVsLeafFor = 0, resolvedEqLeafForLvl = 0;
            let dzMax = 0, dzSum = 0, list1Sum = 0, list1SumLf = 0, mismatchList1 = 0;
            const examples = [];
            for (let i = 0; i < N; i += stride) {
                const px = f[i * 4], py = f[i * 4 + 1];
                const r = resolveLeaf(px, py);
                const d = descendLeaf(px, py);
                const lf = md.leafForParticle[i];
                total++;
                if (r === d) descentAgrees++;
                if (d === lf) descentVsLeafFor++;
                if (md.nodeCenterSize[r * 4 + 3] === md.nodeCenterSize[lf * 4 + 3]) resolvedEqLeafForLvl++;
                if (r === -1) { invalid++; continue; }
                if (!isTerm(r)) nonTerminal++;
                const cx = md.nodeCenterSize[r * 4], cy = md.nodeCenterSize[r * 4 + 1], size = md.nodeCenterSize[r * 4 + 2];
                if (Math.abs(px - cx) > size / 2 + 1e-9 || Math.abs(py - cy) > size / 2 + 1e-9) {
                    notContaining++;
                    if (examples.length < 5) examples.push({ i, px: +px.toFixed(4), py: +py.toFixed(4), r, cx: +cx.toFixed(4), cy: +cy.toFixed(4), size, lvl: md.nodeCenterSize[r * 4 + 3] });
                }
                const dz = Math.hypot(px - cx, py - cy) / size;
                dzSum += dz; if (dz > dzMax) dzMax = dz;
                if (r !== lf) { mismatch++; mismatchList1 += md.listCounts[r * 4]; }
                list1Sum += md.listCounts[r * 4];
                list1SumLf += md.listCounts[lf * 4];
            }
            buf.destroy();
            return {
                totalNodes: md.totalNodes, numLevels: md.numLevels, depth: md.depth,
                sampled: total, mismatchVsLeafFor: mismatch, nonTerminal, notContaining, invalid,
                resolveEqDescent: descentAgrees, descentEqLeafFor: descentVsLeafFor, sameLevelAsLeafFor: resolvedEqLeafForLvl,
                dzMean: +(dzSum / Math.max(1, total - invalid)).toFixed(3), dzMax: +dzMax.toFixed(3),
                list1MeanResolved: +(list1Sum / Math.max(1, total - invalid)).toFixed(1),
                list1MeanLeafFor: +(list1SumLf / Math.max(1, total)).toFixed(1),
                mismatchList1Mean: +(mismatchList1 / Math.max(1, mismatch)).toFixed(1),
                examples,
            };
        });
    });
    console.log(JSON.stringify(out, null, 2));
    await browser.close();
})().catch((e) => { console.error('FAILED:', e); process.exit(1); });
