// Instrumented repro for the adaptive-FMM collapse: hooks the page's own
// globals to count metadata refreshes, buffer churn, submit drops, and to
// fingerprint positions + canvas every second (freeze detection).
//
// Usage: node tools/probe_collapse.js <config> [N] [measureSec] [outdir]
//   config: off | fixed | adaptive | adaptive+ahash0 | adaptive+far0 | fixed+funnel
const { chromium } = require('playwright');
const fs = require('fs');

const CONFIG = process.argv[2] || 'adaptive';
const N = parseInt(process.argv[3] || '500000', 10);
const SEC = parseInt(process.argv[4] || '60', 10);
const OUT = process.argv[5] || 'tools/probe_collapse_results.json';

const EXE_CANDIDATES = [
    process.env.CROSSBENCH_EXE,
    'C:\\Users\\Y\\AppData\\Local\\ms-playwright\\chromium-1234\\chrome-win64\\chrome.exe',
    ...fs.readdirSync(process.env.LOCALAPPDATA + '/ms-playwright', { withFileTypes: true })
        .filter((d) => d.isDirectory() && d.name.startsWith('chromium-') && !d.name.includes('headless'))
        .map((d) => `${process.env.LOCALAPPDATA}/ms-playwright/${d.name}/chrome-win64/chrome.exe`),
].filter(Boolean);
const EXE = EXE_CANDIDATES.find((p) => fs.existsSync(p)) || undefined;

const CFGLIST = {
    'off': { fmm: 'off' },
    'fixed': { fmm: 'fixed' },
    'fixed+funnel': { fmm: 'fixed', hash: 'funnel' },
    'adaptive': { fmm: 'adaptive' },
    'adaptive+ahash0': { fmm: 'adaptive', url: 'adaptiveHash=0' },
    'adaptive+far0': { fmm: 'adaptive', url: 'materializedFar=0' },
};
const cfg = CFGLIST[CONFIG] || CFGLIST.adaptive;

(async () => {
    const browser = await chromium.launch({
        headless: true, executablePath: EXE,
        args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-sandbox',
               '--use-gl=angle', '--use-angle=d3d11',
               '--disable-background-timer-throttling', '--disable-renderer-backgrounding'],
    });
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    const consoleAll = [];
    page.on('console', (m) => { consoleAll.push(`[${m.type()}] ${m.text().slice(0, 150)}`); });
    page.on('pageerror', (err) => consoleAll.push('PAGE: ' + err.message.slice(0, 150)));

    const qs = [`preset=120k`, `scenario=galaxy`, `n=${N}`];
    if (cfg.url) qs.push(cfg.url);
    if (process.env.EXTRA_QS) qs.push(process.env.EXTRA_QS);
    await page.goto(`http://localhost:8123/index.html?${qs.join('&')}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForFunction(() => {
        const el = document.getElementById('engineBadge');
        return el && el.innerText && !el.innerText.includes('Detecting');
    }, { timeout: 30000 });
    const badge = await page.evaluate(() => ({
        badge: document.getElementById('engineBadge').innerText,
        gpu: document.getElementById('gpuName') ? document.getElementById('gpuName').innerText : '',
    }));
    console.error('adapter:', JSON.stringify(badge));

    await page.evaluate((t) => {
        if (t.fmm) document.getElementById('selectFmmMode').value = t.fmm;
        if (t.hash) document.getElementById('selectHashMode').value = t.hash;
        document.getElementById('selectFmmMode').dispatchEvent(new Event('change'));
        document.getElementById('selectHashMode').dispatchEvent(new Event('change'));
    }, { fmm: cfg.fmm, hash: cfg.hash || null });
    await page.waitForTimeout(8000);

    const series = await page.evaluate((sec) => {
        const el = (id) => document.getElementById(id);
        const num = (id) => { const v = parseFloat(el(id) ? el(id).innerText : ''); return Number.isFinite(v) ? v : null; };
        // ---- hooks ----
        const hooks = { uploads: 0, lastSizes: '', sizeChanges: 0, refreshes: 0, submits: 0, drops: 0, workerBroken: () => adaptiveMetaWorkerBroken };
        const origUpload = uploadAdaptiveMetadata;
        uploadAdaptiveMetadata = function (md) {
            hooks.uploads++;
            const names = Object.keys(webgpuAdaptiveBuffers).sort();
            const sizes = names.map((k) => webgpuAdaptiveBuffers[k] ? webgpuAdaptiveBuffers[k].size : 0).join(',');
            if (sizes !== hooks.lastSizes) { hooks.sizeChanges++; hooks.lastSizes = sizes; }
            return origUpload.call(this, md);
        };
        const origRefresh = maybeStartAdaptiveRefresh;
        let refreshStarts = 0;
        maybeStartAdaptiveRefresh = function () {
            const before = hooks.refreshes;
            const r = origRefresh.call(this);
            // pending flips true only when a readback actually started
            if (adaptiveRefreshPending && refreshStarts === before) { refreshStarts++; hooks.refreshes++; }
            else if (!adaptiveRefreshPending) { refreshStarts = hooks.refreshes; }
            return r;
        };
        const origSubmit = webgpuDevice.queue.submit.bind(webgpuDevice.queue);
        webgpuDevice.queue.submit = function (b) { hooks.submits++; return origSubmit(b); };
        window.__hooks = hooks;
        // ---- position checksum via dedicated staging copy ----
        let posBuf = null;
        const posChecksum = () => new Promise((resolve) => {
            try {
                const need = Math.ceil((N * 16) / 256) * 256;
                if (!posBuf || posBuf.size < need) {
                    if (posBuf) posBuf.destroy();
                    posBuf = webgpuDevice.createBuffer({ size: need, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
                }
                const src = webgpuPingPong === 0 ? webgpuPosVelBufferA : webgpuPosVelBufferB;
                const enc = webgpuDevice.createCommandEncoder();
                enc.copyBufferToBuffer(src, 0, posBuf, 0, N * 16);
                webgpuDevice.queue.submit([enc.finish()]);
                posBuf.mapAsync(GPUMapMode.READ).then(() => {
                    try {
                        const f = new Float32Array(posBuf.getMappedRange());
                        let sx = 0, sy = 0, mx = 0;
                        const stride = Math.max(1, N >> 12);
                        let k = 0;
                        for (let i = 0; i < N; i += stride) { sx += f[i * 4]; sy += f[i * 4 + 1]; if (f[i * 4 + 2] > mx) mx = f[i * 4 + 2]; k++; }
                        posBuf.unmap();
                        resolve({ cx: +(sx / k).toFixed(5), cy: +(sy / k).toFixed(5), vmax: +mx.toFixed(5), n: k });
                    } catch (e) { try { posBuf.unmap(); } catch (_) {} resolve({ err: String(e).slice(0, 60) }); }
                }).catch((e) => resolve({ err: String(e).slice(0, 60) }));
            } catch (e) { resolve({ err: String(e).slice(0, 60) }); }
        });
        // ---- canvas pixel hash ----
        const sigCanvas = document.createElement('canvas');
        sigCanvas.width = 64; sigCanvas.height = 64;
        const sigCtx = sigCanvas.getContext('2d', { willReadFrequently: true });
        const canvasHash = () => {
            try {
                sigCtx.drawImage(el('mainCanvas'), 0, 0, 64, 64);
                const d = sigCtx.getImageData(0, 0, 64, 64).data;
                let h = 2166136261;
                for (let i = 0; i < d.length; i += 16) { h = (h ^ d[i]) * 16777619 >>> 0; }
                return h;
            } catch (e) { return -1; }
        };
        // ---- rAF fps ----
        let frames = 0, last = 0, maxGap = 0;
        const tick = (t) => { frames++; if (last) { const g = t - last; if (g > maxGap) maxGap = g; } last = t; requestAnimationFrame(tick); };
        requestAnimationFrame(tick);

        const out = [];
        const start = Date.now();
        return new Promise((resolve) => {
            const step = async () => {
                const pos = await posChecksum();
                const md = adaptiveMetadata;
                out.push({
                    t: +((Date.now() - start) / 1000).toFixed(1),
                    rafFps: frames, maxGapMs: +maxGap.toFixed(0), valFPS: num('valFPS'),
                    main: num('valMainCompute'), render: num('valRenderPass'), gpu: num('valTotalGpu'),
                    build: num('valFmmBuild'), m2l: num('valFmmM2l'), l2p: num('valFmmL2p'),
                    stepMs: num('valComputeTime'),
                    uploads: hooks.uploads, sizeChanges: hooks.sizeChanges, refreshes: hooks.refreshes,
                    submits: hooks.submits,
                    totalNodes: md ? md.totalNodes : null,
                    farEntries: md && md.farEntries ? md.farEntries.length : null,
                    listBytes: md ? md.listData.byteLength : null,
                    bufKB: Math.round(Object.values(webgpuAdaptiveBuffers).reduce((a, b) => a + (b ? b.size : 0), 0) / 1024),
                    pos, canvasHash: canvasHash(),
                    pending: adaptiveRefreshPending, workerBroken: hooks.workerBroken(),
                });
                frames = 0; maxGap = 0;
                if (Date.now() - start >= sec * 1000) resolve(out);
                else setTimeout(step, 1000);
            };
            step();
        });
    }, SEC);

    const result = { config: CONFIG, N, sec: SEC, adapter: badge, series, consoleAll: consoleAll.slice(-80) };
    fs.writeFileSync(OUT, JSON.stringify(result, null, 2));
    // Compact console summary
    const drops = consoleAll.filter((c) => c.includes('submit dropped')).length;
    console.log(`\n== ${CONFIG} N=${N} ==`);
    console.log(`adapter: ${badge.gpu} | ${badge.badge}`);
    console.log(`rAFfps series: ${series.map((s) => s.rafFps).join(',')}`);
    console.log(`valFPS series: ${series.map((s) => s.valFPS).join(',')}`);
    console.log(`uploads:${series[series.length - 1].uploads} sizeChanges:${series[series.length - 1].sizeChanges} refreshes:${series[series.length - 1].refreshes} drops:${drops}`);
    console.log(`totalNodes: ${series.map((s) => s.totalNodes).join(',')} farEntries: ${series.map((s) => s.farEntries).join(',')}`);
    console.log(`pos.cx: ${series.map((s) => (s.pos && s.pos.cx != null) ? s.pos.cx : '?').join(',')} vmax: ${series.map((s) => (s.pos && s.pos.vmax != null) ? s.pos.vmax : '?').join(',')}`);
    console.log(`canvasHash: ${series.map((s) => s.canvasHash).join(',')}`);
    console.log(`gpuMs: ${series.map((s) => s.gpu).join(',')} main: ${series.map((s) => s.main).join(',')} stepMs: ${series.map((s) => s.stepMs).join(',')}`);
    await browser.close();
})().catch((e) => { console.error('FAILED:', e); process.exit(1); });
