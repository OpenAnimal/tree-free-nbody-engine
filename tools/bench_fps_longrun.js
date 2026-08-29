// Long-duration FPS stability bench for the shipped index.html.
//
// Why this exists: tools/browser_crossbench.js samples 5s windows, which
// cannot see performance collapse that develops as the galaxy evolves (core
// collapse grows the adaptive tree over tens of seconds). This harness runs
// each far-field mode for ~2 minutes under the page's DEFAULT vsync-locked
// scheduler (mirroring what a user sees), records per-second FPS, frame-gap
// outlier counts, the page's own per-pass FMM timings (valFmmBuild/M2L/L2P/
// MainCompute/Render/TotalGPU), diagChannel errors, and a coarse canvas
// luminance-centroid signature (mass centroid + spread) so trajectory
// divergence between modes becomes a measured curve instead of an eyeball
// claim.
//
// Usage:  node tools/bench_fps_longrun.js [N] [measureSec]
//   env:  CONFIG=<label> runs one config; UNCAPPED=1 adds ?uncapped=1;
//         WARMUP_SEC, PORT.
// Requires the repo on http://localhost:8123 (python -m http.server 8123).
//
// Output: one JSON line per config (full per-second series) on stdout, a
// markdown summary at the end, and tools/bench_fps_longrun_results.json.

const { chromium } = require('playwright');
const fs = require('fs');

const N = parseInt(process.argv[2] || '120000', 10);
const MEASURE_SEC = parseInt(process.argv[3] || '120', 10);
const WARMUP_MS = (parseInt(process.env.WARMUP_SEC || '8', 10)) * 1000;
const MEASURE_MS = MEASURE_SEC * 1000;
const PORT = process.env.PORT || '8123';
const BASE = `http://localhost:${PORT}/index.html`;

// Full Chromium (not headless-shell — no WebGPU adapter there). Same
// discovery chain as browser_crossbench.js.
const EXE_CANDIDATES = [
    process.env.CROSSBENCH_EXE,
    'C:\\Users\\Y\\AppData\\Local\\ms-playwright\\chromium-1234\\chrome-win64\\chrome.exe',
    ...fs.readdirSync(process.env.LOCALAPPDATA + '/ms-playwright', { withFileTypes: true })
        .filter((d) => d.isDirectory() && d.name.startsWith('chromium-') && !d.name.includes('headless'))
        .map((d) => `${process.env.LOCALAPPDATA}/ms-playwright/${d.name}/chrome-win64/chrome.exe`),
].filter(Boolean);
const EXE = EXE_CANDIDATES.find((p) => fs.existsSync(p)) || undefined;

const CONFIGS = [
    { label: 'off',             fmm: 'off',      hash: null },
    { label: 'direct',          fmm: 'direct',   hash: null },
    { label: 'fixed+counting',  fmm: 'fixed',    hash: null },
    { label: 'fixed+openaddr',  fmm: 'fixed',    hash: 'openaddr' },
    { label: 'fixed+funnel',    fmm: 'fixed',    hash: 'funnel' },
    // ?nfprobe=1 A/B: per-neighbor live hash probes instead of the
    // materialized dense arrays (the honest hash-backend comparison).
    { label: 'fixed+openaddr+nfprobe', fmm: 'fixed', hash: 'openaddr', url: 'nfprobe=1' },
    { label: 'fixed+funnel+nfprobe',   fmm: 'fixed', hash: 'funnel',   url: 'nfprobe=1' },
    { label: 'adaptive',        fmm: 'adaptive', hash: null },
    { label: 'adaptive+ahash0', fmm: 'adaptive', hash: null, url: 'adaptiveHash=0' },
    { label: 'adaptive+far0',   fmm: 'adaptive', hash: null, url: 'materializedFar=0' },
    // Round-14: accuracy-matched adaptive (full List-1 traversal) — the
    // validator's default-tier near-field error (~26% dv rel_l2 at 8k) vs
    // this config isolates what the sampling budget costs/saves in fps.
    { label: 'adaptive+fullNF', fmm: 'adaptive', hash: null, url: 'p2pbudget=4096' },
].filter((c) => !process.env.CONFIG || c.label === process.env.CONFIG);

// In-page sampler: rAF frame counter + gap outlier tracking + telemetry scrape
// + periodic canvas luminance signature. Runs entirely in the page; resolves
// with the per-second series. Passed as a real function so playwright calls
// it with MEASURE_MS (string expressions are evaluated, not invoked).
const SAMPLER = (ms) => new Promise((resolve) => {
    const el = (id) => document.getElementById(id);
    const num = (id) => { const v = parseFloat(el(id) ? el(id).innerText : ''); return Number.isFinite(v) ? v : null; };
    let frames = 0, last = 0, big34 = 0, big100 = 0, maxGap = 0;
    const tick = (t) => {
        frames++;
        if (last) { const g = t - last; if (g > 34) big34++; if (g > 100) big100++; if (g > maxGap) maxGap = g; }
        last = t;
        requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    const series = [];
    const start = Date.now();
    let sig = null;
    const sigCanvas = document.createElement('canvas');
    sigCanvas.width = 64; sigCanvas.height = 64;
    const sigCtx = sigCanvas.getContext('2d', { willReadFrequently: true });
    const captureSig = () => {
        try {
            sigCtx.drawImage(el('mainCanvas'), 0, 0, 64, 64);
            const d = sigCtx.getImageData(0, 0, 64, 64).data;
            let s = 0, sx = 0, sy = 0, sx2 = 0, sy2 = 0;
            for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
                const i = (y * 64 + x) * 4;
                const lum = (d[i] + d[i + 1] + d[i + 2]) / 765;
                s += lum; sx += x * lum; sy += y * lum; sx2 += x * x * lum; sy2 += y * y * lum;
            }
            if (s > 1e-6) {
                const cx = sx / s, cy = sy / s;
                sig = { cx: +cx.toFixed(3), cy: +cy.toFixed(3),
                        rx: +Math.sqrt(Math.max(0, sx2 / s - cx * cx)).toFixed(3),
                        ry: +Math.sqrt(Math.max(0, sy2 / s - cy * cy)).toFixed(3),
                        mass: +(s / (64 * 64)).toFixed(4) };
            } else sig = { mass: 0 };
        } catch (e) { sig = { err: String(e).slice(0, 80) }; }
    };
    captureSig();
    const iv = setInterval(() => {
        let diag = null;
        try { diag = JSON.parse(el('diagChannel') ? el('diagChannel').textContent : 'null'); } catch (e) {}
        series.push({
            t: +((Date.now() - start) / 1000).toFixed(1),
            fps: frames,
            gaps34: big34, gaps100: big100, maxGap: +maxGap.toFixed(0),
            valFPS: num('valFPS'),
            build: num('valFmmBuild'), m2l: num('valFmmM2l'), l2p: num('valFmmL2p'),
            main: num('valMainCompute'), render: num('valRenderPass'), gpu: num('valTotalGpu'),
            axis: el('valFmmAxis') ? el('valFmmAxis').innerText : '',
            errors: diag && diag.errors ? diag.errors.length : 0,
            initErrors: window.__initErrors ? window.__initErrors.length : 0,
            sig,
        });
        frames = 0; big34 = 0; big100 = 0; maxGap = 0;
        if (series.length % 10 === 5) captureSig();
        if (Date.now() - start >= ms) { clearInterval(iv); resolve(series); }
    }, 1000);
});

(async () => {
    const browser = await chromium.launch({
        headless: true,
        executablePath: EXE,
        args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-sandbox',
               '--use-gl=angle', '--use-angle=d3d11',
               '--disable-background-timer-throttling', '--disable-renderer-backgrounding'],
    });
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    const consoleErrors = [];
    page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
    page.on('pageerror', (err) => consoleErrors.push('PAGE: ' + err.message));
    // Round-14: capture the page's 1Hz `TELEM {json}` autoprint lines so the
    // per-config result carries the structured telemetry too (incl. the new
    // adaptiveMeta block: worker rebuild ms / cadence / nodes / depth).
    let telemLines = [];
    page.on('console', (m) => {
        if (m.type() === 'log' && m.text().startsWith('TELEM ')) telemLines.push(m.text().slice(6));
    });

    const results = [];
    for (const cfg of CONFIGS) {
        telemLines = [];
        const qs = [`preset=120k`, `scenario=galaxy`, `n=${N}`, `autoprint=1`];
        if (process.env.UNCAPPED === '1') qs.push('uncapped=1');
        if (cfg.url) qs.push(cfg.url);
        if (process.env.EXTRA_QS) qs.push(...process.env.EXTRA_QS.split('&').filter(Boolean));
        const url = `${BASE}?${qs.join('&')}`;
        process.stderr.write(`[${new Date().toISOString()}] ${cfg.label}: goto\n`);
        await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
        const badgeTimeout = 20000 + Math.round(N / 250000) * 10000;
        await page.waitForFunction(() => {
            const el = document.getElementById('engineBadge');
            return el && el.innerText && !el.innerText.includes('Detecting');
        }, { timeout: badgeTimeout });
        await page.evaluate((t) => {
            if (t.fmm) document.getElementById('selectFmmMode').value = t.fmm;
            if (t.hash) document.getElementById('selectHashMode').value = t.hash;
            document.getElementById('selectFmmMode').dispatchEvent(new Event('change'));
            document.getElementById('selectHashMode').dispatchEvent(new Event('change'));
        }, { fmm: cfg.fmm, hash: cfg.hash });
        await page.waitForTimeout(WARMUP_MS);
        process.stderr.write(`[${new Date().toISOString()}] ${cfg.label}: measuring ${MEASURE_SEC}s\n`);
        const series = await page.evaluate(SAMPLER, MEASURE_MS);
        const errors = [...new Set(consoleErrors.map(e => e.substring(0, 120)))];
        consoleErrors.length = 0;

        const fps = series.map(s => s.fps).filter(v => v > 0).sort((a, b) => a - b);
        const med = (a) => (a.length ? a[Math.floor(a.length / 2)] : 0);
        const p10 = (a) => (a.length ? a[Math.floor(a.length * 0.1)] : 0);
        const pos = series.map(s => s.fps).filter(v => v > 0);
        const first5 = pos.slice(0, 5), last5 = pos.slice(-5);
        const avg = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
        const passMed = (k) => med(series.map(s => s[k]).filter(v => v != null).sort((a, b) => a - b));
        const vfps = series.map(s => s.valFPS).filter(v => v && v > 0).sort((a, b) => a - b);
        const vpos = series.map(s => s.valFPS).filter(v => v && v > 0);
        const vFirst5 = vpos.slice(0, 5), vLast5 = vpos.slice(-5);
        results.push({
            config: cfg.label, N, measureSec: MEASURE_SEC,
            fpsMedian: med(fps), fpsP10: p10(fps), fpsMin: fps.length ? fps[0] : 0,
            fpsFirst5: +avg(first5).toFixed(1), fpsLast5: +avg(last5).toFixed(1),
            collapsePct: first5.length && avg(first5) > 0 ? +(((avg(first5) - avg(last5)) / avg(first5)) * 100).toFixed(1) : null,
            valFpsMedian: med(vfps),
            valFirst5: +avg(vFirst5).toFixed(1), valLast5: +avg(vLast5).toFixed(1),
            valCollapsePct: vFirst5.length && avg(vFirst5) > 0 ? +(((avg(vFirst5) - avg(vLast5)) / avg(vFirst5)) * 100).toFixed(1) : null,
            gaps34PerSec: +(series.reduce((a, s) => a + s.gaps34, 0) / Math.max(1, series.length)).toFixed(2),
            gaps100PerSec: +(series.reduce((a, s) => a + s.gaps100, 0) / Math.max(1, series.length)).toFixed(2),
            maxGapMs: Math.max(0, ...series.map(s => s.maxGap)),
            buildMs: passMed('build'), m2lMs: passMed('m2l'), l2pMs: passMed('l2p'),
            mainMs: passMed('main'), renderMs: passMed('render'), gpuMs: passMed('gpu'),
            diagErrorSecs: series.filter(s => s.errors > 0).length,
            axisLast: series.length ? series[series.length - 1].axis : '',
            sigSeries: series.filter((s, i) => i % 10 === 5).map(s => s.sig),
            fpsSeries: series.map(s => s.fps),
            valFpsSeries: series.map(s => s.valFPS),
            telem: telemLines.map((s) => { try { return JSON.parse(s); } catch (e) { return null; } }).filter(Boolean),
            consoleErrors: errors,
        });
        console.log(JSON.stringify(results[results.length - 1]));
        fs.writeFileSync('tools/bench_fps_longrun_results.json', JSON.stringify(results, null, 2));
    }
    await browser.close();

    console.log('\n| config | steps/s med | first5→last5 | collapse% | rAF fps med | gaps>34ms/s | gaps>100ms/s | maxGap ms | build/m2l/l2p/main/render ms |');
    console.log('|---|---|---|---|---|---|---|---|---|');
    for (const r of results) {
        console.log(`| ${r.config} | ${r.valFpsMedian} | ${r.valFirst5}→${r.valLast5} | ${r.valCollapsePct} | ${r.fpsMedian} | ${r.gaps34PerSec} | ${r.gaps100PerSec} | ${r.maxGapMs} | ${r.buildMs}/${r.m2lMs}/${r.l2pMs}/${r.mainMs}/${r.renderMs} |`);
    }
})().catch(err => { console.error('FAILED:', err); process.exit(1); });
