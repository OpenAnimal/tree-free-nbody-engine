// Cross-benchmark of the demo's FMM axes at a fixed particle count.
//
// Drives the shipped index.html in headless Chromium (WebGPU) via playwright
// and samples the sidebar telemetry for each configuration:
//   far-field: fixed-grid FMM vs adaptive quadtree FMM
//   near-field hash: counting-sort / open-addressing / funnel   (fixed mode only:
//     in adaptive galaxy mode the near field comes from the quadtree List-1
//     ranges, so the shared cell-list axis is inactive by design)
//   adaptive node-hash directory: on/off via ?adaptiveHash       (adaptive only)
//
// Configs are measured in interleaved rounds and per-config MEDIANS are
// reported: single sequential runs proved noise-dominated on this machine
// (background GPU consumers, thermal drift). Frames run uncapped
// (MessageChannel scheduler), so valFPS is steps/sec.
//
// Usage:  node tools/browser_crossbench.js [N] [rounds]   (default 500000 3)
// Requires the repo served on http://localhost:8123 (python -m http.server 8123).
// Prints one JSON line per config, a markdown table, and adaptive screenshots.

const { chromium } = require('playwright');
const fs = require('fs');

const N = parseInt(process.argv[2] || '500000', 10);
const ROUNDS = parseInt(process.argv[3] || '3', 10);
const PORT = process.env.PORT || '8123';
const WARMUP_MS = 5000;
const MEASURE_MS = 5000;
const BASE = `http://localhost:${PORT}/index.html`;
// Full Chromium (NOT the headless-shell: it has no WebGPU adapter and the
// run silently falls back to WebGL2). Override via CROSSBENCH_EXE; if the
// pinned path is absent (other machines), fall back to whatever full
// chromium playwright has installed.
const EXE_CANDIDATES = [
    process.env.CROSSBENCH_EXE,
    'C:\\Users\\Y\\AppData\\Local\\ms-playwright\\chromium-1234\\chrome-win64\\chrome.exe',
    ...fs.readdirSync(process.env.LOCALAPPDATA + '/ms-playwright', { withFileTypes: true })
        .filter((d) => d.isDirectory() && d.name.startsWith('chromium-') && !d.name.includes('headless'))
        .map((d) => `${process.env.LOCALAPPDATA}/ms-playwright/${d.name}/chrome-win64/chrome.exe`),
].filter(Boolean);
const EXE = EXE_CANDIDATES.find((p) => fs.existsSync(p)) || undefined;

const CONFIGS = [
    { label: 'fixed+p0+counting', url: '', toggle: { order: '0' } },
    { label: 'fixed+p1+counting', url: '', toggle: { order: '1' } },
    { label: 'fixed+p2+counting', url: '', toggle: { order: '2' } },
    { label: 'fixed+p4+counting', url: '', toggle: { order: '4' } },
    { label: 'fixed+openaddr', url: '', toggle: { hashMode: 'openaddr', order: '2' } },
    { label: 'fixed+funnel', url: '', toggle: { hashMode: 'funnel', order: '2' } },
    { label: 'fixed+counting', url: '', toggle: { order: '2' } },
    { label: 'adaptive+ahash1', url: '',                toggle: { fmmMode: 'adaptive' } },
    { label: 'adaptive+ahash0', url: '?adaptiveHash=0', toggle: { fmmMode: 'adaptive' } },
    // Round 13 A/B: the materialized far-field CSR gather (default ON since
    // round 13) vs the legacy per-level m2l+l2l chain (?materializedFar=0).
    { label: 'adaptive+far0',   url: '?materializedFar=0', toggle: { fmmMode: 'adaptive' } },
// CONFIG=<label> runs a single config in a fresh process — at 5M the fifth
// in-process navigation stalls (cumulative GPU memory across contexts), so
// extreme-N runs isolate each config in its own browser via CONFIG.
].filter((c) => !process.env.CONFIG || c.label === process.env.CONFIG);

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

    const measureOne = async (cfg) => {
        // ?uncapped=1: benchmark numbers require the MessageChannel scheduler
        // (the page now DEFAULTS to the vsync-locked loop at 60 fps).
        const url = `${BASE}?preset=500k&scenario=galaxy&n=${N}&uncapped=1&seed=42${cfg.url ? '&' + cfg.url.slice(1) : ''}`;
        await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
        // Init at 5M (particle generation + upload + first compile) far
        // exceeds the default 20 s badge wait — scale it with N.
        const badgeTimeout = 20000 + Math.round(N / 250000) * 10000;
        await page.waitForFunction(() => {
            const el = document.getElementById('engineBadge');
            return el && el.innerText && !el.innerText.includes('Detecting');
        }, { timeout: badgeTimeout });
        if (Object.keys(cfg.toggle).length) {
            await page.evaluate((t) => {
                if (t.fmmMode) document.getElementById('selectFmmMode').value = t.fmmMode;
                if (t.order) document.getElementById('selectFmmOrder').value = t.order;
                if (t.hashMode) document.getElementById('selectHashMode').value = t.hashMode;
                document.getElementById('selectFmmMode').dispatchEvent(new Event('change'));
                document.getElementById('selectFmmOrder').dispatchEvent(new Event('change'));
                document.getElementById('selectHashMode').dispatchEvent(new Event('change'));
            }, cfg.toggle);
        }
        await page.waitForTimeout(WARMUP_MS);
        return await page.evaluate((ms) => {
            return new Promise((resolve) => {
                const samples = [];
                const start = Date.now();
                const iv = setInterval(() => {
                    const num = (id) => parseFloat(document.getElementById(id)?.innerText || '') || 0;
                    samples.push({
                        fps: num('valFPS'),
                        pipeline: num('valPipelineRate'),
                        gpuWork: num('valGpuWorkRate'),
                        gpuComplete: num('valGpuComplete'),
                        totalGpu: num('valTotalGpu'),
                        build: num('valFmmBuild'),
                        main: num('valMainCompute'),
                        axis: document.getElementById('valFmmAxis')?.innerText || '',
                        diag: document.getElementById('diagChannel')?.textContent || '',
                    });
                    if (Date.now() - start >= ms) { clearInterval(iv); resolve(samples); }
                }, 500);
            });
        }, MEASURE_MS);
    };

    const acc = new Map(CONFIGS.map(c => [c.label, { rounds: [], axes: new Set(), diagErrs: 0 }]));
    for (let round = 0; round < ROUNDS; round++) {
        for (const cfg of CONFIGS) {
            const tel = await measureOne(cfg);
            const valid = tel.filter(s => s.fps > 0);
            const med = (key) => {
                const v = valid.map(s => s[key]).filter(x => x > 0).sort((a, b) => a - b);
                return v.length ? v[Math.floor(v.length / 2)] : 0;
            };
            const a = acc.get(cfg.label);
            a.rounds.push({ steps: med('fps'), pipeline: med('pipeline'), gpuWork: med('gpuWork'), gpuComplete: med('gpuComplete'), totalGpu: med('totalGpu'), build: med('build'), main: med('main') });
            a.axes.add(tel[tel.length - 1].axis);
            try { if ((JSON.parse(tel[tel.length - 1].diag).errors || []).length) a.diagErrs++; } catch (e) {}
        }
        console.error(`round ${round + 1}/${ROUNDS} done`);
    }
    const errors = [...new Set(consoleErrors.map(e => e.substring(0, 100)))];
    consoleErrors.length = 0;

    const results = [];
    for (const cfg of CONFIGS) {
        const a = acc.get(cfg.label);
        const mid = a.rounds.slice().sort((x, y) => x.steps - y.steps)[Math.floor(a.rounds.length / 2)];
        results.push({ config: cfg.label, N, medianStepsPerSec: mid.steps,
            medianPipelineStepsPerSec: mid.pipeline,
            medianGpuWorkPartsPerSec: mid.gpuWork,
            medianGpuCompleteMs: mid.gpuComplete,
            medianTotalGpuMs: mid.totalGpu,
            medianBuildMs: mid.build, medianMainMs: mid.main,
            rounds: a.rounds.map(x => x.steps), axis: [...a.axes][0], diagErrRounds: a.diagErrs });
    }
    for (const r of results) console.log(JSON.stringify(r));
    if (errors.length) console.log(JSON.stringify({ consoleErrors: errors }));

    // adaptive screenshot for the artifact record (skipped in single-config runs)
    if (CONFIGS.length > 3) {
        await measureOne(CONFIGS[3]);
        await page.screenshot({ path: `crossbench_adaptive_${N}.png` }).catch(() => {});
    }
    await browser.close();

    console.log('\n| config | steps/sec | GPU work M parts/sec | GPU complete ms | total GPU ms | build ms | main ms | rounds |');
    console.log('|---|---:|---:|---:|---:|---:|---:|---|');
    for (const r of results) console.log(`| ${r.config} | ${r.medianStepsPerSec} | ${r.medianGpuWorkPartsPerSec} | ${r.medianGpuCompleteMs} | ${r.medianTotalGpuMs} | ${r.medianBuildMs} | ${r.medianMainMs} | ${r.rounds.join(', ')} |`);
})().catch(err => { console.error('FAILED:', err); process.exit(1); });
