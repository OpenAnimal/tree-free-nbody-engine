// Adaptive quadtree SHAPE sweep: leaf-occupancy target x max depth at a
// fixed particle count, in the real browser (headless Chromium + WebGPU).
//
// Motivation (Lashuk et al., 2012): larger leaf capacity => shallower tree
// => less memory-bound per-level chain walking, more compute-bound
// near-field work. The auto tier (computeAdaptiveLeafTarget) picks 16 at
// 500k; this sweep measures whether 32/64/128 with matching or deeper
// maxDepth buys throughput. Requires the ?leaftarget= / ?adapdepth= URL
// overrides in index.html.
//
// Usage: node tools/adaptive_shape_sweep.js [N] [rounds]   (default 500000 3)
// Requires the repo on http://localhost:8123 (python -m http.server 8123).
// Prints one JSON line per config + a markdown table; interleaved rounds
// with per-config medians, same protocol as browser_crossbench.js.

const { chromium } = require('playwright');
const fs = require('fs');

const N = parseInt(process.argv[2] || '500000', 10);
const ROUNDS = parseInt(process.argv[3] || '3', 10);
const PORT = process.env.PORT || '8123';
const WARMUP_MS = 4000;
const MEASURE_MS = 5000;
const BASE = `http://localhost:${PORT}/index.html`;
const EXE_CANDIDATES = [
    process.env.CROSSBENCH_EXE,
    'C:\\Users\\Y\\AppData\\Local\\ms-playwright\\chromium-1234\\chrome-win64\\chrome.exe',
    ...fs.readdirSync(process.env.LOCALAPPDATA + '/ms-playwright', { withFileTypes: true })
        .filter((d) => d.isDirectory() && d.name.startsWith('chromium-') && !d.name.includes('headless'))
        .map((d) => `${process.env.LOCALAPPDATA}/ms-playwright/${d.name}/chrome-win64/chrome.exe`),
].filter(Boolean);
const EXE = EXE_CANDIDATES.find((p) => fs.existsSync(p)) || undefined;

// (leaftarget, adapdepth) pairs; 0 = auto for either.
const CONFIGS = [
    { label: 'auto(16/d8)', lt: 0,   d: 0 },
    { label: 'lt32',        lt: 32,  d: 0 },
    { label: 'lt64',        lt: 64,  d: 0 },
    { label: 'lt128',       lt: 128, d: 0 },
    { label: 'lt64-d8',     lt: 64,  d: 8 },
    { label: 'lt128-d9',    lt: 128, d: 9 },
];

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
    let lastTreeInfo = '';
    page.on('console', (m) => {
        const t = m.text();
        if (m.type() === 'error') consoleErrors.push(t);
        if (t.includes('Adaptive FMM enabled') || t.includes('quadtree')) lastTreeInfo = t;
    });
    page.on('pageerror', (err) => consoleErrors.push('PAGE: ' + err.message));

    const measureOne = async (cfg) => {
        const url = `${BASE}?preset=500k&scenario=galaxy&n=${N}&uncapped=1` +
            (cfg.lt ? `&leaftarget=${cfg.lt}` : '') + (cfg.d ? `&adapdepth=${cfg.d}` : '');
        await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
        const badgeTimeout = 20000 + Math.round(N / 250000) * 10000;
        await page.waitForFunction(() => {
            const el = document.getElementById('engineBadge');
            return el && el.innerText && !el.innerText.includes('Detecting');
        }, { timeout: badgeTimeout });
        await page.evaluate(() => {
            const sel = document.getElementById('selectFmmMode');
            sel.value = 'adaptive';
            sel.dispatchEvent(new Event('change', { bubbles: true }));
        });
        await page.waitForTimeout(WARMUP_MS);
        return await page.evaluate((ms) => {
            return new Promise((resolve) => {
                const samples = [];
                const start = Date.now();
                const iv = setInterval(() => {
                    samples.push({
                        fps: parseFloat(document.getElementById('valFPS').innerText) || 0,
                        axis: document.getElementById('valFmmAxis')?.innerText || '',
                        diag: document.getElementById('diagChannel')?.textContent || '',
                        info: (typeof lastAdaptiveInfo !== 'undefined') ? '' : '',
                    });
                    if (Date.now() - start >= ms) { clearInterval(iv); resolve(samples); }
                }, 500);
            });
        }, MEASURE_MS);
    };

    const acc = new Map(CONFIGS.map(c => [c.label, { rounds: [], tree: '' }]));
    for (let round = 0; round < ROUNDS; round++) {
        for (const cfg of CONFIGS) {
            lastTreeInfo = '';
            const tel = await measureOne(cfg);
            const v = tel.filter(s => s.fps > 0).map(s => s.fps).sort((a, b) => a - b);
            const median = v.length ? v[Math.floor(v.length / 2)] : 0;
            const a = acc.get(cfg.label);
            a.rounds.push(median);
            if (!a.tree && lastTreeInfo) a.tree = lastTreeInfo;
        }
        console.error(`round ${round + 1}/${ROUNDS} done`);
    }
    const errors = [...new Set(consoleErrors.map(e => e.substring(0, 100)))];

    const results = [];
    for (const cfg of CONFIGS) {
        const a = acc.get(cfg.label);
        const med = a.rounds.slice().sort((x, y) => x - y)[Math.floor(a.rounds.length / 2)];
        results.push({ config: cfg.label, N, medianStepsPerSec: med, rounds: a.rounds, tree: a.tree });
    }
    for (const r of results) console.log(JSON.stringify(r));
    if (errors.length) console.log(JSON.stringify({ consoleErrors: errors }));

    // leave the last config running and screenshot for artifact inspection
    await measureOne(CONFIGS[0]);
    await page.waitForTimeout(20000); // let structure develop
    await page.screenshot({ path: `_sweep_adaptive_${N}.png` }).catch(() => {});
    await browser.close();

    console.log('\n| config | median steps/sec | rounds | tree |');
    console.log('|---|---|---|---|');
    for (const r of results) console.log(`| ${r.config} | ${r.medianStepsPerSec} | ${r.rounds.join(', ')} | ${r.tree.replace(/\|/g, '/')} |`);
})().catch(err => { console.error('FAILED:', err); process.exit(1); });
