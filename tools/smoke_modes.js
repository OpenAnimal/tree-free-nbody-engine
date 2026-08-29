// Mode smoke test: switch through every far-field/hash mode, capture console
// errors, fps, axis string, and pass telemetry. Verifies each mode steps and
// submits cleanly after the round-14 edits.
const { chromium } = require('playwright');
const fs = require('fs');
const EXE = [
    'C:/Users/Y/AppData/Local/ms-playwright/chromium-1234/chrome-win64/chrome.exe',
    ...fs.readdirSync(process.env.LOCALAPPDATA + '/ms-playwright', { withFileTypes: true })
        .filter((d) => d.isDirectory() && d.name.startsWith('chromium-') && !d.name.includes('headless'))
        .map((d) => `${process.env.LOCALAPPDATA}/ms-playwright/${d.name}/chrome-win64/chrome.exe`),
].filter((p) => fs.existsSync(p));

const N = process.argv[2] || '120000';
const SEC = parseInt(process.argv[3] || '6', 10) * 1000;

const MODES = [
    { name: 'off', fmm: 'off' },
    { name: 'direct', fmm: 'direct' },
    { name: 'fixed', fmm: 'fixed' },
    { name: 'fixed+funnel', fmm: 'fixed', hash: 'funnel' },
    { name: 'fixed+openaddr', fmm: 'fixed', hash: 'openaddr' },
    { name: 'adaptive', fmm: 'adaptive' },
];

(async () => {
    const browser = await chromium.launch({
        headless: true, executablePath: EXE[0],
        args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-sandbox',
               '--use-gl=angle', '--use-angle=d3d11',
               '--disable-background-timer-throttling', '--disable-renderer-backgrounding'],
    });
    for (const extra of ['', 'nfprobe=1']) {
        for (const m of MODES) {
            if (extra && m.hash !== 'funnel' && m.hash !== 'openaddr') continue; // nfprobe only affects hash backends
            const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
            const errs = [];
            page.on('console', (c) => { if (c.type() === 'error' || c.type() === 'warning') errs.push(`[${c.type()}] ${c.text().slice(0, 140)}`); });
            page.on('pageerror', (e) => errs.push('PAGE: ' + e.message.slice(0, 140)));
            const qs = [`preset=120k`, `scenario=galaxy`, `n=${N}`];
            if (extra) qs.push(extra);
            await page.goto(`http://localhost:8123/index.html?${qs.join('&')}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
            await page.waitForFunction(() => {
                const el = document.getElementById('engineBadge');
                return el && el.innerText && !el.innerText.includes('Detecting');
            }, { timeout: 30000 });
            await page.evaluate((t) => {
                if (t.fmm) document.getElementById('selectFmmMode').value = t.fmm;
                if (t.hash) document.getElementById('selectHashMode').value = t.hash;
                document.getElementById('selectFmmMode').dispatchEvent(new Event('change'));
                document.getElementById('selectHashMode').dispatchEvent(new Event('change'));
            }, { fmm: m.fmm, hash: m.hash || null });
            await page.waitForTimeout(SEC);
            const r = await page.evaluate(() => {
                const num = (id) => { const v = parseFloat(document.getElementById(id) ? document.getElementById(id).innerText : ''); return Number.isFinite(v) ? v : null; };
                return {
                    badge: document.getElementById('engineBadge').innerText,
                    modeLabel: document.getElementById('valFmmMode').innerText,
                    axis: document.getElementById('valFmmAxis').innerText,
                    fps: num('valFPS'), stepMs: num('valComputeTime'),
                    build: num('valFmmBuild'), m2l: num('valFmmM2l'), l2p: num('valFmmL2p'),
                    main: num('valMainCompute'), render: num('valRenderPass'), gpu: num('valTotalGpu'),
                };
            });
            const uniq = [...new Set(errs)];
            console.log(`[${extra || 'default'}] ${m.name}: fps=${r.fps} step=${r.stepMs}ms build=${r.build} m2l=${r.m2l} l2p=${r.l2p} main=${r.main} gpu=${r.gpu}`);
            console.log(`   badge='${r.badge}' mode='${r.modeLabel}' axis='${r.axis}'`);
            if (uniq.length) console.log(`   ERRORS: ${uniq.slice(0, 5).join(' | ')}`);
            await page.close();
        }
    }
    await browser.close();
})().catch((e) => { console.error('FAILED:', e); process.exit(1); });
