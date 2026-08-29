// Headless smoke test for validate.html (the far-field cross-validation rig).
//
// Serves the repo root with its OWN python http.server on PORT 8124 (NOT
// 8123 — the main agent's harness needs that port), loads validate.html in
// full Chromium with WebGPU enabled, waits for window.__validatorResult,
// prints the result table + JSON, and exits non-zero on any failure:
//   - result missing after the timeout
//   - any non-finite metric
//   - direct self-noise dvRelL2 > 0.01 (GPU determinism control)
//   - pass.allOk === false (includes captured WebGPU validation errors)
//
// Usage: node tools/smoke_validator.js [n] [steps] [extraQuery]
//   e.g. node tools/smoke_validator.js 8000 120 "p2pbudget=4096"
//   env: PORT (default 8124), KEEP_SERVER=1 to skip self-serve (already running)

const { chromium } = require('playwright');
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const N = parseInt(process.argv[2] || '8000', 10);
const STEPS = parseInt(process.argv[3] || '120', 10);
const PORT = process.env.PORT || '8124';
const BASE = `http://localhost:${PORT}`;
const RESULT_TIMEOUT_MS = 180000;

// Full Chromium (not headless-shell — no WebGPU adapter there). Same
// discovery chain as tools/bench_fps_longrun.js.
const EXE_CANDIDATES = [
    process.env.CROSSBENCH_EXE,
    'C:\\Users\\Y\\AppData\\Local\\ms-playwright\\chromium-1234\\chrome-win64\\chrome.exe',
    ...fs.readdirSync(process.env.LOCALAPPDATA + '/ms-playwright', { withFileTypes: true })
        .filter((d) => d.isDirectory() && d.name.startsWith('chromium-') && !d.name.includes('headless'))
        .map((d) => `${process.env.LOCALAPPDATA}/ms-playwright/${d.name}/chrome-win64/chrome.exe`),
].filter(Boolean);
const EXE = EXE_CANDIDATES.find((p) => fs.existsSync(p)) || undefined;

const repoRoot = path.resolve(__dirname, '..');

function waitForServer(url, timeoutMs) {
    const t0 = Date.now();
    return new Promise((resolve, reject) => {
        const attempt = () => {
            const req = http.get(url, (res) => {
                res.resume();
                resolve();
            });
            req.on('error', () => {
                if (Date.now() - t0 > timeoutMs) reject(new Error(`server not reachable at ${url}`));
                else setTimeout(attempt, 300);
            });
        };
        attempt();
    });
}

(async () => {
    let serverProc = null;
    if (process.env.KEEP_SERVER !== '1') {
        process.stderr.write(`[${new Date().toISOString()}] serving repo root on port ${PORT}\n`);
        serverProc = spawn('python', ['-m', 'http.server', PORT], { cwd: repoRoot, stdio: 'ignore' });
    }
    try {
        await waitForServer(BASE + '/validate.html', 15000);

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

        const extraArg = process.argv.slice(4).find((a) => String(a).includes('='));
        const extraQuery = extraArg ? '&' + String(extraArg).replace(/^[?&]/, '') : '';
        const url = `${BASE}/validate.html?n=${N}&steps=${STEPS}${extraQuery}`;
        process.stderr.write(`[${new Date().toISOString()}] goto ${url}\n`);
        await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });

        const result = await page.waitForFunction(
            () => window.__validatorResult || null,
            { timeout: RESULT_TIMEOUT_MS }
        ).then((h) => h.jsonValue()).catch(() => null);

        await browser.close();

        if (!result) {
            console.error('FAILED: window.__validatorResult never appeared ' +
                `(timeout ${RESULT_TIMEOUT_MS} ms). Console errors:\n  ` +
                (consoleErrors.slice(0, 20).join('\n  ') || '(none)'));
            process.exit(1);
        }

        // ---- print the table ----
        const fmt = (v, d = 5) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : String(v));
        console.log(`\nmeta: n=${result.meta.n} steps=${result.meta.steps} dt=${result.meta.dt} ` +
            `leafBits=${result.meta.leafBits} order=${result.meta.order} gpu="${result.meta.gpu}"` +
            (result.meta.ic && result.meta.ic !== 'flyby' ? ` ic=${result.meta.ic}` : ''));
        console.log(`direct self-noise dvRelL2 = ${fmt(result.meta.noiseFloorDvRelL2)} ` +
            `(floor scale applied to dv thresholds: ${fmt(result.meta.noiseFloorScale)})`);
        // Standard-scenario columns (dE/E of the exact softened-log
        // Hamiltonian + the r50 Lagrangian-radii ratio; GPU_NOTES 15) only
        // exist when the rig ran a standard IC (?ic=plummer|collapse).
        const std = !!(result.meta.ic && result.meta.ic !== 'flyby');
        const fmtE = (r) => (typeof r.dEoverE === 'number' && Number.isFinite(r.dEoverE)
            ? `${r.dEoverE >= 0 ? '+' : ''}${(r.dEoverE * 100).toFixed(3)}%` : '—');
        const fmtR = (r) => (typeof r.r50Ratio === 'number' && Number.isFinite(r.r50Ratio)
            ? `x${r.r50Ratio.toFixed(3)}` : '—');
        console.log('\n| mode | vs | dv rel_l2 | dv cos | dv max_abs | pos rel_l2 | pos rms | pos cos |'
            + (std ? ' dE/E | r50(t)/r50(0) |' : '') + ' ms/step | status |');
        console.log('|---|---|---|---|---|---|---|---|' + (std ? '---|---|' : '') + '---|---|');
        for (const r of result.rows) {
            console.log(`| ${r.mode} | ${r.vs} | ${fmt(r.dvRelL2)} | ${fmt(r.dvCos)} | ${fmt(r.dvMaxAbs, 6)} ` +
                `| ${fmt(r.posRelL2)} | ${fmt(r.posRms)} | ${fmt(r.posCos)} ` +
                (std ? `| ${fmtE(r)} | ${fmtR(r)} ` : '') +
                `| ${fmt(r.msPerStep, 2)} | ${r.status} |`);
        }

        // ---- JSON record ----
        console.log('\nVALIDATE ' + JSON.stringify(result));

        // ---- exit criteria ----
        const problems = [];
        if (!result.pass || result.pass.allOk !== true) {
            for (const f of (result.pass && result.pass.failures) || ['pass.allOk !== true']) {
                problems.push(String(f));
            }
        }
        for (const r of result.rows) {
            for (const [k, v] of Object.entries(r)) {
                if (typeof v === 'number' && !Number.isFinite(v)) problems.push(`${r.mode}.${k} is non-finite`);
            }
        }
        const noiseRow = result.rows.find((r) => r.mode === 'direct#2');
        if (noiseRow && !(noiseRow.dvRelL2 <= 0.01)) {
            problems.push(`direct self-noise dvRelL2 ${noiseRow.dvRelL2} > 0.01`);
        }
        if (consoleErrors.length) {
            console.error(`\nconsole errors (${consoleErrors.length}):\n  ` +
                [...new Set(consoleErrors)].slice(0, 20).join('\n  '));
        }
        if (problems.length) {
            console.error('\nSMOKE FAILED:\n  ' + problems.join('\n  '));
            process.exit(1);
        }
        console.log('\nSMOKE PASSED');
        process.exit(0);
    } catch (err) {
        console.error('FAILED:', err && err.message ? err.message : err);
        process.exit(1);
    } finally {
        if (serverProc) serverProc.kill();
    }
})();
