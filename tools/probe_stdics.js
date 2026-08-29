// One-off probe: load the demo with ?ic=plummer|collapse and read the
// self-gravity diagnostics rows (delta-E/E + Lagrangian radii) after a few
// diagnostic intervals. Local artifact, not part of the pytest suite.
//   node tools/probe_stdics.js [ic] [seconds]
const { chromium } = require('playwright');
const fs = require('fs');
const EXE = [
    process.env.CROSSBENCH_EXE,
    'C:/Users/Y/AppData/local/ms-playwright/chromium-1234/chrome-win64/chrome.exe',
    ...fs.readdirSync(process.env.LOCALAPPDATA + '/ms-playwright', { withFileTypes: true })
        .filter((d) => d.isDirectory() && d.name.startsWith('chromium-') && !d.name.includes('headless'))
        .map((d) => `${process.env.LOCALAPPDATA}/ms-playwright/${d.name}/chrome-win64/chrome.exe`),
].filter(Boolean).filter((p) => fs.existsSync(p));

const IC = process.argv[2] || 'plummer';
const SEC = parseInt(process.argv[3] || '14', 10) * 1000;
const N = 120000;

(async () => {
    const browser = await chromium.launch({
        headless: true, executablePath: EXE[0],
        args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-sandbox',
               '--use-gl=angle', '--use-angle=d3d11',
               '--disable-background-timer-throttling', '--disable-renderer-backgrounding'],
    });
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    const errs = [];
    page.on('console', (c) => { if (c.type() === 'error') errs.push(c.text().slice(0, 200)); });
    page.on('pageerror', (e) => errs.push('PAGE: ' + e.message.slice(0, 200)));
    const url = `http://localhost:8124/index.html?ic=${IC}&n=${N}&uncapped=1&autoprint=1`;
    process.stderr.write(`goto ${url}\n`);
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await new Promise((r) => setTimeout(r, SEC));
    const out = await page.evaluate(() => ({
        icLabel: document.getElementById('valGalaxyIC')?.innerText,
        gp: document.getElementById('valGravityGp')?.innerText,
        energy: document.getElementById('valSelfGravEnergy')?.innerText,
        radii: document.getElementById('valLagrangianRadii')?.innerText,
        rowE: document.getElementById('rowSelfGravEnergy')?.style.display,
        rowL: document.getElementById('rowLagrangianRadii')?.style.display,
        impl: document.getElementById('implDetailsLabel')?.innerText,
        fps: document.getElementById('valFPS')?.innerText,
        title: document.getElementById('overlay-info')?.innerText?.slice(0, 120),
    }));
    console.log(JSON.stringify({ ic: IC, n: N, ...out, consoleErrors: errs.slice(0, 6) }, null, 2));
    await browser.close();
    process.exit(errs.length ? 2 : 0);
})();
