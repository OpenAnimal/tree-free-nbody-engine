// Compare adaptive-mode force fields with the funnel directory on vs off,
// early in the run (tree still dense). Reads webgpuFmmField + positions and
// reports per-particle force magnitude stats vs distance from the two cores.
const { chromium } = require('playwright');
const fs = require('fs');
const EXE = [
    'C:/Users/Y/AppData/Local/ms-playwright/chromium-1234/chrome-win64/chrome.exe',
    ...fs.readdirSync(process.env.LOCALAPPDATA + '/ms-playwright', { withFileTypes: true })
        .filter((d) => d.isDirectory() && d.name.startsWith('chromium-') && !d.name.includes('headless'))
        .map((d) => `${process.env.LOCALAPPDATA}/ms-playwright/${d.name}/chrome-win64/chrome.exe`),
].filter((p) => fs.existsSync(p));

const N = process.argv[2] || '500000';
const RUN_SEC = parseFloat(process.argv[3] || '3');

async function runOnce(browser, urlExtra) {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    await page.goto(`http://localhost:8123/index.html?preset=120k&scenario=galaxy&n=${N}${urlExtra}`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
        const el = document.getElementById('engineBadge');
        return el && el.innerText && !el.innerText.includes('Detecting');
    }, { timeout: 30000 });
    await page.evaluate(() => {
        document.getElementById('selectFmmMode').value = 'adaptive';
        document.getElementById('selectFmmMode').dispatchEvent(new Event('change'));
    });
    await page.waitForTimeout(RUN_SEC * 1000);
    const out = await page.evaluate(async () => {
        const need = Math.ceil((N * 16) / 256) * 256;
        const posBuf = webgpuDevice.createBuffer({ size: need, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const fBuf = webgpuDevice.createBuffer({ size: need, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const src = webgpuPingPong === 0 ? webgpuPosVelBufferA : webgpuPosVelBufferB;
        const enc = webgpuDevice.createCommandEncoder();
        enc.copyBufferToBuffer(src, 0, posBuf, 0, N * 16);
        enc.copyBufferToBuffer(webgpuFmmField, 0, fBuf, 0, N * 16);
        webgpuDevice.queue.submit([enc.finish()]);
        await posBuf.mapAsync(GPUMapMode.READ);
        const f = new Float32Array(posBuf.getMappedRange());
        await fBuf.mapAsync(GPUMapMode.READ);
        const g = new Float32Array(fBuf.getMappedRange());
        const stride = Math.max(1, N >> 8);
        let n = 0, magSum = 0, magMax = 0, potSum = 0, nanCount = 0;
        const hist = [0, 0, 0, 0, 0]; // [0,1e-3) [1e-3,1e-2) [1e-2,0.1) [0.1,1) [1+)
        const md = adaptiveMetadata;
        for (let i = 0; i < N; i += stride) {
            const fx = g[i * 4], fy = g[i * 4 + 1], pot = g[i * 4 + 2];
            const mag = Math.hypot(fx, fy);
            if (!Number.isFinite(mag) || !Number.isFinite(pot)) { nanCount++; continue; }
            n++; magSum += mag; if (mag > magMax) magMax = mag; potSum += pot;
            const v = mag; hist[v < 1e-3 ? 0 : v < 1e-2 ? 1 : v < 0.1 ? 2 : v < 1 ? 3 : 4]++;
        }
        posBuf.destroy(); fBuf.destroy();
        return {
            nodes: md.totalNodes, sampled: n, nan: nanCount,
            magMean: +(magSum / Math.max(1, n)).toExponential(3), magMax: +magMax.toExponential(3),
            potMean: +(potSum / Math.max(1, n)).toExponential(3),
            hist,
        };
    });
    await page.close();
    return out;
}

(async () => {
    const browser = await chromium.launch({
        headless: true, executablePath: EXE[0],
        args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-sandbox',
               '--use-gl=angle', '--use-angle=d3d11', '--disable-background-timer-throttling', '--disable-renderer-backgrounding'],
    });
    for (let k = 0; k < 2; k++) {
        const on = await runOnce(browser, '');
        const off = await runOnce(browser, '&adaptiveHash=0');
        console.log(`round${k} dir=ON :`, JSON.stringify(on));
        console.log(`round${k} dir=OFF:`, JSON.stringify(off));
    }
    await browser.close();
})().catch((e) => { console.error('FAILED:', e); process.exit(1); });
