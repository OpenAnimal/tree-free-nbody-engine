// One-off probe: what backend is actually active in headless, do the pass
// timers populate, and does the sim render visibly (screenshot + luminance)?
const { chromium } = require('playwright');
const fs = require('fs');

const EXE = [
    process.env.CROSSBENCH_EXE,
    'C:\\Users\\Y\\AppData\\Local\\ms-playwright\\chromium-1234\\chrome-win64\\chrome.exe',
    ...fs.readdirSync(process.env.LOCALAPPDATA + '/ms-playwright', { withFileTypes: true })
        .filter((d) => d.isDirectory() && d.name.startsWith('chromium-') && !d.name.includes('headless'))
        .map((d) => `${process.env.LOCALAPPDATA}/ms-playwright/${d.name}/chrome-win64/chrome.exe`),
].filter(Boolean).find((p) => fs.existsSync(p));

(async () => {
    const browser = await chromium.launch({
        headless: true, executablePath: EXE,
        args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-sandbox',
               '--use-gl=angle', '--use-angle=d3d11',
               '--disable-background-timer-throttling', '--disable-renderer-backgrounding'],
    });
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    page.on('console', (m) => { if (m.type() === 'error') console.log('CONSOLE-ERR:', m.text().slice(0, 200)); });
    await page.goto('http://localhost:8123/index.html?preset=120k&scenario=galaxy&n=120000&uncapped=1', { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
        const el = document.getElementById('engineBadge');
        return el && el.innerText && !el.innerText.includes('Detecting');
    }, { timeout: 40000 });
    await page.evaluate(() => {
        const sel = document.getElementById('selectFmmMode');
        sel.value = 'adaptive';
        sel.dispatchEvent(new Event('change'));
    });
    await page.waitForTimeout(6000);
    const info = await page.evaluate(() => {
        const t = (id) => { const e = document.getElementById(id); return e ? e.innerText : null; };
        const c = document.getElementById('mainCanvas');
        const cv = document.createElement('canvas'); cv.width = 64; cv.height = 64;
        const cx = cv.getContext('2d', { willReadFrequently: true });
        let lumSum = -1;
        try {
            cx.drawImage(c, 0, 0, 64, 64);
            const d = cx.getImageData(0, 0, 64, 64).data;
            lumSum = 0;
            for (let i = 0; i < d.length; i += 4) lumSum += (d[i] + d[i + 1] + d[i + 2]) / 765;
        } catch (e) { lumSum = 'ERR ' + e.message; }
        return {
            engine: t('engineBadge'), gpu: t('gpuName'), vram: t('vramMode'),
            fps: t('valFPS'), build: t('valFmmBuild'), m2l: t('valFmmM2l'), l2p: t('valFmmL2p'),
            main: t('valMainCompute'), render: t('valRenderPass'), total: t('valTotalGpu'),
            axis: t('valFmmAxis'), impl: t('implDetails') || t('valImpl'),
            canvasW: c.width, canvasH: c.height, drawImageLumSum: lumSum,
        };
    });
    console.log(JSON.stringify(info, null, 1));
    await page.screenshot({ path: 'tools/probe_shot.png' });
    await browser.close();
})().catch((e) => { console.error('FAILED:', e); process.exit(1); });
