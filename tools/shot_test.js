// Screenshot-pair test: does the composited canvas evolve over 10s?
const { chromium } = require('playwright');
const fs = require('fs');
const EXE = [
    'C:/Users/Y/AppData/Local/ms-playwright/chromium-1234/chrome-win64/chrome.exe',
    ...fs.readdirSync(process.env.LOCALAPPDATA + '/ms-playwright', { withFileTypes: true })
        .filter((d) => d.isDirectory() && d.name.startsWith('chromium-') && !d.name.includes('headless'))
        .map((d) => `${process.env.LOCALAPPDATA}/ms-playwright/${d.name}/chrome-win64/chrome.exe`),
].filter((p) => fs.existsSync(p));

const MODE = process.argv[2] || 'off';
const N = process.argv[3] || '500000';

(async () => {
    const browser = await chromium.launch({
        headless: true, executablePath: EXE[0],
        args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-sandbox',
               '--use-gl=angle', '--use-angle=d3d11',
               '--disable-background-timer-throttling', '--disable-renderer-backgrounding'],
    });
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    await page.goto(`http://localhost:8123/index.html?preset=120k&scenario=galaxy&n=${N}`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
        const el = document.getElementById('engineBadge');
        return el && el.innerText && !el.innerText.includes('Detecting');
    }, { timeout: 30000 });
    await page.evaluate((m) => {
        document.getElementById('selectFmmMode').value = m;
        document.getElementById('selectFmmMode').dispatchEvent(new Event('change'));
    }, MODE);
    await page.waitForTimeout(5000);
    const a = await page.screenshot();
    await page.waitForTimeout(10000);
    const b = await page.screenshot();
    console.log(`mode=${MODE}: screenshots identical=${a.equals(b)} sizes=${a.length}/${b.length}`);
    fs.writeFileSync('tools/shot_a.png', a);
    fs.writeFileSync('tools/shot_b.png', b);
    await browser.close();
})().catch((e) => { console.error('ERR', e && e.message); process.exit(1); });
