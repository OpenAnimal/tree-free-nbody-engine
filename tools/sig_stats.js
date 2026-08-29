// Luminance centroid/spread stats for screenshot files (mass/rx/ry like the bench sig).
const { chromium } = require('playwright');
const path = require('path');

(async () => {
    const files = process.argv.slice(2);
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    await page.goto('http://localhost:8123/index.html', { waitUntil: 'domcontentloaded' });
    const stats = await page.evaluate(async (fileUrls) => {
        const out = {};
        for (const [name, url] of fileUrls) {
            const img = new Image();
            img.src = url;
            await img.decode();
            const c = document.createElement('canvas'); c.width = 64; c.height = 64;
            const x = c.getContext('2d');
            x.drawImage(img, 0, 0, 64, 64);
            const d = x.getImageData(0, 0, 64, 64).data;
            let s = 0, sx = 0, sy = 0, sx2 = 0, sy2 = 0;
            for (let y = 0; y < 64; y++) for (let xx = 0; xx < 64; xx++) {
                const i = (y * 64 + xx) * 4;
                const lum = (d[i] + d[i + 1] + d[i + 2]) / 765;
                s += lum; sx += xx * lum; sy += y * lum; sx2 += xx * xx * lum; sy2 += y * y * lum;
            }
            const cx = sx / s, cy = sy / s;
            out[name] = {
                mass: +(s / (64 * 64)).toFixed(4),
                cx: +cx.toFixed(2), cy: +cy.toFixed(2),
                rx: +Math.sqrt(Math.max(0, sx2 / s - cx * cx)).toFixed(2),
                ry: +Math.sqrt(Math.max(0, sy2 / s - cy * cy)).toFixed(2),
            };
        }
        return out;
    }, files.map((f) => [path.basename(f), 'http://localhost:8123/tools/' + path.basename(f)]));
    console.log(JSON.stringify(stats, null, 1));
    await browser.close();
})().catch((e) => { console.error('ERR', e.message); process.exit(1); });
