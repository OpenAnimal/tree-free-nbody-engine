const puppeteer = require('puppeteer');

const TESTS = [
    { label: '120k', n: 120000, preset: 'btnPreset120k' },
    { label: '500k', n: 500000, preset: 'btnPreset500k' },
    { label: '5M',   n: 5000000, preset: 'btnPreset5M' },
];
const WARMUP_MS = 4000;
const MEASURE_MS = 8000;

async function tryLaunch(args, label) {
    console.log(`\nTrying ${label}...`);
    const browser = await puppeteer.launch({
        head: false,
        executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
        args
    });
    try {
        const page = await browser.newPage();
        await page.setViewport({ width: 1280, height: 720 });
        await page.goto('http://localhost:8000/index.html', { waitUntil: 'domcontentloaded', timeout: 15000 });
        const debug = await page.evaluate(async () => {
            try {
                if (!navigator.gpu) return 'no navigator.gpu';
                const a = await navigator.gpu.requestAdapter({});
                if (!a) return 'adapter=null';
                let info = 'ok';
                if (a.info) info = `${a.info.vendor||'?'} ${a.info.architecture||''}`;
                return 'adapter=' + info;
            } catch (e) { return 'err:' + e.message; }
        });
        console.log(`  Result: ${debug}`);
        if (debug.startsWith('adapter=')) {
            return { browser, page };
        }
    } catch (e) {
        console.log(`  Error: ${e.message}`);
    }
    await browser.close();
    return null;
}

async function runTest() {
    // Try multiple GPU backend configurations
    const configs = [
        { label: 'D3D11', args: ['--enable-unsafe-webgpu', '--use-gl=angle', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--no-sandbox', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows', '--disable-features=CalculateNativeWinOcclusion', '--window-size=1280,720'] },
        { label: 'D3D9', args: ['--enable-unsafe-webgpu', '--use-gl=angle', '--use-angle=d3d9', '--ignore-gpu-blocklist', '--no-sandbox', '--window-size=1280,720'] },
        { label: 'default', args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-sandbox', '--window-size=1280,720'] },
        { label: 'Vulkan', args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-gl=angle', '--use-angle=vulkan', '--ignore-gpu-blocklist', '--no-sandbox', '--window-size=1280,720'] },
    ];

    let browser, page;
    for (const c of configs) {
        const result = await tryLaunch(c.args, c.label);
        if (result) { browser = result.browser; page = result.page; break; }
    }

    if (!browser) {
        console.log('\n*** Could not get WebGPU in any config. Falling back to WebGL test. ***');
        browser = await puppeteer.launch({
            head: false,
            executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
            args: ['--no-sandbox', '--window-size=1280,720']
        });
        page = await browser.newPage();
        await page.setViewport({ width: 1280, height: 720 });
        await page.goto('http://localhost:8000/index.html', { waitUntil: 'networkidle0', timeout: 30000 });
    }

    const consoleErrors = [];
    page.on('console', (msg) => { if (msg.type() === 'error') consoleErrors.push(msg.text()); });
    page.on('pageerror', (err) => consoleErrors.push(`PAGE: ${err.message}`));

    await page.waitForFunction(() => {
        const el = document.getElementById('engineBadge');
        return el && el.innerText && !el.innerText.includes('Detecting');
    }, { timeout: 20000 });

    const backend = await page.evaluate(() => document.getElementById('engineBadge').innerText);
    const gpuName = await page.evaluate(() => document.getElementById('gpuName').innerText);
    console.log(`\nBackend: ${backend}`);
    console.log(`GPU: ${gpuName}`);

    const isWebGPU = backend.includes('WebGPU');
    if (!isWebGPU) console.log('*** WebGPU NOT active - FMM telemetry will be zero ***');

    // Ensure galaxy scenario
    await page.evaluate(() => { const b = document.getElementById('btnGalaxy'); if (b) b.click(); });
    await new Promise(r => setTimeout(r, 500));

    for (const test of TESTS) {
        console.log(`\n=== ${test.label} particles ===`);
        await page.evaluate((id) => { const b = document.getElementById(id); if (b) b.click(); }, test.preset);
        await new Promise(r => setTimeout(r, 1000));

        const activeN = await page.evaluate(() => document.getElementById('valActiveParticles')?.innerText);
        console.log(`  Active: ${activeN}`);

        console.log(`  Warming up ${WARMUP_MS}ms...`);
        await new Promise(r => setTimeout(r, WARMUP_MS));

        console.log(`  Measuring ${MEASURE_MS}ms...`);
        const telemetry = await page.evaluate((ms) => {
            return new Promise((resolve) => {
                const samples = [];
                const start = Date.now();
                const iv = setInterval(() => {
                    const g = (id) => { const el = document.getElementById(id); if (!el) return 0; return parseFloat(el.innerText.replace(/[^0-9.\-]/g, '')) || 0; };
                    samples.push({
                        fps: g('valFPS'), stepMs: g('valComputeTime'),
                        fmmBuild: g('valFmmBuild'), fmmM2l: g('valFmmM2l'),
                        fmmL2p: g('valFmmL2p'), mainCompute: g('valMainCompute'),
                        render: g('valRenderPass'),
                        totalGpu: g('valTotalGpu'),
                        gpuComplete: g('valGpuComplete'),
                        hashLoad: document.getElementById('valFmmHash')?.innerText || '',
                        p2pBudget: document.getElementById('valP2pBudget')?.innerText || '',
                    });
                    if (Date.now() - start >= ms) { clearInterval(iv); resolve(samples); }
                }, 400);
            });
        }, MEASURE_MS);

        const v = telemetry.filter(s => s.fps > 0);
        if (!v.length) { console.log('  No data'); continue; }
        const avg = (a, k) => a.reduce((s, x) => s + x[k], 0) / a.length;
        const fps = avg(v, 'fps');
        const tput = (test.n * fps) / 1e6;

        console.log(`  FPS: ${fps.toFixed(1)} | Step: ${avg(v, 'stepMs').toFixed(3)} ms | Throughput: ${tput.toFixed(1)}M/s`);
        console.log(`  FMM Build: ${avg(v, 'fmmBuild').toFixed(3)} ms | M2L: ${avg(v, 'fmmM2l').toFixed(3)} ms | L2P: ${avg(v, 'fmmL2p').toFixed(3)} ms`);
        console.log(`  Main Compute: ${avg(v, 'mainCompute').toFixed(3)} ms | Render: ${avg(v, 'render').toFixed(3)} ms`);
        const totalGpuTs = avg(v, 'totalGpu');
        const totalGpu = avg(v, 'fmmBuild') + avg(v, 'fmmM2l') + avg(v, 'fmmL2p') + avg(v, 'mainCompute') + avg(v, 'render');
        const frameTime = 1000 / fps;
        console.log(`  Total GPU (sum): ${totalGpu.toFixed(3)} ms | Total GPU (ts): ${totalGpuTs.toFixed(3)} ms | GPU Complete: ${avg(v, 'gpuComplete').toFixed(3)} ms`);
        console.log(`  Frame time: ${frameTime.toFixed(1)} ms | Gap (frame - GPU fence): ${(frameTime - avg(v, 'gpuComplete')).toFixed(1)} ms`);
        console.log(`  Hash: ${v[0].hashLoad} | P2P: ${v[0].p2pBudget}`);
        if (consoleErrors.length > 0) {
            const unique = [...new Set(consoleErrors.map(e => e.substring(0, 120)))];
            console.log(`  Errors: ${consoleErrors.length} (${unique.length} unique)`);
            unique.slice(0, 3).forEach(e => console.log(`    ${e}`));
        } else {
            console.log(`  Errors: 0`);
        }
        consoleErrors.length = 0;
    }

    await browser.close();
    console.log('\nDone.');
}

runTest().catch(err => { console.error('FAILED:', err); process.exit(1); });
