// Captures real app footage for the brag video, frame-perfect: the page clock is
// paused and stepped 1/30s per frame, so rAF/framer-motion/timers advance exactly.
import { chromium } from "playwright";
import fs from "node:fs";

const APP = process.env.APP_URL || "http://localhost:3000";
const OUT = new URL("./out/cap/", import.meta.url).pathname;
fs.mkdirSync(OUT + "draw", { recursive: true });

const FPS = 30, DT = 1000 / FPS;
const bez = (p0, p1, p2, p3, n) =>
  Array.from({ length: n + 1 }, (_, i) => {
    const t = i / n, u = 1 - t;
    return [0, 1].map((k) => u * u * u * p0[k] + 3 * u * u * t * p1[k] + 3 * u * t * t * p2[k] + t * t * t * p3[k]);
  });
// handwritten "3": two bowls, one stroke
const STROKE = [
  ...bez([0.3, 0.24], [0.48, 0.12], [0.74, 0.2], [0.62, 0.38], 14),
  ...bez([0.62, 0.38], [0.56, 0.47], [0.46, 0.49], [0.42, 0.5], 6).slice(1),
  ...bez([0.42, 0.5], [0.66, 0.5], [0.8, 0.66], [0.62, 0.78], 14).slice(1),
  ...bez([0.62, 0.78], [0.5, 0.86], [0.34, 0.84], [0.27, 0.75], 8).slice(1),
];

const b = await chromium.launch();
try {
  const ctx = await b.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const p = await ctx.newPage();
  p.on("pageerror", (e) => console.log("pageerror", e.message));
  await p.clock.install();
  await p.goto(APP, { waitUntil: "networkidle", timeout: 180000 });
  await p.waitForFunction(() => !document.body.innerText.includes("LOADING WEIGHTS"), null, { timeout: 180000 });
  await p.waitForTimeout(1500);

  let n = 0;
  const shot = async () => {
    await p.waitForTimeout(25); // let real async work (WASM inference) land
    await p.screenshot({ path: `${OUT}draw/${String(n++).padStart(4, "0")}.png` });
  };
  const step = async () => { await p.clock.runFor(DT); await shot(); };

  await p.clock.pauseAt(await p.evaluate(() => Date.now() + 1000));
  const box = await p.locator('canvas[aria-label="Drawing canvas for character input"]').boundingBox();
  const P = ([u, v]) => [box.x + u * box.width, box.y + v * box.height];

  for (let i = 0; i < 15; i++) await step();                 // 0.5s idle hero
  await p.mouse.move(...P(STROKE[0]));
  await p.mouse.down();
  for (let i = 1; i < STROKE.length; i++) {                  // ~1.4s of drawing, 1 point/frame
    await p.mouse.move(...P(STROKE[i]));
    await step();
  }
  await p.mouse.up();
  for (let i = 0; i < 6 * FPS; i++) await step();             // shrink + reveal + settle
  console.log("draw frames", n, "stroke frames", STROKE.length);
  fs.writeFileSync(`${OUT}draw/meta.json`, JSON.stringify({ fps: FPS, frames: n, strokeStart: 15, strokeEnd: 15 + STROKE.length - 1 }));

  console.log("caption:", await p.evaluate(() => document.body.innerText.match(/FIG\. 1[^\n]*/)?.[0] || ""));
} finally {
  await b.close();
}
