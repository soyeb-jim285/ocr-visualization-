// Renders composition/index.html frame-by-frame (pure function of t) → brag.mp4 (+ poster, review stills).
import { chromium } from "playwright";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

const ROOT = path.dirname(new URL(import.meta.url).pathname);
const OUT = path.join(ROOT, "out");
const POSTER_T = 6.7;                                   // revealed network + "Watch a network read it."
const REVIEW_T = [0.4, 1.6, 2.8, 3.3, 4.6, 6.7, 7.4, 8.6, 11.0, 12.6, 14.6, 16.4, 18.2, 19.6, 21.0];

const MIME = { ".html": "text/html", ".png": "image/png", ".json": "application/json", ".js": "text/javascript" };
const server = http.createServer((req, res) => {
  const f = path.join(ROOT, decodeURIComponent(req.url.split("?")[0]));
  fs.readFile(f, (err, buf) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": MIME[path.extname(f)] || "application/octet-stream" });
    res.end(buf);
  });
}).listen(8123);

const b = await chromium.launch();
try {
  const p = await (await b.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 })).newPage();
  await p.goto("http://localhost:8123/composition/index.html", { waitUntil: "networkidle" });
  await p.evaluate(async () => {
    await document.fonts.ready;
    await Promise.all([...document.images].map((i) => i.decode().catch(() => {})));
  });
  const { DURATION, FPS } = await p.evaluate(() => ({ DURATION: window.DURATION, FPS: window.FPS }));
  const frame = async (t) => { await p.evaluate((t) => window.renderAt(t), t); return p.screenshot({ type: "png" }); };

  fs.mkdirSync(path.join(OUT, "review"), { recursive: true });
  for (const t of REVIEW_T) fs.writeFileSync(path.join(OUT, "review", `t${t.toFixed(1).padStart(4, "0")}.png`), await frame(t));
  const poster = await frame(POSTER_T);
  fs.writeFileSync(path.join(OUT, "brag-poster.png"), poster);

  const ff = spawn("ffmpeg", ["-y", "-f", "image2pipe", "-framerate", String(FPS), "-i", "-",
    "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", path.join(OUT, "video.mp4")],
    { stdio: ["pipe", "inherit", "inherit"] });
  const n = Math.round(DURATION * FPS);
  for (let i = 0; i < n; i++) {
    const png = i === 0 ? poster : await frame(i / FPS);   // frame 0 = poster (platform thumbnails)
    if (!ff.stdin.write(png)) await new Promise((r) => ff.stdin.once("drain", r));
    if (i % 60 === 0) console.log(`frame ${i}/${n}`);
  }
  ff.stdin.end();
  await new Promise((r, j) => ff.on("close", (c) => (c ? j(new Error("ffmpeg " + c)) : r())));
} finally {
  await b.close();
  server.close();
}
