/* Credit Nirvana Geocoder - front end. No dependencies; talks to /api/locate and /api/visit. */
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const TC = { h: "#2f9e44", m: "#e0a526", l: "#d64545" };
const TNAME = { h: "high", m: "medium", l: "low" };
const NS = "http://www.w3.org/2000/svg";
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = n => Math.round(n).toLocaleString("en-IN");
const TEAM = [
  ["Lakshay Shrivastava", "25MT10046"], ["Shivansh Goyal", "25MT10080"], ["Swarit Mishra", "25AG10010"], ["Devyansh Agarwal", "25MI10045"], ["Roumya Lenka", "25CE10006"],
  ["Vansh Lohia", "25ME10169"], ["Mayank Tiwari", "25AE10045"], ["Reshmika Jyothikumar", "25BT10061"], ["Devarati Mukherjee", "25CH10053"], ["Janees Zafar", "25MT10039"]
];

async function getJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(url + " " + r.status); return r.json(); }
async function postJSON(url, body, ms = 45000) {
  const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), ms);
  try {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: ctl.signal });
    const j = await r.json().catch(() => ({ error: "unexpected response from the server" }));
    if (!r.ok && !j.error) j.error = "request failed (" + r.status + ")";
    return j;
  } catch (e) { return { error: e.name === "AbortError" ? "the request timed out, please try again" : "the live API is not reachable (static view only)" }; }
  finally { clearTimeout(t); }
}

/* ------------------------------------------------------------------ team */
$("#teamGrid").innerHTML = TEAM.map(([n, r]) => `<div class="member"><div class="avatar">${n.split(" ").map(w => w[0]).slice(0, 2).join("")}</div><b>${esc(n)}</b><span>${r}</span></div>`).join("");

/* ------------------------------------------------------------------ svg chart helpers */
function svg(w, h, inner) { return `<svg viewBox="0 0 ${w} ${h}" role="img" xmlns="${NS}">${inner}</svg>`; }
const T = (x, y, s, o = {}) => `<text x="${x}" y="${y}" font-size="${o.size || 11}" fill="${o.fill || "#5b6781"}" text-anchor="${o.anchor || "start"}" ${o.weight ? `font-weight="${o.weight}"` : ""}>${esc(s)}</text>`;
function axisY(x0, y0, y1, ticks, label) { return ticks.map(([v, y]) => `<line x1="${x0}" x2="${y1}" y1="${y}" y2="${y}" stroke="#e6ebf4"/>` + T(x0 - 6, y + 4, v, { anchor: "end", size: 10 })).join(""); }

function chartImpact(S) {
  const d = S.impact, W = 520, H = 260, L = 38, B = 44, top = 18, bw = (W - L - 10) / d.length, mx = 0.45;
  let o = "";
  [0, .1, .2, .3, .4].forEach(v => { const y = top + (H - top - B) * (1 - v / mx); o += `<line x1="${L}" x2="${W - 6}" y1="${y}" y2="${y}" stroke="#e6ebf4"/>` + T(L - 6, y + 4, Math.round(v * 100) + "%", { anchor: "end", size: 10 }); });
  d.forEach((b, i) => {
    const h = (H - top - B) * b.not_traceable / mx, x = L + i * bw + bw * .14, y = H - B - h, last = i === d.length - 1;
    o += `<rect x="${x}" y="${y}" width="${bw * .72}" height="${h}" rx="6" fill="${last ? "#1d4486" : "#f5c542"}"/>`;
    o += T(x + bw * .36, y - 6, Math.round(b.not_traceable * 100) + "%", { anchor: "middle", weight: 700, fill: "#14305f", size: 12 });
    o += T(x + bw * .36, H - B + 18, b.bin, { anchor: "middle", size: 10.5 });
    o += T(x + bw * .36, H - B + 32, b.minutes.toFixed(0) + " min", { anchor: "middle", size: 9.5, fill: "#8a93a5" });
  });
  $("#chartImpact").innerHTML = svg(W, H, o);
}

function lineChart(el, series, opt) {
  const W = 520, H = 280, L = 48, R = 14, top = 14, B = 46, logx = opt.logx;
  const xs = series.flatMap(s => s.pts.map(p => p[0])), xmin = opt.xmin ?? Math.min(...xs), xmax = opt.xmax ?? Math.max(...xs);
  const X = v => L + (W - L - R) * (logx ? (Math.log10(v) - Math.log10(xmin)) / (Math.log10(xmax) - Math.log10(xmin)) : (v - xmin) / (xmax - xmin));
  const Y = v => top + (H - top - B) * (1 - (v - opt.ymin) / (opt.ymax - opt.ymin));
  let o = "";
  opt.yticks.forEach(v => { o += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e6ebf4"/>` + T(L - 6, Y(v) + 4, opt.yfmt(v), { anchor: "end", size: 10 }); });
  opt.xticks.forEach(v => { o += `<line x1="${X(v)}" x2="${X(v)}" y1="${top}" y2="${H - B}" stroke="#f0f3f9"/>` + T(X(v), H - B + 16, opt.xfmt(v), { anchor: "middle", size: 10 }); });
  o += T((L + W - R) / 2, H - 8, opt.xlabel, { anchor: "middle", size: 11 }) + `<text transform="translate(13 ${(top + H - B) / 2}) rotate(-90)" font-size="11" fill="#5b6781" text-anchor="middle">${esc(opt.ylabel)}</text>`;
  if (opt.diag) o += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="#8a93a5" stroke-dasharray="5 4"/>`;
  series.forEach(s => {
    if (s.hline != null) { o += `<line x1="${L}" x2="${W - R}" y1="${Y(s.hline)}" y2="${Y(s.hline)}" stroke="${s.color}" stroke-width="2" stroke-dasharray="6 5"/>`; return; }
    o += `<polyline fill="none" stroke="${s.color}" stroke-width="2.6" stroke-linejoin="round" points="${s.pts.map(p => X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join(" ")}"/>`;
    if (s.dots) s.pts.forEach(p => o += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3.6" fill="${s.color}"/>`);
  });
  let lx = L + 10;
  series.forEach(s => { o += `<rect x="${lx}" y="${top + 2}" width="14" height="4" rx="2" fill="${s.color}"/>` + T(lx + 20, top + 8, s.name, { size: 10.5, fill: "#14305f" }); lx += 28 + s.name.length * 6; });
  el.innerHTML = svg(W, H, o);
}

function renderCharts(S) {
  chartImpact(S);
  lineChart($("#chartCdf"), [
    { name: "old geocoder", color: "#8a93a5", pts: S.cdf_surveyed_old.filter(p => p[0] > 0) },
    { name: "our pins", color: "#1d4486", pts: S.cdf_surveyed_new.filter(p => p[0] > 0) }
  ], { logx: true, xmin: 3, xmax: 6000, ymin: 0, ymax: 1, yticks: [0, .25, .5, .75, 1], xticks: [10, 100, 1000], xfmt: v => v + " m", yfmt: v => Math.round(v * 100) + "%", xlabel: "distance error (log scale)", ylabel: "share of addresses" });
  const lp = S.learning.map(p => [p.n, p.median]);
  lineChart($("#chartLearn"), [
    { name: "old geocoder", color: "#8a93a5", hline: S.learning_old, pts: [] },
    { name: "our pins", color: "#2f9e44", pts: lp, dots: true }
  ], { xmin: 0, xmax: Math.max(...lp.map(p => p[0])), ymin: 100, ymax: 340, yticks: [100, 150, 200, 250, 300], xticks: [0, 200, 400, 600, 800], xfmt: v => v, yfmt: v => v + " m", xlabel: "visited addresses the system has learned from", ylabel: "median error (m)" });
  // ablation
  const A = S.ablation, W = 520, H = 40 + A.length * 38, mx = 400; let o = "";
  A.forEach((a, i) => {
    const y = 10 + i * 38, w = (W - 250) * a.median / mx, last = i === A.length - 1;
    o += T(0, y + 18, a.variant.replace("+ ", "+ ").replace(" (full model)", ""), { size: 11, fill: "#14305f" });
    o += `<rect x="178" y="${y + 3}" width="${Math.max(w, 2)}" height="22" rx="5" fill="${i === 0 ? "#8a93a5" : (last ? "#1d4486" : "#f5c542")}"/>` + T(178 + w + 6, y + 19, Math.round(a.median) + " m", { size: 11, weight: 700, fill: "#14305f" });
  });
  $("#chartAblation").innerHTML = svg(W, H, o);
  const rel = S.reliability, cols = ["#1d4486", "#2f9e44"];
  lineChart($("#chartRel"), [{ name: "perfect", color: "#c9d2e4", pts: [[0, 0], [1, 1]] }, ...rel.map((r, i) => ({ name: r.curve.startsWith("unvisited") ? "no visit (n=" + r.n + ")" : "visited (n=" + r.n + ")", color: cols[i % 2], pts: r.points, dots: true }))],
    { xmin: 0, xmax: 1, ymin: 0, ymax: 1, yticks: [0, .25, .5, .75, 1], xticks: [0, .25, .5, .75, 1], xfmt: v => Math.round(v * 100) + "%", yfmt: v => Math.round(v * 100) + "%", xlabel: "promised coverage", ylabel: "observed coverage" });
  const s = S.surveyed, hp = S.hide_and_predict, c = S.calibration;
  $("#metricRow").innerHTML = [
    [`${fmt(s.old_median)} → ${fmt(s.new_median)} m`, "median error, 100 surveyed addresses"],
    [`${fmt(s.old_p90)} → ${fmt(s.new_p90)} m`, "90th-percentile error"],
    [`${fmt(hp.old_median)} → ${fmt(hp.new_median)} m`, `addresses with no visit (${fmt(hp.n)} checked)`],
    [`${Math.round(c.cov90 * 100)}%`, `of true locations inside the 90% circle (95% CI ${Math.round(c.cov90_ci[0] * 100)}–${Math.round(c.cov90_ci[1] * 100)}%)`]
  ].map(([b, t]) => `<div class="metric"><b>${b}</b><span>${t}</span></div>`).join("");
  const q = S.tiers.quality, cnt = S.tiers.counts;
  $("#tierTable").innerHTML = `<table class="tbl"><tr><th>Tier</th><th style="text-align:left">Action</th><th>Addresses</th><th>Within 100 m</th><th>Median error</th></tr>` +
    [["high", "Visit now"], ["medium", "Visit with directions"], ["low", "Verify first"]].map(([t, a]) =>
      `<tr><td><span class="pill ${t}">${t}</span></td><td style="text-align:left">${a}</td><td>${fmt(cnt[t] || 0)}</td><td>${Math.round(q[t].within_100 * 100)}%</td><td>${Math.round(q[t].median_err)} m</td></tr>`).join("") + `</table>`;
}

/* ------------------------------------------------------------------ hero animation */
function heroAnim(town) {
  const cv = $("#heroCanvas"), cx = cv.getContext("2d"); let W = 0, H = 0, raf = 0, vis = true;
  const reduce = matchMedia("(prefers-reduced-motion:reduce)").matches;
  const A = town.addresses.filter((a, i) => i % 9 === 0).slice(0, 46);
  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9; A.forEach(a => { [[a.x, a.y], [a.ox, a.oy]].forEach(p => { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }); });
  function size() { const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1; W = r.width; H = r.height; cv.width = W * d; cv.height = H * d; cx.setTransform(d, 0, 0, d, 0, 0); }
  const P = (x, y) => { const s = Math.min(W * .62 / (x1 - x0), H * .8 / (y1 - y0)); return [W * .66 + (x - (x0 + x1) / 2) * s, H * .52 - (y - (y0 + y1) / 2) * s]; };
  function frame(t) {
    cx.clearRect(0, 0, W, H);
    A.forEach((a, i) => {
      const ph = ((t / 5200) + i * .137) % 1, k = Math.min(1, Math.max(0, (ph - .15) / .5)), e = k * k * (3 - 2 * k);
      const [ox, oy] = P(a.ox, a.oy), [nx, ny] = P(a.x, a.y);
      cx.strokeStyle = "rgba(255,255,255,.14)"; cx.lineWidth = 1; cx.beginPath(); cx.moveTo(ox, oy); cx.lineTo(ox + (nx - ox) * e, oy + (ny - oy) * e); cx.stroke();
      cx.fillStyle = "rgba(255,255,255,.35)"; cx.beginPath(); cx.arc(ox, oy, 3.2, 0, 6.3); cx.fill();
      const px = ox + (nx - ox) * e, py = oy + (ny - oy) * e;
      cx.fillStyle = `rgba(245,197,66,${.25 + .75 * e})`; cx.beginPath(); cx.arc(px, py, 3 + 3 * e, 0, 6.3); cx.fill();
      if (k >= 1) { const g = Math.min(1, (ph - .65) / .3); cx.strokeStyle = `rgba(245,197,66,${.5 * (1 - g)})`; cx.lineWidth = 1.5; cx.beginPath(); cx.arc(nx, ny, 6 + g * 24, 0, 6.3); cx.stroke(); }
    });
  }
  function loop(t) { if (vis) frame(t); raf = requestAnimationFrame(loop); }
  size(); addEventListener("resize", size);
  new IntersectionObserver(e => { vis = e[0].isIntersecting; }).observe(cv);
  if (reduce) frame(2600); else raf = requestAnimationFrame(loop);
}

/* ------------------------------------------------------------------ map */
const M = { town: null, data: {}, view: { x: 0, y: 0, s: .1 }, sel: null, loc: null, teach: null, pick: null, tab: "inspect", W: 0, H: 0, dpr: 1 };
const cv = $("#map"), ctx = cv.getContext("2d");
const sx = x => (x - M.view.x) * M.view.s + M.W / 2, sy = y => M.H / 2 - (y - M.view.y) * M.view.s;
const wx = px => (px - M.W / 2) / M.view.s + M.view.x, wy = py => M.view.y - (py - M.H / 2) / M.view.s;

function resizeMap() { const r = cv.getBoundingClientRect(); M.dpr = devicePixelRatio || 1; M.W = r.width; M.H = r.height; cv.width = M.W * M.dpr; cv.height = M.H * M.dpr; ctx.setTransform(M.dpr, 0, 0, M.dpr, 0, 0); if (!M.fitted && M.town) fitMap(); draw(); }
function fitMap() {
  const T = M.data[M.town]; let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
  T.addresses.forEach(a => [[a.x, a.y], [a.ox, a.oy]].forEach(p => { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }));
  M.view = { x: (x0 + x1) / 2, y: (y0 + y1) / 2, s: Math.min(M.W / (x1 - x0 + 500), M.H / (y1 - y0 + 500)) }; M.fitted = true;
}
function zoomAt(px, py, k) { const bx = wx(px), by = wy(py); M.view.s = Math.min(Math.max(M.view.s * k, .02), 8); M.view.x = bx - (px - M.W / 2) / M.view.s; M.view.y = by + (py - M.H / 2) / M.view.s; draw(); }
function flyTo(x, y, s) { M.view.x = x; M.view.y = y; if (s) M.view.s = Math.max(M.view.s, s); draw(); }
function circ(x, y, r, fill, stroke, w = 1.5, dash) { ctx.beginPath(); ctx.arc(sx(x), sy(y), Math.max(r * M.view.s, 1), 0, 6.2832); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = w; ctx.setLineDash(dash || []); ctx.stroke(); ctx.setLineDash([]); } }
function diamond(x, y, r, fill) { const X = sx(x), Y = sy(y); ctx.beginPath(); ctx.moveTo(X, Y - r); ctx.lineTo(X + r, Y); ctx.lineTo(X, Y + r); ctx.lineTo(X - r, Y); ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke(); }
function arrow(a, b, color, w = 2) { const x0 = sx(a[0]), y0 = sy(a[1]), x1 = sx(b[0]), y1 = sy(b[1]); ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); const an = Math.atan2(y1 - y0, x1 - x0); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - 8 * Math.cos(an - .4), y1 - 8 * Math.sin(an - .4)); ctx.lineTo(x1 - 8 * Math.cos(an + .4), y1 - 8 * Math.sin(an + .4)); ctx.closePath(); ctx.fill(); }
function ring(x, y, r, color) { ctx.beginPath(); ctx.arc(sx(x), sy(y), r, 0, 6.2832); ctx.fillStyle = "#fff"; ctx.fill(); ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.stroke(); }
function dot(x, y, r, color) { ctx.beginPath(); ctx.arc(sx(x), sy(y), r, 0, 6.2832); ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke(); }
function star(x, y, r, color) { const X = sx(x), Y = sy(y); ctx.beginPath(); for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * .45 : r; ctx.lineTo(X + rr * Math.cos(a), Y + rr * Math.sin(a)); } ctx.closePath(); ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = "#14305f"; ctx.lineWidth = 1.5; ctx.stroke(); }

function draw() {
  if (!M.town || !M.data[M.town]) return;
  const T = M.data[M.town]; ctx.clearRect(0, 0, M.W, M.H);
  ctx.fillStyle = "#f7f9fd"; ctx.fillRect(0, 0, M.W, M.H);
  ctx.font = "600 11px Inter, Segoe UI, sans-serif"; ctx.textAlign = "center"; ctx.fillStyle = "rgba(100,115,145,.7)";
  T.localities.forEach(l => ctx.fillText(l.locality_name, sx(l.centroid_x), sy(l.centroid_y)));
  if ($("#lLm").checked) T.landmarks.forEach(l => { ctx.fillStyle = "#7a6bb0"; ctx.fillRect(sx(l.x) - 3, sy(l.y) - 3, 6, 6); if (M.view.s > .25) { ctx.fillStyle = "rgba(100,86,150,.95)"; ctx.font = "10px Inter, sans-serif"; ctx.textAlign = "left"; ctx.fillText(l.name, sx(l.x) + 6, sy(l.y) + 3); ctx.textAlign = "center"; } });
  const A = T.addresses, inView = a => { const X = sx(a.x), Y = sy(a.y); return X > -60 && X < M.W + 60 && Y > -60 && Y < M.H + 60; };
  if ($("#lCirc").checked && M.view.s > .1) A.forEach(a => { if (inView(a)) circ(a.x, a.y, a.r9, TC[a.t] + "14", TC[a.t] + "55", 1); });
  if ($("#lOld").checked) A.forEach(a => { ctx.fillStyle = "rgba(138,147,165,.55)"; ctx.fillRect(sx(a.ox) - 1.6, sy(a.oy) - 1.6, 3.2, 3.2); });
  if ($("#lNew").checked) A.forEach(a => { ctx.fillStyle = TC[a.t]; ctx.beginPath(); ctx.arc(sx(a.x), sy(a.y), a.v ? 2.8 : 2.3, 0, 6.2832); ctx.fill(); });
  if ($("#lTruth").checked) A.forEach(a => { if (a.tx === undefined) return; ctx.strokeStyle = "rgba(0,0,0,.35)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(sx(a.ox), sy(a.oy)); ctx.lineTo(sx(a.tx), sy(a.ty)); ctx.stroke(); diamond(a.tx, a.ty, 5, "#111"); });
  const s = M.sel; if (s && s.town === M.town) {
    const a = s.a; circ(a.x, a.y, a.r9, TC[a.t] + "24", TC[a.t], 2); circ(a.x, a.y, a.r5, TC[a.t] + "38", TC[a.t], 1.5);
    ctx.setLineDash([5, 4]); ctx.strokeStyle = "#8a93a5"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(sx(a.ox), sy(a.oy)); ctx.lineTo(sx(a.x), sy(a.y)); ctx.stroke(); ctx.setLineDash([]);
    ring(a.ox, a.oy, 6.5, "#8a93a5"); dot(a.x, a.y, 7.5, TC[a.t]); if (a.tx !== undefined) diamond(a.tx, a.ty, 9, "#111");
  }
  const L = M.loc; if (L && L.town === M.town) {
    circ(L.pin.x, L.pin.y, L.r90, TC[L.t] + "22", TC[L.t], 2); circ(L.pin.x, L.pin.y, L.r50, TC[L.t] + "38", TC[L.t], 1.5);
    ctx.setLineDash([5, 4]); ctx.strokeStyle = "#8a93a5"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(sx(L.old.x), sy(L.old.y)); ctx.lineTo(sx(L.pin.x), sy(L.pin.y)); ctx.stroke(); ctx.setLineDash([]);
    ring(L.old.x, L.old.y, 7, "#8a93a5"); dot(L.pin.x, L.pin.y, 9, TC[L.t]); if (L.truth) diamond(L.truth[0], L.truth[1], 10, "#111");
  }
  const K = M.teach; if (K && K.town === M.town) {
    K.res.neighbours.forEach(n => { arrow(n.before, n.after, "#1d4486", 1.8); dot(n.after[0], n.after[1], 4, "#1d4486"); ring(n.before[0], n.before[1], 3.6, "#8a93a5"); });
    arrow(K.res.target.before, K.res.target.after, "#f08c3c", 2.6); dot(K.res.target.after[0], K.res.target.after[1], 7, "#2f9e44");
    star(K.res.visit[0], K.res.visit[1], 13, "#f5c542");
  }
  if (M.pick && M.pick.town === M.town) star(M.pick.x, M.pick.y, 12, "#f5c542");
}
function nearest(px, py) { const T = M.data[M.town]; let b = null, bd = 144; T.addresses.forEach(a => { const d = (sx(a.x) - px) ** 2 + (sy(a.y) - py) ** 2; if (d < bd) { bd = d; b = a; } }); return b; }

let drag = null;
cv.addEventListener("pointerdown", e => { cv.setPointerCapture(e.pointerId); drag = { x: e.clientX, y: e.clientY, vx: M.view.x, vy: M.view.y, moved: false }; cv.style.cursor = "grabbing"; });
cv.addEventListener("pointermove", e => { if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true; if (drag.moved) { M.view.x = drag.vx - dx / M.view.s; M.view.y = drag.vy + dy / M.view.s; draw(); } });
cv.addEventListener("pointerup", e => {
  cv.style.cursor = "grab"; const d = drag; drag = null; if (!d || d.moved) return;
  const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top, a = nearest(px, py);
  if (M.tab === "teach" && !a) { M.pick = { town: M.town, x: wx(px), y: wy(py) }; teachUpdate(); draw(); return; }
  if (a) { select(a.id, false); } else if (M.tab === "teach" && M.sel) { M.pick = { town: M.town, x: wx(px), y: wy(py) }; teachUpdate(); draw(); }
});
cv.addEventListener("wheel", e => { e.preventDefault(); const r = cv.getBoundingClientRect(); zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * .0015)); }, { passive: false });
$("#zoomIn").onclick = () => zoomAt(M.W / 2, M.H / 2, 1.5); $("#zoomOut").onclick = () => zoomAt(M.W / 2, M.H / 2, 1 / 1.5); $("#zoomFit").onclick = () => { fitMap(); draw(); };
$$(".toggles input").forEach(i => i.addEventListener("change", draw));
addEventListener("resize", resizeMap);

/* ------------------------------------------------------------------ town + selection */
async function loadTown(id) { if (!M.data[id]) M.data[id] = await getJSON(`data/town_${id}.json`); return M.data[id]; }
async function setTown(id, keep) {
  await loadTown(id); M.town = id; M.fitted = false; $$("#townTabs button").forEach(b => b.classList.toggle("on", b.dataset.t === id));
  if (!keep) { M.sel = null; M.loc = null; M.teach = null; M.pick = null; renderInspect(); renderTeachResult(null); } fitMap(); draw();
}
async function select(id, fly = true) {
  const town = id.startsWith("AD") ? null : null; let a = null, tid = null;
  for (const t of Object.keys(M.data)) { const f = M.data[t].addresses.find(x => x.id === id); if (f) { a = f; tid = t; break; } }
  if (!a) for (const t of ["T1", "T2", "T3"]) { await loadTown(t); const f = M.data[t].addresses.find(x => x.id === id); if (f) { a = f; tid = t; break; } }
  if (!a) { $("#inspectBody").innerHTML = `<div class="err">No address with ID “${esc(id)}”.</div>`; return; }
  if (tid !== M.town) await setTown(tid, true);
  M.sel = { a, town: tid }; M.loc = null; M.teach = null; if (M.tab !== "teach") M.pick = null;
  if (fly && M.view.s < .5) flyTo(a.x, a.y, 1.1); else draw();
  renderInspect(); teachUpdate(); renderTeachResult(null);
}
function renderInspect() {
  const el = $("#inspectBody"), s = M.sel;
  if (!s) { el.innerHTML = `<div class="box"><h4>Pick an address</h4>Click any pin on the map, search an ID, or switch to <b>Locate an address</b> to type your own. Surveyed addresses show a black diamond: the true location, so you can see old vs new error.</div>`; return; }
  const a = s.a, why = a.w.split("|").map(w => w.replace(/_/g, " ").toLowerCase()).join(", ");
  el.innerHTML = `<div class="addr">${esc(a.s)}</div>
  <div><span class="pill ${TNAME[a.t]}">${TNAME[a.t]} confidence</span> &nbsp;<b>${esc(a.a.replace(/_/g, " "))}</b></div>
  <div class="kv"><span>ID</span><span>${a.id} &middot; ${esc(M.data[s.town].name)}</span><span>90% circle</span><span>${fmt(a.r9)} m</span><span>50% circle</span><span>${fmt(a.r5)} m</span><span>Moved from old pin</span><span>${fmt(Math.hypot(a.x - a.ox, a.y - a.oy))} m</span><span>Based on</span><span>${esc(a.m.replace(/_/g, " "))}${a.v ? " (own field visit)" : ""}</span><span>Why</span><span>${esc(why)}</span></div>
  ${a.tx !== undefined ? `<div class="compare"><div class="old"><small>old geocoder</small><b>${fmt(a.eo)} m</b></div><div class="new"><small>our pin</small><b>${fmt(a.en)} m</b></div></div><p class="muted small">Distance from the surveyed true location. ${a.en <= a.r9 ? "The truth is inside our 90% circle." : "The truth is outside the 90% circle (expected for about 1 in 10)."}</p>` : ""}
  <div class="box"><h4>Directions for the agent (work offline)</h4>${esc(a.d)}${a.dl ? `<br><span class="muted small">${esc(a.dl)}</span>` : ""}</div>`;
}
$("#searchGo").onclick = () => { const v = $("#searchId").value.trim().toUpperCase(); if (v) select(v); };
$("#searchId").addEventListener("keydown", e => { if (e.key === "Enter") $("#searchGo").click(); });

/* ------------------------------------------------------------------ tabs */
$$(".tab").forEach(t => t.onclick = () => { M.tab = t.dataset.tab; $$(".tab").forEach(x => x.classList.toggle("on", x === t)); $$(".tabpane").forEach(p => p.classList.toggle("on", p.id === "pane-" + M.tab)); if (M.tab !== "teach") M.pick = null; draw(); });
function showTab(n) { $(`.tab[data-tab="${n}"]`).click(); }

/* ------------------------------------------------------------------ locate */
let CASES = { examples: [], scenarios: [] }, TOWNS = [];
$("#locGo").onclick = async () => {
  const text = $("#locText").value.trim(), town = $("#locTown").value, body = $("#locBody");
  if (text.length < 6) { body.innerHTML = `<div class="err">Please enter a fuller address.</div>`; return; }
  $("#locGo").disabled = true; body.innerHTML = `<p class="muted"><span class="spin"></span>Parsing the address and fusing clues&hellip;</p>`;
  const ex = CASES.examples.find(e => e.text === text && e.town === town);
  const r = await postJSON("/api/locate", { text, town }); $("#locGo").disabled = false;
  if (r.error) { body.innerHTML = `<div class="err">${esc(r.error)}</div>`; return; }
  await setTown(town, true);
  M.sel = null; M.teach = null; M.loc = { town, pin: r.pin, old: r.old_pin, r50: r.radius_50, r90: r.radius_90, t: r.tier[0], truth: ex ? ex.truth : null };
  flyTo(r.pin.x, r.pin.y, 1.0); renderInspect();
  const p = r.parsed, bits = [r.locality_name && "locality: " + r.locality_name, p.cross && p.main && `${p.cross}th cross / ${p.main}th main`, p.gali && "gali " + p.gali, p.block && p.road && `block ${p.block.toUpperCase()}, road ${p.road}`, p.ward && "ward " + p.ward, p.lm_type && "landmark: " + p.lm_type.replace(/_/g, " ")].filter(Boolean);
  let cmp = `<div class="compare"><div class="old"><small>old geocoder placed it</small><b>${fmt(r.moved_m)} m</b><small>from our pin</small></div><div class="new"><small>our 90% circle</small><b>${fmt(r.radius_90)} m</b><small>radius</small></div></div>`;
  if (ex) { cmp = `<div class="compare"><div class="old"><small>old geocoder error</small><b>${fmt(ex.old_error)} m</b></div><div class="new"><small>our error</small><b>${fmt(Math.hypot(r.pin.x - ex.truth[0], r.pin.y - ex.truth[1]))} m</b></div></div><p class="muted small">Measured against the surveyed true location (black diamond). This address has no visit, so the model never saw its location.</p>`; }
  body.innerHTML = `<div style="margin-top:12px"><span class="pill ${r.tier}">${r.tier} confidence</span> &nbsp;<b>${esc(r.action.replace(/_/g, " "))}</b></div>
   <p class="muted small" style="margin:6px 0">${esc(r.advice)}</p>${cmp}
   <div class="box"><h4>Directions</h4>${esc(r.directions)}${r.directions_local ? `<br><span class="muted small">${esc(r.directions_local)}</span>` : ""}</div>
   <div class="box"><h4>What the parser understood</h4>${bits.length ? esc(bits.join(" · ")) : "no locality or street keys found (a pincode-level address)"}<br><span class="muted small">script: ${esc(r.script)} &middot; ${r.n_candidate_localities > 1 ? r.n_candidate_localities + " candidate localities" : "locality resolved"}</span></div>
   <div class="box"><h4>Why this pin</h4>Based on <b>${esc(r.method.replace(/_/g, " "))}</b> (clues: ${esc(r.clues.split(",").join(", "))}).<br><span class="muted small">${r.reason_codes.map(c => esc(c.replace(/_/g, " ").toLowerCase())).join("; ")}</span></div>`;
};
$("#locTown").onchange = () => setTown($("#locTown").value);

/* ------------------------------------------------------------------ teach */
function teachUpdate() {
  const ok = !!(M.sel && M.pick && M.pick.town === M.town);
  $("#teachGo").disabled = !ok;
  $("#teachPick").innerHTML = M.sel ? (M.pick ? `Visit location set for <b>${M.sel.a.id}</b>. Click the map to move it, then press the button.` : `Selected <b>${M.sel.a.id}</b>. Now click the map where the agent met the borrower.`) : `Or select an address in <b>Inspect</b>, then click the map to place the visit.`;
}
function renderTeachResult(r) {
  const el = $("#teachBody"); if (!r) { el.innerHTML = ""; return; }
  if (r.error) { el.innerHTML = `<div class="err">${esc(r.error)}</div>`; return; }
  const med = r.neighbours.length ? [...r.neighbours].map(n => n.moved_m).sort((a, b) => a - b)[Math.floor(r.neighbours.length / 2)] : 0;
  el.innerHTML = `<div class="box"><h4>One confirmed visit in ${esc(r.locality || "the locality")}</h4><b>${r.neighbours_moved}</b> of ${r.neighbours_checked} unvisited neighbours re-estimated their pin (median move ${fmt(med)} m). The confirmed address's own uncertainty fell from ${fmt(r.target.spread_before)} m to ${fmt(r.target.spread_after)} m. No retraining.</div>
  <table class="tbl"><tr><th>Neighbour on the same street</th><th>Pin moved</th></tr>${r.neighbours.slice(0, 8).map(n => `<tr><td title="${esc(n.text)}">${n.id}<br><span class="muted small">${esc(n.text.length > 44 ? n.text.slice(0, 42) + "…" : n.text)}</span></td><td>${fmt(n.moved_m)} m</td></tr>`).join("")}</table>
  <p class="muted small">Blue arrows on the map: each neighbour's pin before (grey ring) and after (blue dot). Gold star: the confirmed visit.</p>`;
}
async function runTeach(id, x, y, townId) {
  const body = $("#teachBody"); body.innerHTML = `<p class="muted"><span class="spin"></span>Adding the visit and re-estimating the street&hellip;</p>`; $("#teachGo").disabled = true;
  const r = await postJSON("/api/visit", { address_id: id, x, y }); renderTeachResult(r); teachUpdate();
  if (!r.error) { M.teach = { town: townId, res: r }; M.pick = null; draw(); }
}
$("#teachGo").onclick = () => { if (M.sel && M.pick) runTeach(M.sel.a.id, M.pick.x, M.pick.y, M.town); };

/* ------------------------------------------------------------------ boot */
async function boot() {
  const S = await getJSON("data/stats.json"); TOWNS = S.towns; renderCharts(S);
  const sel = $("#locTown"); sel.innerHTML = TOWNS.map(t => `<option value="${t.town_id}">${esc(t.name)} (${t.town_id})</option>`).join("");
  $("#townTabs").innerHTML = TOWNS.map(t => `<button role="tab" data-t="${t.town_id}">${esc(t.name)}</button>`).join("");
  $$("#townTabs button").forEach(b => b.onclick = () => setTown(b.dataset.t));
  CASES = await getJSON("data/demo_cases.json").catch(() => ({ examples: [], scenarios: [] }));
  $("#locExamples").innerHTML = CASES.examples.map((e, i) => `<button class="chip" data-i="${i}" title="${esc(e.text)}">${esc(e.text.length > 38 ? e.text.slice(0, 36) + "…" : e.text)}<small>${esc(TOWNS.find(t => t.town_id === e.town).name)} &middot; held-out address</small></button>`).join("");
  $$("#locExamples .chip").forEach(c => c.onclick = () => { const e = CASES.examples[+c.dataset.i]; $("#locText").value = e.text; $("#locTown").value = e.town; $("#locGo").click(); });
  $("#teachScenarios").innerHTML = CASES.scenarios.map((e, i) => `<button class="chip" data-i="${i}" title="${esc(e.text)}">Replay a real visit in ${esc(e.locality || "")}<small>${esc(TOWNS.find(t => t.town_id === e.town).name)} &middot; ${e.neighbours_moved} neighbours move</small></button>`).join("");
  $$("#teachScenarios .chip").forEach(c => c.onclick = async () => { const e = CASES.scenarios[+c.dataset.i]; await select(e.id, true); M.pick = { town: M.town, x: e.visit[0], y: e.visit[1] }; flyTo(e.visit[0], e.visit[1], .9); teachUpdate(); runTeach(e.id, e.visit[0], e.visit[1], e.town); });
  const first = TOWNS[0].town_id; await setTown(first); resizeMap();
  const T1 = M.data[first]; heroAnim(T1);
  const demo = T1.addresses.find(a => a.tx !== undefined && a.t === "m") || T1.addresses.find(a => a.tx !== undefined); if (demo) select(demo.id, true);
}
boot().catch(e => { console.error(e); const h = $("#mapHint"); if (h) h.textContent = "Could not load data: " + e.message; });
