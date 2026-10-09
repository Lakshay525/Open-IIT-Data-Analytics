"""Presentation outputs: figures, the offline demo map and the headline-numbers summary.

Run: python -m geocoder.present  ->  outputs/demo_map.html, outputs/results/fig_*.png, outputs/summary.md, summary_metrics.json
"""
from __future__ import annotations

import json
import os
from tempfile import TemporaryDirectory

_mpl = TemporaryDirectory(prefix="mpl_")
os.environ.setdefault("MPLCONFIGDIR", _mpl.name)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
import pandas as pd

from .paths import DATA, OUT, RES


# ======================================================================
# demo map
# ======================================================================
def demo_map():
    pred = pd.read_csv(OUT / "predictions.csv")
    addr = pd.read_csv(DATA / "addresses.csv").set_index("address_id")
    sv = pd.read_csv(DATA / "surveyed_addresses.csv").set_index("address_id")
    lm = pd.read_csv(DATA / "landmarks_poi.csv")
    loc = pd.read_csv(DATA / "localities.csv")
    towns = pd.read_csv(DATA / "towns.csv")
    ps2 = pd.read_csv(OUT / "ps2_location_confidence.csv").set_index("address_id")

    rows = []
    for r in pred.itertuples():
        d = {"id": r.address_id, "t": r.town_id, "ox": r.old_pin_x, "oy": r.old_pin_y, "x": r.pin_x, "y": r.pin_y,
             "r5": r.radius_50, "r9": r.radius_90, "tier": r.confidence_tier[0], "act": r.suggested_action,
             "dir": r.directions, "dl": r.directions_local if isinstance(r.directions_local, str) else "",
             "why": r.reason_codes, "m": r.method_used, "txt": addr.at[r.address_id, "address_text"],
             "v": int(r.has_own_visit), "c": round(float(r.location_confidence), 2),
             "hf": int(ps2.at[r.address_id, "hard_to_find_flag"])}
        if r.address_id in sv.index:
            tx, ty = sv.at[r.address_id, "surveyed_x"], sv.at[r.address_id, "surveyed_y"]
            d["tx"], d["ty"] = round(float(tx), 1), round(float(ty), 1)
            d["eo"] = round(float(np.hypot(r.old_pin_x - tx, r.old_pin_y - ty)), 1)
            d["en"] = round(float(np.hypot(r.pin_x - tx, r.pin_y - ty)), 1)
        rows.append(d)

    ab = pd.read_csv(RES / "pin_model_ablation.csv")
    cov = pd.read_csv(RES / "calibration_coverage.csv")
    lc = pd.read_csv(RES / "learning_curve.csv")
    imp = pd.read_csv(RES / "impact_estimate.csv")
    hp = pd.read_csv(RES / "hide_and_predict.csv")
    tq = pd.read_csv(RES / "confidence_tier_quality.csv").set_index("tier")
    full = ab[ab.variant.str.startswith("5")].iloc[0]
    old = ab[ab.variant.str.startswith("0")].iloc[0]
    c90 = cov[(cov.dimension == "overall") & (cov.nominal == 0.9)].iloc[0]
    c50 = cov[(cov.dimension == "overall") & (cov.nominal == 0.5)].iloc[0]
    stats = {
        "old_median": round(float(old.all100_median), 0), "new_median": round(float(full.all100_median), 0),
        "old_p90": round(float(old.all100_p90), 0), "new_p90": round(float(full.all100_p90), 0),
        "within100_old": 0.09, "within100_new": round(float(full.within100), 2),
        "hp_old_median": round(float(hp.err_old.median()), 0), "hp_new_median": round(float(hp.err_model.median()), 0),
        "cov90": round(float(c90.coverage), 2), "cov90_lo": round(float(c90.ci95_low), 2), "cov90_hi": round(float(c90.ci95_high), 2),
        "cov50": round(float(c50.coverage), 2),
        "tiers": pred.confidence_tier.value_counts().to_dict(),
        "tier_quality": {k: {"n": int(v.n), "w100": round(float(v.within_100), 2), "med": round(float(v.median_err), 0)} for k, v in tq.iterrows()},
        "curve": [{"n": int(r.visited_addresses_known), "med": round(float(r.median_error_m), 1), "w100": round(float(r.within_100m), 3)} for r in lc.itertuples()],
        "old_curve_median": round(float(lc.old_geocoder_median_m.iloc[0]), 1),
        "impact": {r.metric: {"old": round(float(r.old), 3), "new": round(float(r.new), 3), "rel": round(float(r.relative_change), 3),
                              "lo": round(float(r.ci95_low), 3), "hi": round(float(r.ci95_high), 3)} for r in imp.itertuples()},
    }
    data = {
        "towns": towns.to_dict("records"),
        "localities": loc[["locality_id", "town_id", "locality_name", "centroid_x", "centroid_y"]].round(1).to_dict("records"),
        "landmarks": lm[["poi_id", "town_id", "landmark_type", "name", "x", "y"]].round(1).to_dict("records"),
        "addresses": rows, "stats": stats,
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    out = OUT / "demo_map.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Learning Geocoder Demo</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2430;--mute:#6b7585;--line:#e2e6ec;--hi:#2f7d5b;--me:#d19a2a;--lo:#c0443a;--old:#8a8f98;--acc:#2b5fa5}
@media (prefers-color-scheme:dark){:root{--bg:#12161c;--card:#1b212a;--ink:#e8ecf2;--mute:#97a1b0;--line:#2b3340;--acc:#6aa0e8}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:10px 16px;border-bottom:1px solid var(--line);background:var(--card);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header h1{font-size:16px;margin:0;font-weight:650}header .sub{color:var(--mute);font-size:12.5px}
.tabs button{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:5px 12px;border-radius:7px;cursor:pointer;margin-right:4px}
.tabs button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
main{display:grid;grid-template-columns:1fr 380px;height:calc(100vh - 54px)}
@media (max-width:900px){main{grid-template-columns:1fr;height:auto}#map{height:60vh}}
#mapwrap{position:relative;min-width:0}canvas{display:block;width:100%;height:100%;background:var(--card);cursor:grab}
.tools{position:absolute;left:10px;top:10px;background:var(--card);border:1px solid var(--line);border-radius:9px;padding:8px 10px;font-size:12.5px;max-width:330px}
.tools label{display:inline-flex;align-items:center;gap:4px;margin:2px 8px 2px 0;white-space:nowrap}
.tools input[type=text]{width:130px;padding:3px 6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
.legend{position:absolute;left:10px;bottom:10px;background:var(--card);border:1px solid var(--line);border-radius:9px;padding:7px 10px;font-size:12px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
aside{background:var(--bg);border-left:1px solid var(--line);overflow:auto;padding:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:11px 13px;margin-bottom:10px}
.card h3{margin:0 0 6px;font-size:13px;color:var(--mute);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px}.kv span:nth-child(odd){color:var(--mute)}
.pill{display:inline-block;padding:1px 9px;border-radius:99px;color:#fff;font-size:12px;font-weight:600}
.big{font-size:25px;font-weight:700;line-height:1.1}.row{display:flex;gap:10px}.row>div{flex:1}
.dir{background:var(--bg);border-radius:8px;padding:8px 10px;margin:6px 0}
.mute{color:var(--mute)}.small{font-size:12px}
.sidetabs button{border:0;background:none;color:var(--mute);padding:5px 8px;cursor:pointer;font-weight:600}.sidetabs button.on{color:var(--acc);border-bottom:2px solid var(--acc)}
table{border-collapse:collapse;width:100%}td,th{padding:3px 6px;text-align:right;border-bottom:1px solid var(--line);font-size:12.5px}th:first-child,td:first-child{text-align:left}
.tip{position:fixed;pointer-events:none;background:#111;color:#fff;padding:4px 8px;border-radius:6px;font-size:12px;display:none;z-index:9;max-width:300px}
</style></head><body>
<header><h1>Learning Geocoder &middot; demo</h1><span class="sub">synthetic data &middot; works offline &middot; no map tiles needed</span>
<span class="tabs" id="towntabs"></span></header>
<main><div id="mapwrap"><canvas id="map"></canvas>
<div class="tools">
<div><label><input type="checkbox" id="lOld" checked>old pin</label><label><input type="checkbox" id="lNew" checked>our pin</label>
<label><input type="checkbox" id="lCirc">circles (all)</label><label><input type="checkbox" id="lLm" checked>landmarks</label>
<label><input type="checkbox" id="lTruth" checked>surveyed truth</label></div>
<div style="margin-top:4px">Show: <label><input type="checkbox" class="tf" value="h" checked>high</label><label><input type="checkbox" class="tf" value="m" checked>medium</label><label><input type="checkbox" class="tf" value="l" checked>low</label>
<label><input type="checkbox" id="fSurv">surveyed only</label></div>
<div style="margin-top:4px"><input type="text" id="q" placeholder="address id e.g. AD000123"> <button id="go">find</button></div></div>
<div class="legend"><span class="dot" style="background:var(--hi)"></span>high (visit now) <span class="dot" style="background:var(--me);margin-left:8px"></span>medium <span class="dot" style="background:var(--lo);margin-left:8px"></span>low (verify first)<br>
<span class="dot" style="background:var(--old)"></span>old geocoder pin &nbsp; &#9670; surveyed truth &nbsp; &#9633; landmark &nbsp; scroll = zoom, drag = pan</div>
</div>
<aside><div class="sidetabs"><button data-t="addr" class="on">Address</button><button data-t="score">Scorecard</button><button data-t="learn">How it learns</button></div>
<div id="panel"></div></aside></main><div class="tip" id="tip"></div>
<script>
const D=__DATA__;
const TC={h:'#2f7d5b',m:'#d19a2a',l:'#c0443a'},TN={h:'high',m:'medium',l:'low'};
const cv=document.getElementById('map'),cx=cv.getContext('2d'),tip=document.getElementById('tip');
let town=D.towns[0].town_id,sel=null,hover=null,V={x:0,y:0,s:0.1},W=0,H=0,DPR=window.devicePixelRatio||1,tab='addr';
const byTown={};D.addresses.forEach(a=>{(byTown[a.t]=byTown[a.t]||[]).push(a)});const byId={};D.addresses.forEach(a=>byId[a.id]=a);
const $=id=>document.getElementById(id);
function fit(){const A=byTown[town];let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;A.forEach(a=>{[[a.x,a.y],[a.ox,a.oy]].forEach(p=>{x0=Math.min(x0,p[0]);x1=Math.max(x1,p[0]);y0=Math.min(y0,p[1]);y1=Math.max(y1,p[1])})});
V.x=(x0+x1)/2;V.y=(y0+y1)/2;V.s=Math.min(W/(x1-x0+600),H/(y1-y0+600));}
function resize(){const r=cv.getBoundingClientRect();W=r.width;H=r.height;cv.width=W*DPR;cv.height=H*DPR;cx.setTransform(DPR,0,0,DPR,0,0);draw()}
const sx=x=>(x-V.x)*V.s+W/2,sy=y=>H/2-(y-V.y)*V.s;
function visible(a){const f=[...document.querySelectorAll('.tf')].filter(c=>c.checked).map(c=>c.value);if(!f.includes(a.tier))return false;if($('fSurv').checked&&a.tx===undefined)return false;return true}
function circle(x,y,r,fill,stroke,w){cx.beginPath();cx.arc(sx(x),sy(y),Math.max(r*V.s,1),0,6.2832);if(fill){cx.fillStyle=fill;cx.fill()}if(stroke){cx.strokeStyle=stroke;cx.lineWidth=w||1;cx.stroke()}}
function draw(){cx.clearRect(0,0,W,H);
 D.localities.filter(l=>l.town_id===town).forEach(l=>{cx.fillStyle='rgba(120,130,150,.55)';cx.font='11px system-ui';cx.textAlign='center';cx.fillText(l.locality_name,sx(l.centroid_x),sy(l.centroid_y))});
 if($('lLm').checked){D.landmarks.filter(l=>l.town_id===town).forEach(l=>{cx.fillStyle='#7a6bb0';cx.fillRect(sx(l.x)-3,sy(l.y)-3,6,6);if(V.s>0.22){cx.fillStyle='rgba(122,107,176,.9)';cx.font='10px system-ui';cx.textAlign='left';cx.fillText(l.name,sx(l.x)+6,sy(l.y)+3)}})}
 const A=byTown[town].filter(visible);
 if($('lCirc').checked&&V.s>0.12)A.forEach(a=>{if(sx(a.x)<-50||sx(a.x)>W+50||sy(a.y)<-50||sy(a.y)>H+50)return;circle(a.x,a.y,a.r9,TC[a.tier]+'12',TC[a.tier]+'55',1)});
 if($('lOld').checked)A.forEach(a=>{cx.fillStyle='rgba(138,143,152,.45)';cx.fillRect(sx(a.ox)-1.5,sy(a.oy)-1.5,3,3)});
 if($('lNew').checked)A.forEach(a=>{cx.fillStyle=TC[a.tier];cx.beginPath();cx.arc(sx(a.x),sy(a.y),a.v?2.6:2.2,0,6.2832);cx.fill()});
 if($('lTruth').checked)A.forEach(a=>{if(a.tx===undefined)return;const X=sx(a.tx),Y=sy(a.ty);cx.strokeStyle='rgba(0,0,0,.55)';cx.lineWidth=1;cx.beginPath();cx.moveTo(sx(a.ox),sy(a.oy));cx.lineTo(X,Y);cx.stroke();cx.beginPath();cx.moveTo(X,Y-5);cx.lineTo(X+5,Y);cx.lineTo(X,Y+5);cx.lineTo(X-5,Y);cx.closePath();cx.fillStyle='#111';cx.fill()});
 if(sel&&sel.t===town){const a=sel;circle(a.x,a.y,a.r9,TC[a.tier]+'22',TC[a.tier],2);circle(a.x,a.y,a.r5,TC[a.tier]+'33',TC[a.tier],1.5);
  cx.strokeStyle='#888';cx.setLineDash([5,4]);cx.beginPath();cx.moveTo(sx(a.ox),sy(a.oy));cx.lineTo(sx(a.x),sy(a.y));cx.stroke();cx.setLineDash([]);
  cx.fillStyle='#fff';cx.strokeStyle='#8a8f98';cx.lineWidth=2.5;cx.beginPath();cx.arc(sx(a.ox),sy(a.oy),6,0,6.2832);cx.fill();cx.stroke();
  cx.fillStyle=TC[a.tier];cx.strokeStyle='#fff';cx.lineWidth=2;cx.beginPath();cx.arc(sx(a.x),sy(a.y),7,0,6.2832);cx.fill();cx.stroke();
  if(a.tx!==undefined){const X=sx(a.tx),Y=sy(a.ty);cx.fillStyle='#111';cx.beginPath();cx.moveTo(X,Y-9);cx.lineTo(X+9,Y);cx.lineTo(X,Y+9);cx.lineTo(X-9,Y);cx.closePath();cx.fill();cx.strokeStyle='#fff';cx.lineWidth=1.5;cx.stroke()}}}
function nearest(mx,my){let b=null,bd=100;byTown[town].filter(visible).forEach(a=>{const d=(sx(a.x)-mx)**2+(sy(a.y)-my)**2;if(d<bd){bd=d;b=a}});return b}
let drag=null;cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,vx:V.x,vy:V.y,moved:false};cv.style.cursor='grabbing'});
window.addEventListener('mouseup',e=>{cv.style.cursor='grab';if(drag&&!drag.moved){const r=cv.getBoundingClientRect(),a=nearest(e.clientX-r.left,e.clientY-r.top);if(a)select(a)}drag=null});
window.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
 if(drag){const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>3)drag.moved=true;V.x=drag.vx-dx/V.s;V.y=drag.vy+dy/V.s;draw();return}
 if(mx<0||my<0||mx>W||my>H){tip.style.display='none';return}const a=nearest(mx,my);if(a){tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';tip.textContent=a.id+' · '+TN[a.tier]+' · ±'+Math.round(a.r9)+' m'}else tip.style.display='none'});
cv.addEventListener('wheel',e=>{e.preventDefault();const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,wx=(mx-W/2)/V.s+V.x,wy=V.y-(my-H/2)/V.s,k=Math.exp(-e.deltaY*0.0015);V.s=Math.min(Math.max(V.s*k,0.02),6);V.x=wx-(mx-W/2)/V.s;V.y=wy+(my-H/2)/V.s;draw()},{passive:false});
function select(a){sel=a;if(a.t!==town){setTown(a.t,true)}tab='addr';tabs();if(V.s<0.5){V.x=a.x;V.y=a.y;V.s=1.2}panel();draw()}
function dist(a,b,c,d){return Math.hypot(a-c,b-d)}
function panel(){const P=$('panel');
 if(tab==='addr'){if(!sel){P.innerHTML='<div class="card"><h3>Pick an address</h3>Click any dot, or search an id. Surveyed addresses (&#9670;) show the true location so you can see the old vs new error.</div>';return}
  const a=sel;let h=`<div class="card"><h3>${a.id} &middot; ${a.t}</h3><div class="mute small">${a.txt}</div><div style="margin:8px 0"><span class="pill" style="background:${TC[a.tier]}">${TN[a.tier]} confidence</span> <b>${a.act.replace(/_/g,' ')}</b></div>
  <div class="kv"><span>our pin</span><span>${a.x.toFixed(0)}, ${a.y.toFixed(0)}</span><span>90% radius</span><span>${Math.round(a.r9)} m</span><span>50% radius</span><span>${Math.round(a.r5)} m</span><span>P(within 100 m)</span><span>${Math.round(a.c*100)}%</span><span>moved from old pin</span><span>${Math.round(dist(a.x,a.y,a.ox,a.oy))} m</span><span>based on</span><span>${a.m.replace(/_/g,' ')}${a.v?' (own field visit)':''}</span><span>why</span><span class="small">${a.why.replace(/\|/g,', ').replace(/_/g,' ').toLowerCase()}</span></div></div>`;
  if(a.tx!==undefined){const imp=a.eo/Math.max(a.en,1);h+=`<div class="card"><h3>Against the surveyed truth</h3><div class="row"><div><div class="mute small">old geocoder</div><div class="big" style="color:var(--old)">${Math.round(a.eo)} m</div></div><div><div class="mute small">our pin</div><div class="big" style="color:${TC[a.tier]}">${Math.round(a.en)} m</div></div></div><div class="small mute" style="margin-top:6px">${a.en<=a.r9?'Truth is inside the 90% circle.':'Truth is outside the 90% circle (expected for ~10%).'}</div></div>`}
  h+=`<div class="card"><h3>Directions for the agent (works offline)</h3><div class="dir">${a.dir}</div>${a.dl?`<div class="dir small"><span class="mute">local flavour</span><br>${a.dl}</div>`:''}</div>`;
  if(a.hf)h+=`<div class="card"><h3>Hand-off to Problem Statement 2</h3>Flagged <b>hard to find</b>: a failed visit here must not be read as "address does not exist". Location confidence ${Math.round(a.c*100)}%.</div>`;
  P.innerHTML=h}
 else if(tab==='score'){const S=D.stats;P.innerHTML=`<div class="card"><h3>Distance error vs 100 surveyed addresses</h3><div class="row"><div><div class="mute small">median, old</div><div class="big" style="color:var(--old)">${S.old_median} m</div></div><div><div class="mute small">median, ours</div><div class="big" style="color:var(--acc)">${S.new_median} m</div></div></div><div class="small mute" style="margin-top:6px">90th percentile ${S.old_p90} m &rarr; ${S.new_p90} m &middot; within 100 m: ${Math.round(S.within100_old*100)}% &rarr; ${Math.round(S.within100_new*100)}% (out-of-fold)</div></div>
  <div class="card"><h3>Addresses with no visit (1,231 hide-and-predict)</h3>Median error <b>${S.hp_old_median} m &rarr; ${S.hp_new_median} m</b>. This is the realistic number for addresses nobody has visited yet.</div>
  <div class="card"><h3>Is the confidence honest?</h3>The 90% circle contained the true location <b>${Math.round(S.cov90*100)}%</b> of the time (95% CI ${Math.round(S.cov90_lo*100)}&ndash;${Math.round(S.cov90_hi*100)}%); the 50% circle ${Math.round(S.cov50*100)}%. Calibrated by conformal prediction on data the model never trained on.</div>
  <div class="card"><h3>What the tiers mean (surveyed)</h3><table><tr><th>tier</th><th>addresses</th><th>within 100 m</th><th>median err</th></tr>${['high','medium','low'].map(t=>`<tr><td><span class="dot" style="background:${TC[t[0]]}"></span>${t}</td><td>${S.tiers[t]||0}</td><td>${S.tier_quality[t]?Math.round(S.tier_quality[t].w100*100)+'%':'-'}</td><td>${S.tier_quality[t]?S.tier_quality[t].med+' m':'-'}</td></tr>`).join('')}</table><div class="small mute">Counts are all 2,880 addresses; accuracy columns use the 100 surveyed.</div></div>
  <div class="card"><h3>Indicative field impact*</h3>${Object.entries(S.impact).map(([k,v])=>`<div class="small">${k}: <b>${v.rel>0?'+':''}${(v.rel*100).toFixed(0)}%</b> <span class="mute">(${(v.lo*100).toFixed(0)} to ${(v.hi*100).toFixed(0)}%)</span></div>`).join('')}<div class="small mute" style="margin-top:4px">*observational mapping on synthetic visit logs, not a pilot result.</div></div>`}
 else{const S=D.stats,c=S.curve,mx=Math.max(S.old_curve_median,c[0].med)*1.1,w=330,h=150;const px=i=>30+i/(c.length-1)*(w-40),py=v=>h-20-v/mx*(h-30);
  P.innerHTML=`<div class="card"><h3>Every confirmed visit improves the next guess</h3><div class="small mute">Median error for addresses with no visit of their own, as the system learns from more visited addresses (5 random splits).</div>
  <svg viewBox="0 0 ${w} ${h}" width="100%"><line x1="30" y1="${py(S.old_curve_median)}" x2="${w-10}" y2="${py(S.old_curve_median)}" stroke="#8a8f98" stroke-dasharray="4 3"/><text x="${w-12}" y="${py(S.old_curve_median)-4}" text-anchor="end" font-size="9" fill="#8a8f98">old geocoder ${Math.round(S.old_curve_median)} m</text>
  <polyline fill="none" stroke="#2b5fa5" stroke-width="2.5" points="${c.map((p,i)=>px(i)+','+py(p.med)).join(' ')}"/>${c.map((p,i)=>`<circle cx="${px(i)}" cy="${py(p.med)}" r="3" fill="#2b5fa5"/><text x="${px(i)}" y="${py(p.med)-7}" text-anchor="middle" font-size="9" fill="currentColor">${Math.round(p.med)}</text><text x="${px(i)}" y="${h-6}" text-anchor="middle" font-size="9" fill="#8a8f98">${p.n}</text>`).join('')}
  <text x="${w/2}" y="${h}" text-anchor="middle" font-size="9" fill="#8a8f98"></text></svg><div class="small mute">x axis: visited addresses learned from. Within 100 m: ${Math.round(c[0].w100*100)}% &rarr; ${Math.round(c[c.length-1].w100*100)}%.</div></div>
  <div class="card"><h3>How it works, in one paragraph</h3>Each clue about an address (its own visit, neighbours on the same street, the street-number grid, a landmark, the old pin) gives a guess and an honest uncertainty. The model combines them by trust, drops clues that contradict the rest, then turns its uncertainty into calibrated circles. A new visit is just one more clue, so it helps every address on that street at once.</div>`}}
function tabs(){document.querySelectorAll('.sidetabs button').forEach(b=>b.classList.toggle('on',b.dataset.t===tab))}
document.querySelectorAll('.sidetabs button').forEach(b=>b.onclick=()=>{tab=b.dataset.t;tabs();panel()});
function setTown(t,keep){town=t;document.querySelectorAll('#towntabs button').forEach(b=>b.classList.toggle('on',b.dataset.t===t));if(!keep){sel=null}fit();draw();panel()}
D.towns.forEach(t=>{const b=document.createElement('button');b.dataset.t=t.town_id;b.textContent=t.town_name+' ('+t.town_id+')';b.onclick=()=>setTown(t.town_id);$('towntabs').appendChild(b)});
document.querySelectorAll('.tools input').forEach(i=>i.addEventListener('change',draw));
$('go').onclick=()=>{const a=byId[$('q').value.trim().toUpperCase()];if(a)select(a);else alert('not found')};$('q').addEventListener('keydown',e=>{if(e.key==='Enter')$('go').click()});
window.addEventListener('resize',resize);resize();setTown(town);
const first=byTown[town].find(a=>a.tx!==undefined&&a.tier==='m')||byTown[town].find(a=>a.tx!==undefined);if(first)select(first);
</script></body></html>
"""


# ======================================================================
# figures
# ======================================================================
TC = {"high": "#2f7d5b", "medium": "#d19a2a", "low": "#c0443a"}


def architecture():
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.axis("off"); ax.set_xlim(0, 110); ax.set_ylim(0, 43)

    def box(x, y, w, h, text, fc, ec="#33415c", fs=8.3, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#33415c", lw=1.2))

    box(1, 28, 22, 12, "Inputs from CN\naddress text · old pin\nvisit GPS trails · outcomes\nagent remarks · landmarks", "#eef3fb")
    box(28, 28, 24, 12, "Visit cleaning\none trusted point per visit\nrole · uncertainty · integrity\n(fake-visit safeguards)", "#e8f4ee")
    box(28, 6, 24, 12, "Address parser\nlocality · street numbers\nlandmark type (EN/HI/KN)", "#e8f4ee")
    box(58, 17, 25, 14, "Pin model\nevidence fusion: own visit ·\nstreet neighbours · street-number\ngrid · landmarks · old pin", "#fdf1dc", bold=False)
    box(88, 28, 21, 12, "Calibration\nconformal 50/90% radii\nconfidence tier · directions", "#f6e7ea")
    box(88, 6, 21, 12, "Outputs\nfield app (offline) · visit planner\naddress records · PS2 · dashboards", "#eef3fb")
    arrow(23, 34, 28, 34); arrow(52, 34, 58, 27); arrow(40, 18, 58, 22); arrow(83, 26, 88, 32); arrow(98.5, 28, 98.5, 18)
    arrow(23, 31, 28, 14)
    ax.annotate("", xy=(70, 17), xytext=(98, 6.5), arrowprops=dict(arrowstyle="-|>", color="#c26a3d", lw=1.6, ls="--", connectionstyle="arc3,rad=-0.25"))
    ax.text(79, 3.0, "every confirmed visit becomes new evidence", color="#c26a3d", fontsize=8.5, ha="center", style="italic")
    plt.tight_layout(); plt.savefig(RES / "fig_architecture.png", dpi=160); plt.close()


def example_map():
    """Six typical surveyed addresses (the 40th and 60th error percentile inside each tier - not cherry-picked)."""
    p = pd.read_csv(OUT / "predictions.csv").set_index("address_id")
    sv = pd.read_csv(DATA / "surveyed_addresses.csv").set_index("address_id")
    lm = pd.read_csv(DATA / "landmarks_poi.csv")
    d = pd.read_csv(RES / "calibration_surveyed_detail.csv").set_index("address_id")
    picks = []
    for tier in ("high", "medium", "low"):
        g = d[p.loc[d.index, "confidence_tier"] == tier].sort_values("error_m")
        for q in (0.4, 0.6):
            if len(g):
                picks.append(g.index[int(q * (len(g) - 1))])
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    for ax, a in zip(axes.T.ravel(), picks):
        r = p.loc[a]; t = sv.loc[a].to_numpy(float)
        col = TC[r.confidence_tier]
        ext = max(1.25 * r.radius_90, 1.15 * np.hypot(r.old_pin_x - t[0], r.old_pin_y - t[1]), 1.15 * np.hypot(r.pin_x - t[0], r.pin_y - t[1]), 120)
        cx, cy = (r.pin_x + t[0]) / 2, (r.pin_y + t[1]) / 2
        ax.add_patch(Circle((r.pin_x, r.pin_y), r.radius_90, fc=col, alpha=.12, ec=col, lw=1.2))
        ax.add_patch(Circle((r.pin_x, r.pin_y), r.radius_50, fc=col, alpha=.22, ec=col, lw=1.2))
        ax.plot([r.old_pin_x, r.pin_x], [r.old_pin_y, r.pin_y], color="#8a8f98", lw=1, ls=":")
        ax.scatter(r.old_pin_x, r.old_pin_y, s=70, facecolor="white", edgecolor="#8a8f98", lw=2, zorder=3)
        ax.scatter(r.pin_x, r.pin_y, s=70, color=col, edgecolor="white", lw=1.2, zorder=4)
        ax.scatter(*t, s=80, marker="D", color="black", zorder=5)
        L = lm[(lm.town_id == r.town_id)]
        L = L[np.hypot(L.x - cx, L.y - cy) < ext * 1.5]
        ax.scatter(L.x, L.y, marker="s", s=28, color="#7a6bb0", zorder=2)
        for q in L.itertuples():
            ax.annotate(q.name, (q.x, q.y), xytext=(4, 3), textcoords="offset points", fontsize=7, color="#5b4d8f")
        ax.set_xlim(cx - ext * 1.4, cx + ext * 1.4); ax.set_ylim(cy - ext * 1.4, cy + ext * 1.4); ax.set_aspect("equal")
        eo = np.hypot(r.old_pin_x - t[0], r.old_pin_y - t[1]); en = np.hypot(r.pin_x - t[0], r.pin_y - t[1])
        ax.set_title(f"{a} - {r.confidence_tier}\nold {eo:.0f} m  ->  ours {en:.0f} m  (90% radius {r.radius_90:.0f} m)", fontsize=8.5)
        ax.tick_params(labelsize=7)
        sb = 10 ** np.floor(np.log10(ext * .8)); x0 = cx - ext * 1.25; y0 = cy - ext * 1.25
        ax.plot([x0, x0 + sb], [y0, y0], color="k", lw=2); ax.text(x0, y0 + ext * .05, f"{sb:.0f} m", fontsize=7)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="#8a8f98", mew=2, label="old geocoder pin"),
                        Line2D([], [], marker="o", ls="", color="#555555", label="our pin (colour = confidence tier)"),
                        Line2D([], [], marker="D", ls="", color="black", label="surveyed true location"),
                        Line2D([], [], marker="s", ls="", color="#7a6bb0", label="landmark")], loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(RES / "fig_example_map.png", dpi=150); plt.close()



# ======================================================================
# summary
# ======================================================================
def summary():
    ab = pd.read_csv(RES / "pin_model_ablation.csv")
    sc = pd.read_csv(RES / "pin_model_scores.csv")
    base = pd.read_csv(RES / "baseline_scores.csv")
    cov = pd.read_csv(RES / "calibration_coverage.csv")
    tq = pd.read_csv(RES / "confidence_tier_quality.csv").set_index("tier")
    hp = pd.read_csv(RES / "hide_and_predict.csv")
    pred = pd.read_csv(OUT / "predictions.csv")
    lc = pd.read_csv(RES / "learning_curve.csv")
    imp = pd.read_csv(RES / "impact_estimate.csv")
    far = pd.read_csv(RES / "impact_far_addresses.csv").iloc[0]
    dchk = pd.read_csv(RES / "directions_checks.csv").iloc[0]
    sv = pd.read_csv(DATA / "surveyed_addresses.csv").set_index("address_id")
    p = pred.set_index("address_id")

    full = ab[ab.variant.str.startswith("5")].iloc[0]
    old = ab[ab.variant.str.startswith("0")].iloc[0]
    row = lambda df, dim, seg: df[(df.dimension == dim) & (df.segment == seg)].iloc[0]
    o_all = row(sc, "overall", "all")
    oo = row(base, "overall", "all")
    vis, unv = row(sc, "visit_status", "visited"), row(sc, "visit_status", "unvisited")
    ov, uv = row(base, "visit_status", "visited"), row(base, "visit_status", "unvisited")
    c90 = cov[(cov.dimension == "overall") & (cov.nominal == 0.9)].iloc[0]
    c50 = cov[(cov.dimension == "overall") & (cov.nominal == 0.5)].iloc[0]

    eo = np.hypot(p.loc[sv.index, "old_pin_x"] - sv.surveyed_x, p.loc[sv.index, "old_pin_y"] - sv.surveyed_y)
    en = np.hypot(p.loc[sv.index, "pin_x"] - sv.surveyed_x, p.loc[sv.index, "pin_y"] - sv.surveyed_y)
    tiers = pred.confidence_tier.value_counts().to_dict()
    S = {
        "surveyed": {
            "n": 100, "old_median": float(oo.median_error_m), "old_p90": float(oo.p90_error_m),
            "new_median": float(o_all.median_error_m), "new_p90": float(o_all.p90_error_m),
            "new_median_ci": [float(o_all.median_ci95_low_m), float(o_all.median_ci95_high_m)],
            "old_within": [float(oo[f"share_within_{r}m"]) for r in (50, 100, 250)],
            "new_within": [float(o_all[f"share_within_{r}m"]) for r in (50, 100, 250)],
            "visited": {"n": int(vis.n), "old_median": float(ov.median_error_m), "new_median": float(vis.median_error_m)},
            "unvisited": {"n": int(unv.n), "old_median": float(uv.median_error_m), "new_median": float(unv.median_error_m),
                          "old_p90": float(uv.p90_error_m), "new_p90": float(unv.p90_error_m)},
            "dev85": {"old_median": float(ab.iloc[0].dev85_median), "new_median": float(full.dev85_median),
                      "old_p90": float(ab.iloc[0].dev85_p90), "new_p90": float(full.dev85_p90)},
            "test15": {"old_median": float(old.test15_median), "new_median": float(full.test15_median),
                       "old_p90": float(old.test15_p90), "new_p90": float(full.test15_p90)},
            "share_worse_than_old_by_25m": float((en > eo + 25).mean()),
            "share_better_than_old_by_25m": float((en < eo - 25).mean()),
        },
        "ablation": [{"variant": r.variant, "median": float(r.all100_median), "p90": float(r.all100_p90)} for r in ab.itertuples()],
        "hide_and_predict": {
            "n": int(len(hp)), "old_median": float(hp.err_old.median()), "new_median": float(hp.err_model.median()),
            "old_p90": float(hp.err_old.quantile(.9)), "new_p90": float(hp.err_model.quantile(.9)),
            "old_within100": float((hp.err_old <= 100).mean()), "new_within100": float((hp.err_model <= 100).mean()),
            "share_worse_than_old_by_25m": float((hp.err_model > hp.err_old + 25).mean()),
            "by_precision": {k: {"n": int(len(g)), "old": float(g.err_old.median()), "new": float(g.err_model.median())}
                             for k, g in hp.groupby("precision")},
        },
        "calibration": {"cov90": float(c90.coverage), "cov90_ci": [float(c90.ci95_low), float(c90.ci95_high)], "cov50": float(c50.coverage),
                        "cov50_ci": [float(c50.ci95_low), float(c50.ci95_high)]},
        "tiers": {"counts": tiers, "share_low": tiers.get("low", 0) / len(pred),
                  "quality": {k: {"n": int(v.n), "median_err": float(v.median_err), "within_100": float(v.within_100), "within_250": float(v.within_250)} for k, v in tq.iterrows()}},
        "learning_curve": [{"visited_known": int(r.visited_addresses_known), "median": float(r.median_error_m), "within100": float(r.within_100m)} for r in lc.itertuples()],
        "impact": {r.metric: {"old": float(r.old), "new": float(r.new), "relative": float(r.relative_change), "ci": [float(r.ci95_low), float(r.ci95_high)]} for r in imp.itertuples()},
        "far_old_pin": {"n": int(far.far_old_pin_addresses), "share_within_200m": float(far.share_brought_within_200m)},
        "directions": {k: (float(v) if not isinstance(v, str) else v) for k, v in dchk.items()},
        "coverage_of_evidence": {"addresses_with_visit_pin": int(p.has_own_visit.sum()), "total": int(len(p))},
    }
    (RES / "summary_metrics.json").write_text(json.dumps(S, indent=2), encoding="utf-8")
    s = S["surveyed"]
    md = f"""# Headline numbers (generated by make_summary.py)

| | old geocoder | ours |
|---|---:|---:|
| 100 surveyed, median / P90 (m) | {s['old_median']:.0f} / {s['old_p90']:.0f} | **{s['new_median']:.0f} / {s['new_p90']:.0f}** |
| within 50 / 100 / 250 m | {s['old_within'][0]:.0%} / {s['old_within'][1]:.0%} / {s['old_within'][2]:.0%} | {s['new_within'][0]:.0%} / {s['new_within'][1]:.0%} / {s['new_within'][2]:.0%} |
| visited ({s['visited']['n']}) median | {s['visited']['old_median']:.0f} | {s['visited']['new_median']:.0f} |
| unvisited ({s['unvisited']['n']}) median / P90 | {s['unvisited']['old_median']:.0f} / {s['unvisited']['old_p90']:.0f} | {s['unvisited']['new_median']:.0f} / {s['unvisited']['new_p90']:.0f} |
| hide-and-predict ({S['hide_and_predict']['n']}) median / P90 | {S['hide_and_predict']['old_median']:.0f} / {S['hide_and_predict']['old_p90']:.0f} | {S['hide_and_predict']['new_median']:.0f} / {S['hide_and_predict']['new_p90']:.0f} |

Calibration: 90% radius covers {S['calibration']['cov90']:.0%} (95% CI {S['calibration']['cov90_ci'][0]:.0%}-{S['calibration']['cov90_ci'][1]:.0%}); 50% radius covers {S['calibration']['cov50']:.0%}.
Tiers: {tiers}. Surveyed pins worse than the old pin by >25 m: {s['share_worse_than_old_by_25m']:.0%}; better by >25 m: {s['share_better_than_old_by_25m']:.0%}.
"""
    (OUT / "summary.md").write_text(md, encoding="utf-8")
    print(md)


def main():
    architecture()
    example_map()
    summary()
    demo_map()


if __name__ == "__main__":
    main()
