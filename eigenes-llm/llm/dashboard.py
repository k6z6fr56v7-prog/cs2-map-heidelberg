"""Live-Dashboard: Trainingsfortschritt im Browser (auch vom Handy aus).

Start (in einem zweiten Fenster, waehrend das Training laeuft):
    python -m llm dashboard              -> http://localhost:8900
    python -m llm dashboard --netzwerk   -> auch vom Handy im selben WLAN erreichbar

Liest nur die Dateien, die das Training ohnehin schreibt (status.json und die
Log-CSV im Modellordner). Das Training wird dadurch nicht langsamer.
"""

from __future__ import annotations

import csv
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def find_runs(work: Path) -> list[dict]:
    runs = []
    for status_path in work.glob("*/status.json"):
        try:
            st = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        st["ordner"] = status_path.parent.name
        runs.append(st)
    runs.sort(key=lambda r: r.get("aktualisiert", 0), reverse=True)
    return runs


def read_log(path: Path, max_points: int = 800) -> dict:
    train, val = [], []
    try:
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    step = int(row["schritt"])
                except (KeyError, ValueError):
                    continue
                if row.get("loss"):
                    train.append([step, float(row["loss"]), int(row.get("tokens") or 0), int(row.get("sekunden") or 0)])
                if row.get("val_loss"):
                    val.append([step, float(row["val_loss"])])
    except OSError:
        pass
    if len(train) > max_points:  # ausduennen, aber den letzten Punkt behalten
        k = len(train) / max_points
        train = [train[int(i * k)] for i in range(max_points - 1)] + [train[-1]]
    return {"train": train, "val": val}


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # sendet nichts, ermittelt nur die eigene Adresse
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


HTML = r"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training live</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--good:#006300;--warn:#a35f00}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--good:#0ca30c;--warn:#e0a030}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--good:#0ca30c;--warn:#e0a030}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 40px}
header{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:6px 16px;margin-bottom:16px}
h1{font-size:20px;margin:0}
.sub{color:var(--ink2);font-size:13px}
.live{display:inline-flex;align-items:center;gap:6px;color:var(--ink2);font-size:13px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--good)}
.dot.stale{background:var(--warn)}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px;margin-bottom:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:12px 14px}
.tile .k{color:var(--ink2);font-size:12px}
.tile .v{font-size:24px;font-weight:600;margin-top:2px}
.tile .n{color:var(--muted);font-size:12px}
.bar{height:8px;border-radius:4px;background:var(--grid);overflow:hidden;margin-top:10px}
.bar>div{height:100%;background:var(--s1);border-radius:4px;transition:width .6s}
h2{font-size:15px;margin:0 0 10px}
.legend{display:flex;gap:16px;font-size:13px;color:var(--ink2);margin-bottom:6px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:14px;height:3px;border-radius:2px}
.sw.d{width:9px;height:9px;border-radius:50%}
#chart{position:relative}
svg{display:block;width:100%;height:auto;overflow:visible}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--ring);border-radius:8px;
padding:8px 10px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.12);white-space:nowrap;display:none}
.tip b{font-weight:600}
.sample{font-size:14px;color:var(--ink);border-left:3px solid var(--grid);padding:4px 0 4px 10px;margin:8px 0}
.note{color:var(--ink2);font-size:13px}
.warnbox{border-color:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
td,th{text-align:right;padding:4px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:500}
details summary{cursor:pointer;color:var(--ink2);font-size:13px}
.res{display:grid;grid-template-columns:auto 1fr auto;gap:6px 10px;align-items:center;font-size:13px}
.res .track{height:8px;background:var(--grid);border-radius:4px;position:relative}
.res .fill{height:100%;background:var(--s1);border-radius:4px}
.res .chance{position:absolute;top:-3px;width:2px;height:14px;background:var(--ink2)}
select{font:inherit;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:8px;padding:4px 8px}
</style></head><body><main>
<header>
  <div><h1 id="title">Training live</h1><div class="sub" id="sub">warte auf Daten …</div></div>
  <div><select id="runsel" aria-label="Lauf wählen"></select> <span class="live"><span class="dot" id="dot"></span><span id="age"></span></span></div>
</header>
<div id="stale" class="card warnbox" style="display:none"></div>
<div class="tiles">
  <div class="tile"><div class="k">Fortschritt</div><div class="v" id="t-prog">–</div><div class="n" id="t-step"></div></div>
  <div class="tile"><div class="k">Restzeit</div><div class="v" id="t-eta">–</div><div class="n" id="t-elapsed"></div></div>
  <div class="tile"><div class="k">Loss (Training)</div><div class="v" id="t-loss">–</div><div class="n" id="t-val"></div></div>
  <div class="tile"><div class="k">Tempo</div><div class="v" id="t-tps">–</div><div class="n" id="t-mem"></div></div>
</div>
<div class="card"><div class="bar"><div id="bar" style="width:0"></div></div></div>
<div class="card">
  <h2>Loss-Verlauf <span class="note">(kleiner = besser)</span></h2>
  <div class="legend"><span><i class="sw" style="background:var(--s1)"></i>Training</span><span><i class="sw d" style="background:var(--s2)"></i>Validierung (ungesehener Text)</span></div>
  <div id="chart"><svg id="svg" viewBox="0 0 900 320" role="img" aria-label="Loss-Verlauf"></svg><div class="tip" id="tip"></div></div>
  <details style="margin-top:10px"><summary>Als Tabelle</summary><div id="table"></div></details>
</div>
<div class="card"><h2>Was das Modell gerade schreibt</h2><div id="samples" class="note">Beispiele erscheinen alle paar hundert Schritte.</div></div>
<div class="card" id="evalcard" style="display:none"><h2>Deutsch-Test</h2><div id="eval"></div></div>
</main>
<script>
const $=id=>document.getElementById(id);
let current=null, data=null, hoverIdx=null;
const fmtT=s=>{s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h} h ${String(m).padStart(2,'0')} min`:`${m} min`};
const num=(x,d=2)=>x==null?'–':x.toLocaleString('de-DE',{minimumFractionDigits:d,maximumFractionDigits:d});
async function j(u){const r=await fetch(u,{cache:'no-store'});return r.json()}
async function refresh(){
  try{
    const runs=await j('/api/runs');
    const sel=$('runsel');
    const names=runs.map(r=>r.ordner);
    if(sel.options.length!==names.length||[...sel.options].some((o,i)=>o.value!==names[i])){
      const keep=current;sel.innerHTML=names.map(n=>`<option>${n}</option>`).join('');
      if(keep&&names.includes(keep))sel.value=keep;
    }
    if(!current||!names.includes(current))current=names[0];
    if(!current){$('sub').textContent='Noch kein Training gefunden. Starte 3_trainieren.bat.';return}
    sel.value=current;
    data=await j('/api/run?ordner='+encodeURIComponent(current));
    render();
  }catch(e){$('age').textContent='Verbindung verloren';$('dot').classList.add('stale')}
}
$('runsel').onchange=e=>{current=e.target.value;refresh()};
function render(){
  const s=data.status, L=data.log;
  $('title').textContent=s.lauf==='sft'?'Chat-Training live':'Vortraining live';
  $('sub').textContent=`${s.ordner} · ${(s.params/1e6).toLocaleString('de-DE',{maximumFractionDigits:1})} Mio. Parameter`+(s.stunden?` · Zeitbudget ${s.stunden} h`:'');
  const age=Date.now()/1000-(s.aktualisiert||0);
  const stale=!s.fertig&&age>180;
  $('dot').classList.toggle('stale',stale||s.fertig);
  $('age').textContent=s.fertig?(s.unterbrochen?'unterbrochen':'fertig'):`aktualisiert vor ${age<90?Math.round(age)+' s':fmtT(age)}`;
  $('stale').style.display=stale?'block':'none';
  if(stale)$('stale').textContent=`Seit ${fmtT(age)} keine neuen Daten. Läuft das Training noch? (Fenster geschlossen, PC im Standby oder Neustart durch Windows-Update?) Fortsetzen mit 3b_weitertrainieren.bat.`;
  const p=s.fortschritt||0;
  $('t-prog').textContent=Math.round(p*100)+' %';$('bar').style.width=(p*100).toFixed(1)+'%';
  $('t-step').textContent=s.schritt!=null?`Schritt ${s.schritt.toLocaleString('de-DE')}`:'';
  $('t-eta').textContent=s.fertig?'fertig':(s.rest_s!=null?fmtT(s.rest_s):'–');
  $('t-elapsed').textContent=s.vergangen_s!=null?`${fmtT(s.vergangen_s)} vergangen`:'';
  $('t-loss').textContent=num(s.loss,3);
  $('t-val').textContent=s.val_loss!=null?`Validierung ${num(s.val_loss,3)}`:'Validierung folgt';
  $('t-tps').textContent=s.tokens_pro_s?Math.round(s.tokens_pro_s/1000).toLocaleString('de-DE')+'k':'–';
  $('t-mem').textContent='Tokens/s'+(s.vram_gb?` · ${num(s.vram_gb,1)} GB VRAM`:'');
  const sm=s.beispiele||[];
  if(sm.length)$('samples').innerHTML=`<div class="note">bei Schritt ${(s.beispiele_schritt||0).toLocaleString('de-DE')}:</div>`+sm.map(t=>`<div class="sample">${esc(t)}</div>`).join('');
  if(data.bewertung){$('evalcard').style.display='block';$('eval').innerHTML=evalHtml(data.bewertung)}else $('evalcard').style.display='none';
  drawChart(L);
  const rows=L.train.slice(-15).reverse();
  $('table').innerHTML='<table><tr><th>Schritt</th><th>Loss</th><th>Tokens</th><th>Zeit</th></tr>'+rows.map(r=>`<tr><td>${r[0].toLocaleString('de-DE')}</td><td>${num(r[1],3)}</td><td>${(r[2]/1e6).toLocaleString('de-DE',{maximumFractionDigits:1})} Mio.</td><td>${fmtT(r[3])}</td></tr>`).join('')+'</table>';
}
function esc(t){return t.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function evalHtml(b){
  const t=b.test;let h='<div class="res">';
  for(const [k,v] of Object.entries(t)){h+=`<span>${k}</span><span class="track"><span class="fill" style="width:${v.quote*100}%;display:block"></span><span class="chance" style="left:${v.zufall*100}%" title="Raten"></span></span><span>${Math.round(v.quote*100)} %</span>`}
  return h+'</div><div class="note" style="margin-top:6px">Strich = Trefferquote durch Raten'+(b.perplexitaet?` · Perplexität ${num(b.perplexitaet,1)}`:'')+'</div>';
}
function drawChart(L){
  const svg=$('svg'),W=Math.max(300,Math.round($('chart').clientWidth)),H=W<600?240:320,m={l:44,r:W<600?14:76,t:10,b:28};
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  const tr=L.train,va=L.val;
  if(tr.length<2){svg.innerHTML=`<text x="${W/2}" y="${H/2}" text-anchor="middle" fill="var(--muted)" font-size="14">Noch zu wenige Messpunkte</text>`;return}
  const x0=tr[0][0],x1=Math.max(tr[tr.length-1][0],va.length?va[va.length-1][0]:0);
  const all=tr.map(r=>r[1]).concat(va.map(r=>r[1]));
  const late=tr.slice(Math.floor(tr.length*0.05)).map(r=>r[1]).concat(va.map(r=>r[1]));
  let ymin=Math.min(...all),ymax=Math.max(...late);
  const pad=(ymax-ymin)*0.08||0.1;ymin=Math.max(0,ymin-pad);ymax=ymax+pad;
  const X=s=>m.l+(s-x0)/Math.max(x1-x0,1)*(W-m.l-m.r), Y=v=>m.t+(1-(Math.min(v,ymax)-ymin)/(ymax-ymin))*(H-m.t-m.b);
  let g='';
  const ticks=5;for(let i=0;i<=ticks;i++){const v=ymin+(ymax-ymin)*i/ticks,y=Y(v);
    g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y}" y2="${y}" stroke="var(--grid)" stroke-width="1"/><text x="${m.l-8}" y="${y+4}" text-anchor="end" font-size="11" fill="var(--muted)" style="font-variant-numeric:tabular-nums">${num(v,2)}</text>`}
  const nx=W<600?2:4;for(let i=0;i<=nx;i++){const s=x0+(x1-x0)*i/nx;g+=`<text x="${X(s)}" y="${H-8}" text-anchor="middle" font-size="11" fill="var(--muted)">${Math.round(s).toLocaleString('de-DE')}</text>`}
  g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${H-m.b}" y2="${H-m.b}" stroke="var(--axis)"/>`;
  g+=`<polyline fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${tr.map(r=>X(r[0]).toFixed(1)+','+Y(r[1]).toFixed(1)).join(' ')}"/>`;
  if(va.length>1)g+=`<polyline fill="none" stroke="var(--s2)" stroke-width="2" stroke-dasharray="1 0" points="${va.map(r=>X(r[0]).toFixed(1)+','+Y(r[1]).toFixed(1)).join(' ')}"/>`;
  for(const r of va)g+=`<circle cx="${X(r[0])}" cy="${Y(r[1])}" r="4.5" fill="var(--s2)" stroke="var(--surface)" stroke-width="2"/>`;
  const lt=tr[tr.length-1];if(m.r>40){g+=`<text x="${X(lt[0])+8}" y="${Y(lt[1])+4}" font-size="12" fill="var(--ink2)">Training</text>`;
  if(va.length){const lv=va[va.length-1];let y=Y(lv[1])+4;if(Math.abs(y-Y(lt[1])-4)<14)y+=14;g+=`<text x="${X(lv[0])+8}" y="${y}" font-size="12" fill="var(--ink2)">Validierung</text>`}}
  g+=`<line id="xh" x1="0" x2="0" y1="${m.t}" y2="${H-m.b}" stroke="var(--axis)" stroke-width="1" visibility="hidden"/><circle id="xd" r="5" fill="var(--s1)" stroke="var(--surface)" stroke-width="2" visibility="hidden"/>`;
  g+=`<rect x="${m.l}" y="${m.t}" width="${W-m.l-m.r}" height="${H-m.t-m.b}" fill="transparent" id="hit"/>`;
  svg.innerHTML=g;
  const hit=$('hit'),tip=$('tip'),xh=$('xh'),xd=$('xd');
  const move=ev=>{const pt=svg.createSVGPoint();const e=ev.touches?ev.touches[0]:ev;pt.x=e.clientX;pt.y=e.clientY;
    const p=pt.matrixTransform(svg.getScreenCTM().inverse());const step=x0+(p.x-m.l)/(W-m.l-m.r)*(x1-x0);
    let best=0;for(let i=1;i<tr.length;i++)if(Math.abs(tr[i][0]-step)<Math.abs(tr[best][0]-step))best=i;
    const r=tr[best];const vx=X(r[0]),vy=Y(r[1]);
    xh.setAttribute('x1',vx);xh.setAttribute('x2',vx);xh.setAttribute('visibility','visible');
    xd.setAttribute('cx',vx);xd.setAttribute('cy',vy);xd.setAttribute('visibility','visible');
    let v=null;for(const q of va)if(q[0]<=r[0])v=q;
    tip.innerHTML=`<b>Schritt ${r[0].toLocaleString('de-DE')}</b><br>Training: ${num(r[1],3)}`+(v?`<br>Validierung: ${num(v[1],3)} <span style="color:var(--muted)">(Schritt ${v[0].toLocaleString('de-DE')})</span>`:'')+`<br><span style="color:var(--muted)">${(r[2]/1e6).toLocaleString('de-DE',{maximumFractionDigits:1})} Mio. Tokens · ${fmtT(r[3])}</span>`;
    const box=$('chart').getBoundingClientRect(),sx=vx/W*box.width;tip.style.display='block';
    tip.style.left=Math.min(Math.max(sx+12,0),box.width-tip.offsetWidth)+'px';tip.style.top='8px'};
  hit.addEventListener('mousemove',move);hit.addEventListener('touchmove',move,{passive:true});hit.addEventListener('touchstart',move,{passive:true});
  hit.addEventListener('mouseleave',()=>{tip.style.display='none';xh.setAttribute('visibility','hidden');xd.setAttribute('visibility','hidden')});
}
refresh();setInterval(refresh,10000);
let rz;addEventListener('resize',()=>{clearTimeout(rz);rz=setTimeout(()=>data&&drawChart(data.log),150)});
</script></body></html>"""


def serve(work: Path, port: int = 8900, network: bool = False, open_browser: bool = False) -> None:
    page = HTML.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/api/runs":
                runs = [{"ordner": r["ordner"], "lauf": r.get("lauf"), "aktualisiert": r.get("aktualisiert")}
                        for r in find_runs(work)]
                self._send(json.dumps(runs).encode(), "application/json")
            elif url.path == "/api/run":
                name = parse_qs(url.query).get("ordner", [""])[0]
                folder = work / Path(name).name  # nur Unterordner des Arbeitsordners
                try:
                    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    self.send_error(404)
                    return
                status["ordner"] = folder.name
                bew = None
                if (folder / "bewertung.json").exists():
                    try:
                        bew = json.loads((folder / "bewertung.json").read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        bew = None
                body = {"status": status, "log": read_log(folder / status.get("log", "vortraining_log.csv")),
                        "bewertung": bew}
                self._send(json.dumps(body).encode(), "application/json")
            else:
                self._send(page, "text/html; charset=utf-8")

    host = "0.0.0.0" if network else "127.0.0.1"
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"  Dashboard: http://localhost:{port}")
    if network:
        print(f"  Vom Handy (gleiches WLAN): http://{local_ip()}:{port}")
        print("  Hinweis: Windows fragt evtl. nach einer Firewall-Freigabe -> 'Private Netzwerke' erlauben.")
    print("  Beenden mit Strg+C")
    if open_browser:
        import webbrowser

        webbrowser.open(f"http://localhost:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
