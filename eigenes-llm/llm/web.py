"""Kleine lokale Chat-Oberflaeche im Browser (nur Python-Standardbibliothek).

Start:  python -m llm web --modell arbeit/chat_klein
Dann im Browser: http://localhost:8800
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .generate import encode_chat, generate, load_model
from .tokenizer import StreamDecoder

HTML = """<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__NAME__</title>
<style>
:root{--bg:#f6f5f2;--fg:#1d1d1f;--muted:#6b6b70;--card:#fff;--line:#e3e1dc;--acc:#2f6f4f;--me:#e6efe9}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececee;--muted:#9a9aa2;--card:#1d1d21;--line:#2c2c31;--acc:#6fbf95;--me:#1f2b25}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;display:flex;flex-direction:column;height:100vh}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px 12px}
header>div:last-child{display:flex;align-items:center;gap:10px}
header b{font-size:17px}header small{color:var(--muted)}
#log{flex:1;overflow-y:auto;padding:18px;max-width:820px;width:100%;margin:0 auto}
.m{padding:10px 14px;border-radius:12px;margin:8px 0;white-space:pre-wrap;word-wrap:break-word;max-width:85%}
.u{background:var(--me);margin-left:auto}.a{background:var(--card);border:1px solid var(--line)}
form{display:flex;gap:8px;padding:12px 16px;border-top:1px solid var(--line);max-width:820px;width:100%;margin:0 auto}
textarea{flex:1;resize:none;font:inherit;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--fg);height:48px}
button{font:inherit;border:0;border-radius:10px;padding:0 18px;background:var(--acc);color:#fff;cursor:pointer}
button.sec{background:transparent;color:var(--muted);border:1px solid var(--line);padding:6px 12px}
label{color:var(--muted);font-size:14px}input[type=range]{vertical-align:middle;width:90px}
</style></head><body>
<header><div><b>__NAME__</b> <small>__INFO__</small></div>
<div><label>Temperatur <input id="temp" type="range" min="0" max="1.5" step="0.05" value="0.7"> <span id="tv">0.7</span></label>
<button class="sec" id="neu" type="button">Neuer Chat</button></div></header>
<div id="log"></div>
<form id="f"><textarea id="q" placeholder="Nachricht schreiben ... (Enter = senden, Umschalt+Enter = neue Zeile)"></textarea><button>Senden</button></form>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),temp=document.getElementById('temp');
let msgs=[];let busy=false;
temp.oninput=()=>document.getElementById('tv').textContent=temp.value;
document.getElementById('neu').onclick=()=>{msgs=[];log.innerHTML='';q.focus()};
function add(cls,t){const d=document.createElement('div');d.className='m '+cls;d.textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('f').requestSubmit()}});
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const t=q.value.trim();if(!t||busy)return;busy=true;q.value='';
add('u',t);msgs.push({role:'user',content:t});const d=add('a','…');let out='';
try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:msgs,temperature:+temp.value})});
const rd=r.body.getReader(),dec=new TextDecoder();for(;;){const {done,value}=await rd.read();if(done)break;out+=dec.decode(value,{stream:true});d.textContent=out;log.scrollTop=log.scrollHeight}}
catch(err){out='[Fehler: '+err+']';d.textContent=out}
msgs.push({role:'assistant',content:out});busy=false;q.focus()};
q.focus();
</script></body></html>"""


def serve(model_path: Path, port: int = 8800, system: str | None = None, name: str = "Eigen-LM",
          open_browser: bool = False) -> None:
    model, tok, ckpt = load_model(model_path)
    lock = threading.Lock()
    info = f"{model.num_params() / 1e6:.1f}".replace(".", ",") + " Mio. Parameter, lokal"
    page = HTML.replace("__NAME__", name).replace("__INFO__", info).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # noetig fuer gestreamte (chunked) Antworten

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self):
            if self.path != "/api/chat":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            msgs = body.get("messages", [])
            if system:
                msgs = [{"role": "system", "content": system}] + msgs
            temperature = float(body.get("temperature", 0.7))
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def send(text: str) -> None:
                if not text:
                    return
                data = text.encode("utf-8")
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            dec = StreamDecoder(tok, lstrip=True)
            try:
                with lock:
                    generate(model, encode_chat(tok, msgs), max_new_tokens=400, temperature=temperature,
                             top_k=50, top_p=0.9, repetition_penalty=1.1, stop_ids={tok.eos_id, tok.user_id},
                             on_token=lambda t: send(dec.push(t)))
                self.wfile.write(b"0\r\n\r\n")
            except OSError:  # Browser geschlossen/neu geladen (unter Windows u.a. WinError 10053)
                pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"  {name} laeuft auf http://localhost:{port}  (Beenden mit Strg+C)")
    if open_browser:  # erst jetzt, wenn das Modell geladen ist
        import webbrowser

        webbrowser.open(f"http://localhost:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
