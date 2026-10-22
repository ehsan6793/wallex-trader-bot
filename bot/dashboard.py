"""Local web dashboard (no extra dependencies): python -m bot dashboard  ->  http://127.0.0.1:8765

Reads the bot's log files (it never talks to the exchange itself) and writes a small control file the engine
polls every loop: pause/resume new entries, emergency flatten, and approve/reject signals in APPROVAL_MODE.
"""
from __future__ import annotations

import csv
import json
import webbrowser
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ASSETS = Path(__file__).resolve().parent / "assets"


def _tail(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", errors="replace") as f:
        return list(deque(f, maxlen=n))


def _csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _num(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


class Store:
    def __init__(self, s):
        self.s = s

    def p(self, name):
        return self.s.log_dir / name

    def summary(self, mode: str) -> dict:
        eq = _csv(self.p(f"equity_{mode}.csv"))
        trades = _csv(self.p(f"trades_{mode}.csv"))
        state = {}
        if self.p(f"state_{mode}.json").exists():
            try:
                state = json.loads(self.p(f"state_{mode}.json").read_text(encoding="utf-8"))
            except Exception:
                state = {}
        control = {}
        if self.p(f"control_{mode}.json").exists():
            try:
                control = json.loads(self.p(f"control_{mode}.json").read_text(encoding="utf-8"))
            except Exception:
                control = {}
        # equity series (downsample to <= 600 points)
        pts = [(r["time"], _num(r["equity"]), _num(r["drawdown"])) for r in eq if r.get("equity")]
        step = max(1, len(pts) // 600)
        series = pts[::step] + (pts[-1:] if pts and pts[-1] not in pts[::step] else [])
        start = state.get("start_equity") or (pts[0][1] if pts else self.s.start_capital)
        last = pts[-1][1] if pts else start
        max_dd = max((d for _, _, d in pts), default=0.0)
        sells = [t for t in trades if t.get("side") == "SELL"]
        wins = [t for t in sells if _num(t.get("pnl")) > 0]
        # latest signal per symbol + LLM verdicts + model info from the decision log
        latest, llm, model, scans = {}, [], None, {}
        for line in _tail(self.p(f"decisions_{mode}.jsonl"), 1500):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") == "signal":
                latest[d["symbol"]] = d
            elif d.get("type") == "scan":
                scans[d["symbol"]] = {"time": d["time"], "results": d.get("results", []), "mtf": d.get("mtf", {})}
            elif d.get("type") == "llm_review":
                llm.append({"time": d["time"], "reason": d.get("reason", ""), "tokens": d.get("tokens", 0),
                            "calls_today": d.get("calls_today"), "tokens_total": d.get("tokens_total"), **d.get("verdict", {})})
            elif d.get("type") == "model_trained":
                model = {"time": d["time"], **(d.get("meta") or {}), "threshold": d.get("threshold")}
        keep = ("BUY", "SELL", "⛔", "⚠", "خطا", "LLM", "داشبورد", "شروع", "فروش", "تسویه", "چالش")
        events = [l.rstrip("\n") for l in _tail(self.p(f"events_{mode}.log"), 400) if any(k in l for k in keep)][-14:]
        last_beat = pts[-1][0] if pts else None
        alive = False
        if last_beat:
            try:
                alive = (datetime.now(timezone.utc) - datetime.fromisoformat(last_beat)).total_seconds() < 180
            except ValueError:
                pass
        end = self.s.end_dt()
        return {
            "mode": mode, "alive": alive, "last_beat": last_beat,
            "start_equity": start, "equity": last, "pnl": last - start, "pnl_pct": (last / start - 1) * 100 if start else 0,
            "max_dd_pct": max_dd * 100, "orders": len(trades), "round_trips": len(sells),
            "win_rate": (len(wins) / len(sells) * 100) if sells else None,
            "realized_pnl": sum(_num(t.get("pnl")) for t in sells),
            "volume": sum(_num(t.get("toman")) for t in trades),
            "series": series, "positions": state.get("positions") or {}, "halted": state.get("halted", False),
            "control": {"paused": control.get("paused", False), "pending": control.get("pending", [])},
            "signals": sorted(latest.values(), key=lambda d: d.get("prob", 0), reverse=True),
            "llm": llm[-5:], "model": model, "events": events[::-1], "scans": scans,
            "trades": trades[-60:][::-1],
            "end_time": end.isoformat() if end else None,
            "settings": {"timeframe_min": self.s.bar_seconds // 60, "symbols": self.s.symbols, "tp": self.s.tp_min, "sl": self.s.sl_min,
                         "drop": self.s.drop_trigger, "drop_hours": self.s.drop_lookback_bars * self.s.bar_seconds / 3600,
                         "approval_mode": self.s.approval_mode, "llm": bool(self.s.llm_key), "trade_mode": self.s.trade_mode},
        }

    def prices(self, symbol: str, mode: str) -> dict:
        path = self.s.data_dir / f"{symbol}_{self.s.resolution}.csv"
        rows = _csv(path)[-200:]
        candles = [{"t": r[""] if "" in r else list(r.values())[0], "o": _num(r["open"]), "h": _num(r["high"]),
                    "l": _num(r["low"]), "c": _num(r["close"])} for r in rows]
        marks = [{"t": t["time"], "side": t["side"], "price": _num(t["price"]), "reason": t.get("reason", "")}
                 for t in _csv(self.p(f"trades_{mode}.csv")) if t.get("symbol") == symbol]
        return {"symbol": symbol, "candles": candles, "marks": marks}

    def control(self, mode: str, action: str, arg: str | None = None) -> dict:
        path = self.p(f"control_{mode}.json")
        c = {"paused": False, "flatten": False, "pending": [], "decisions": {}}
        if path.exists():
            try:
                c.update(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
        if action == "pause":
            c["paused"] = True
        elif action == "resume":
            c["paused"] = False
        elif action == "flatten":
            c["flatten"] = True
        elif action in ("approve", "reject") and arg:
            c.setdefault("decisions", {})[arg] = action
        else:
            raise ValueError("unknown action")
        self.s.log_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        with self.p(f"events_{mode}.log").open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} داشبورد: دستور «{action}» {arg or ''}\n")
        return {"ok": True, "control": c}


def make_handler(store: Store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else (json.dumps(body, ensure_ascii=False, default=str).encode() if not isinstance(body, str) else body.encode())
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            mode = q.get("mode", "live")
            try:
                if u.path == "/":
                    return self._send(200, PAGE, "text/html; charset=utf-8")
                if u.path == "/font.woff2":
                    return self._send(200, (ASSETS / "Vazirmatn.woff2").read_bytes(), "font/woff2")
                if u.path == "/api/summary":
                    return self._send(200, store.summary(mode))
                if u.path == "/api/prices":
                    return self._send(200, store.prices(q.get("symbol", "BTCTMN"), mode))
            except Exception as e:  # never crash the dashboard
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/api/control":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                return self._send(200, store.control(body.get("mode", "live"), body.get("action", ""), body.get("id")))
            except Exception as e:
                return self._send(400, {"error": str(e)})
    return H


def serve(s, port: int = 8765, open_browser: bool = True):
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(Store(s)))
    url = f"http://127.0.0.1:{port}"
    print(f"داشبورد: {url}  (برای بستن Ctrl+C)", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    srv.serve_forever()


PAGE = r"""<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>داشبورد ربات والکس</title>
<style>
@font-face{font-family:Vaz;src:url(/font.woff2) format("woff2");font-weight:100 900}
:root{--bg:#eef1f5;--card:#fff;--ink:#162231;--ink2:#526070;--ink3:#8a96a3;--line:#dde3ea;--up:#0f8a5f;--upl:#dcf3ea;--dn:#c23a2c;--dnl:#fbe4e0;
--acc:#2f4fb5;--accl:#e3e9fb;--warn:#a86b00;--warnl:#fdf0d5;--r:14px}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0e141c;--card:#16202b;--ink:#e6edf4;--ink2:#a9b5c2;--ink3:#73808e;--line:#253241;
--upl:#123528;--dnl:#3a1b18;--accl:#1c2850;--warnl:#3a2c0d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Vaz,Tahoma,sans-serif;font-size:14px;line-height:1.6}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:14px 20px;background:var(--card);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
h1{font-size:17px;margin:0;font-weight:800}
.pill{padding:3px 12px;border-radius:99px;font-size:12.5px;font-weight:700}
.on{background:var(--upl);color:var(--up)}.off{background:var(--dnl);color:var(--dn)}.idle{background:var(--warnl);color:var(--warn)}
.seg{display:flex;background:var(--bg);border-radius:99px;padding:3px}.seg button{border:0;background:none;padding:4px 14px;border-radius:99px;font:inherit;color:var(--ink2);cursor:pointer}
.seg button.sel{background:var(--card);color:var(--acc);font-weight:700;box-shadow:0 1px 3px rgba(0,0,0,.12)}
.sp{flex:1}.muted{color:var(--ink3);font-size:12.5px}
main{padding:16px 20px;display:grid;gap:14px;grid-template-columns:repeat(12,1fr);max-width:1400px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:14px 16px;min-width:0}
.card h2{font-size:14px;margin:0 0 10px;font-weight:700;color:var(--ink2)}
.kpis{grid-column:1/-1;display:grid;grid-template-columns:repeat(6,1fr);gap:14px}
.kpi b{display:block;font-size:22px;font-weight:800;letter-spacing:-.01em;direction:ltr;text-align:right}.kpi span{color:var(--ink3);font-size:12.5px}
.up{color:var(--up)}.dn{color:var(--dn)}
.c8{grid-column:span 8}.c4{grid-column:span 4}.c6{grid-column:span 6}.c12{grid-column:1/-1}
svg{width:100%;display:block}
table{width:100%;border-collapse:collapse;font-size:13px}th{color:var(--ink3);font-weight:500;text-align:right;padding:6px 6px;border-bottom:1px solid var(--line)}
td{padding:7px 6px;border-bottom:1px solid var(--line);vertical-align:top}td.n{direction:ltr;text-align:right;font-variant-numeric:tabular-nums}
.btn{border:0;border-radius:10px;padding:9px 14px;font:inherit;font-weight:700;cursor:pointer;background:var(--accl);color:var(--acc)}
.btn.red{background:var(--dnl);color:var(--dn)}.btn.green{background:var(--upl);color:var(--up)}
.ctl{display:flex;flex-direction:column;gap:10px}
.feed{max-height:330px;overflow:auto;font-size:12.5px}.feed div{padding:5px 0;border-bottom:1px dashed var(--line)}
.feed .t{color:var(--ink3);direction:ltr;display:inline-block;margin-left:6px}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11.5px;font-weight:700}
.llm{background:var(--accl);border-radius:10px;padding:10px 12px;margin-bottom:8px}
.bar{height:6px;border-radius:9px;background:var(--line);overflow:hidden;margin-top:4px}.bar i{display:block;height:100%;background:var(--acc)}
select{font:inherit;padding:4px 8px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
.scan th.grp{text-align:center;border-bottom:1px dashed var(--line)}.scan .tf{text-align:center;white-space:nowrap}.scan td.tf b{font-size:12px}
.chip{display:inline-block;margin:2px;padding:1px 7px;border-radius:99px;font-size:11.5px;background:var(--bg);color:var(--ink3);border:1px solid var(--line)}.chip.sig{background:var(--upl);color:var(--up);border-color:var(--up);font-weight:700}
.legend{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px;margin-top:12px;font-size:12.5px}.legend>div{border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.empty{color:var(--ink3);padding:18px 0;text-align:center}
@media(max-width:1000px){.c8,.c4,.c6{grid-column:1/-1}.kpis{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<header>
  <h1>ربات معامله‌گر والکس</h1>
  <span id="status" class="pill idle">در انتظار داده</span>
  <div class="seg" id="modes"><button data-m="live" class="sel">زنده</button><button data-m="shadow">شادو (آزمایشی)</button></div>
  <span class="sp"></span>
  <span class="muted" id="cfg"></span>
  <span class="pill idle" id="countdown"></span>
</header>
<main>
  <section class="kpis">
    <div class="card kpi"><span>ارزش حساب (تومان)</span><b id="k-eq">—</b></div>
    <div class="card kpi"><span>سود / زیان</span><b id="k-pnl">—</b></div>
    <div class="card kpi"><span>بازده</span><b id="k-ret">—</b></div>
    <div class="card kpi"><span>حداکثر افت سرمایه</span><b id="k-dd">—</b></div>
    <div class="card kpi"><span>معاملات بسته‌شده / نرخ برد</span><b id="k-tr">—</b></div>
    <div class="card kpi"><span>حجم معاملات (تومان)</span><b id="k-vol">—</b></div>
  </section>
  <section class="card c8"><h2>منحنی سرمایه</h2><div id="eqchart"></div></section>
  <section class="card c4"><h2>کنترل</h2><div class="ctl">
    <div id="ctlstate" class="muted"></div>
    <button class="btn" id="b-pause">توقف ورودهای جدید</button>
    <button class="btn green" id="b-resume">ادامهٔ معاملات خودکار</button>
    <button class="btn red" id="b-flat">فروش اضطراری همه به تومان</button>
    <div class="muted">طبق قوانین چالش، تصمیم خرید و فروش فقط با الگوریتم است. این دکمه‌ها فقط کنترل ایمنی‌اند.</div>
    <div id="pending"></div>
  </div></section>
  <section class="card c8"><h2 style="display:flex;gap:10px;align-items:center">نمودار قیمت و معاملات <select id="sym"></select></h2><div id="pxchart"></div></section>
  <section class="card c4"><h2>ناظر هوش مصنوعی و اعلان‌های مهم</h2><div id="llm"></div><div class="feed" id="feed"></div></section>
  <section class="card c12"><h2>اسکنر بازارها: تایم‌فریم‌ها و استراتژی‌ها (هر ۱۵ دقیقه)</h2><div id="scan"></div>
    <div class="muted" style="margin-top:6px">فقط استراتژی‌ای که در هر ۳ هفتهٔ بک‌تست واقعی (با کارمزد ۰٫۳٪) سودده بود معامله می‌کند؛ بقیه فقط پایش می‌شوند. چیپ سبز یعنی آن استراتژی الان سیگنال داده، ولی معامله نمی‌شود. بازار اسپات تومانی Short ندارد.</div></section>
  <section class="card c12"><h2>پوزیشن‌های باز</h2><div id="pos"></div></section>
  <section class="card c12"><h2>معاملات</h2><div id="trades"></div></section>
</main>
<script>
let MODE='live', SYM=null, DATA=null;
const $=s=>document.querySelector(s);
const fa=n=>String(n).replace(/\d/g,d=>'۰۱۲۳۴۵۶۷۸۹'[d]);
const money=n=>n==null||isNaN(n)?'—':fa(Math.round(n).toLocaleString('en-US')).replace(/,/g,'٬');
const pct=(n,d=2)=>n==null||isNaN(n)?'—':(n>0?'+':'')+fa(n.toFixed(d))+'٪';
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const tfa=t=>{try{return new Date(t).toLocaleString('fa-IR',{timeZone:'Asia/Tehran',hour:'2-digit',minute:'2-digit',month:'short',day:'numeric'})}catch(e){return t}};
function line(el,pts,{h=230,color='var(--acc)',base=null,marks=[],hl=[]}={}){
  if(!pts.length){el.innerHTML='<div class="empty">هنوز داده‌ای ثبت نشده؛ ربات را اجرا کنید.</div>';return}
  const W=el.clientWidth||700,P=8,ys=pts.map(p=>p[1]).concat(base!=null?[base]:[]).concat(marks.map(m=>m.price));
  let lo=Math.min(...ys),hi=Math.max(...ys);if(hi===lo){hi+=1;lo-=1}const pad=(hi-lo)*.08;lo-=pad;hi+=pad;
  const t0=+new Date(pts[0][0]),t1=+new Date(pts[pts.length-1][0])||t0+1;
  const X=t=>P+(W-2*P)*((+new Date(t)-t0)/Math.max(1,t1-t0)),Y=v=>P+(h-2*P)*(1-(v-lo)/(hi-lo));
  const d=pts.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join(' ');
  const area=d+` L${X(pts[pts.length-1][0]).toFixed(1)} ${h-P} L${X(pts[0][0]).toFixed(1)} ${h-P} Z`;
  let s=`<svg viewBox="0 0 ${W} ${h}" role="img" style="direction:ltr"><defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${color}" stop-opacity=".18"/><stop offset="1" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>`;
  for(let i=0;i<=3;i++){const v=lo+(hi-lo)*i/3;s+=`<line x1="${P}" x2="${W-P}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${W-P}" y="${Y(v)-3}" font-size="10.5" fill="var(--ink3)" text-anchor="end">${money(v)}</text>`}
  if(base!=null)s+=`<line x1="${P}" x2="${W-P}" y1="${Y(base)}" y2="${Y(base)}" stroke="var(--ink3)" stroke-dasharray="4 4"/>`;
  for(const x of hl){s+=`<line x1="${P}" x2="${W-P}" y1="${Y(x.v)}" y2="${Y(x.v)}" stroke="${x.c}" stroke-width="1.5" stroke-dasharray="6 4"/><text x="${P+4}" y="${Y(x.v)-4}" font-size="11" fill="${x.c}">${x.label} ${money(x.v)}</text>`}
  s+=`<path d="${area}" fill="url(#g)"/><path d="${d}" fill="none" stroke="${color}" stroke-width="2"/>`;
  for(const m of marks){const x=X(m.t);if(x<P||x>W-P)continue;const up=m.side==='BUY';s+=`<g><title>${up?'خرید':'فروش'} ${money(m.price)} — ${esc(m.reason)}</title><path d="M${x} ${Y(m.price)+(up?12:-12)} l-6 ${up?9:-9} h12 z" transform="translate(0 ${up?-9:9})" fill="${up?'var(--up)':'var(--dn)'}"/></g>`}
  el.innerHTML=s+'</svg>';
}
async function control(action,id){
  if(action==='flatten'&&!confirm('همهٔ پوزیشن‌های ربات به تومان فروخته شود؟'))return;
  await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:MODE,action,id})});refresh();
}
$('#b-pause').onclick=()=>control('pause');$('#b-resume').onclick=()=>control('resume');$('#b-flat').onclick=()=>control('flatten');
document.querySelectorAll('#modes button').forEach(b=>b.onclick=()=>{MODE=b.dataset.m;document.querySelectorAll('#modes button').forEach(x=>x.classList.toggle('sel',x===b));refresh()});
$('#sym').onchange=e=>{SYM=e.target.value;loadPrices()};
async function loadPrices(){if(!SYM)return;const r=await fetch(`/api/prices?symbol=${SYM}&mode=${MODE}`).then(r=>r.json());
  const p=DATA&&DATA.positions[SYM];const hl=p?[{v:p.tp_price,c:'#0f8a5f',label:'حد سود'},{v:p.entry,c:'#5b6fd6',label:'ورود'},{v:p.sl_price,c:'#c23a2c',label:'حد ضرر'}]:[];
  line($('#pxchart'),r.candles.map(c=>[c.t,c.c]),{h:260,color:'#5b6fd6',marks:r.marks,hl});}
async function refresh(){
  let d;try{d=await fetch('/api/summary?mode='+MODE).then(r=>r.json())}catch(e){$('#status').textContent='ارتباط با داشبورد قطع است';return}
  DATA=d;const st=$('#status');
  st.className='pill '+(d.halted?'off':d.alive?(d.control.paused?'idle':'on'):'idle');
  st.textContent=d.halted?'متوقف (حد افت سرمایه)':d.alive?(d.control.paused?'ورود جدید متوقف':'در حال اجرا'):'ربات اجرا نمی‌شود';
  const c=d.settings;$('#cfg').textContent=`استراتژی برگشت · تایم‌فریم ${fa(c.timeframe_min)} دقیقه · ماشه افت ${fa(Math.abs(c.drop*100))}٪ در ${fa(c.drop_hours)} ساعت · حد سود ${fa(c.tp*100)}٪ / ضرر ${fa(c.sl*100)}٪ · LLM ${c.llm?'فعال':'غیرفعال'}`;
  if(d.end_time){const m=Math.round((new Date(d.end_time)-new Date())/60000);$('#countdown').textContent=m>0?`تا پایان چالش: ${fa(Math.floor(m/60))} ساعت و ${fa(m%60)} دقیقه`:'چالش تمام شده'}
  $('#k-eq').textContent=money(d.equity);
  const k=$('#k-pnl');k.textContent=(d.pnl>0?'+':'')+money(d.pnl);k.className=d.pnl>0?'up':d.pnl<0?'dn':'';
  const r=$('#k-ret');r.textContent=pct(d.pnl_pct);r.className=d.pnl_pct>0?'up':d.pnl_pct<0?'dn':'';
  $('#k-dd').textContent=pct(-d.max_dd_pct);$('#k-tr').textContent=fa(d.round_trips)+' / '+(d.win_rate==null?'—':fa(Math.round(d.win_rate))+'٪');$('#k-vol').textContent=money(d.volume);
  line($('#eqchart'),d.series.map(p=>[p[0],p[1]]),{base:d.start_equity,color:d.pnl>=0?'#0f8a5f':'#c23a2c'});
  $('#ctlstate').innerHTML=d.control.paused?'<span class="tag" style="background:var(--warnl);color:var(--warn)">ورود جدید متوقف است</span>':'<span class="tag on">معاملات خودکار فعال</span>';
  $('#pending').innerHTML=(d.control.pending||[]).map(p=>`<div class="llm"><b>${esc(p.symbol)}</b> — ${money(p.amount)} تومان، احتمال ${fa(Math.round(p.prob*100))}٪<div class="muted">${esc((p.reasons||[]).join(' | '))}</div><button class="btn green" onclick="control('approve','${p.id}')">تأیید</button> <button class="btn red" onclick="control('reject','${p.id}')">رد</button></div>`).join('');
  const lv=d.llm.length?d.llm[d.llm.length-1]:null;
  $('#llm').innerHTML=!d.settings.llm?'<div class="muted" style="margin-bottom:8px">ناظر LLM غیرفعال است (LLM_API_KEY در .env)</div>':lv?`<div class="llm"><b>ناظر LLM</b> <span class="muted">${tfa(lv.time)} · ${esc(lv.reason)}</span><div>ضریب ریسک ${fa(lv.risk_multiplier)}${lv.veto&&lv.veto.length?' · وتو: '+esc(lv.veto.join('، ')):' · بدون وتو'}</div><div class="muted">${esc(lv.rationale)}</div><div class="muted">مصرف امروز: ${fa(lv.calls_today??0)} فراخوانی · ${fa(lv.tokens_total??0)} توکن (فقط هنگام سیگنال خرید فراخوانی می‌شود)</div></div>`:'<div class="llm"><b>ناظر LLM</b><div class="muted">فعال و منتظر اولین سیگنال خرید؛ در زمان‌های بی‌سیگنال توکنی مصرف نمی‌شود.</div></div>';
  $('#feed').innerHTML=d.events.length?d.events.map(e=>{const i=e.indexOf(' ');return `<div><span class="t">${tfa(e.slice(0,i))}</span>${esc(e.slice(i+1))}</div>`}).join(''):'<div class="empty">رویدادی نیست</div>';
  const pos=Object.values(d.positions);
  $('#pos').innerHTML=pos.length?`<table><tr><th>بازار</th><th>ورود</th><th>حد سود</th><th>حد ضرر</th><th>مبلغ</th><th>دلیل</th></tr>${pos.map(p=>`<tr><td><b>${esc(p.symbol)}</b><div class="muted">${tfa(p.entry_time)}</div></td><td class="n">${money(p.entry)}</td><td class="n up">${money(p.tp_price)}</td><td class="n dn">${money(p.sl_price)}</td><td class="n">${money(p.cost)}</td><td class="muted">${esc(p.reason).slice(0,140)}</td></tr>`).join('')}</table>`:'<div class="empty">پوزیشن بازی نیست؛ ربات نقد است و منتظر ماشه می‌ماند.</div>';
  $('#trades').innerHTML=d.trades.length?`<table><tr><th>زمان</th><th>بازار</th><th>نوع</th><th>قیمت</th><th>مبلغ (تومان)</th><th>سود/زیان</th><th>دلیل</th></tr>${d.trades.map(t=>{const p=+t.pnl;return `<tr><td class="muted">${tfa(t.time)}</td><td><b>${esc(t.symbol)}</b></td><td>${t.side==='BUY'?'<span class="tag on">خرید</span>':'<span class="tag off">فروش</span>'}</td><td class="n">${money(+t.price)}</td><td class="n">${money(+t.toman)}</td><td class="n ${p>0?'up':p<0?'dn':''}">${t.pnl===''?'':(p>0?'+':'')+money(p)}</td><td class="muted">${esc(t.reason).slice(0,160)}</td></tr>`}).join('')}</table>`:'<div class="empty">هنوز معامله‌ای انجام نشده</div>';
  const sc=Object.entries(d.scans||{});const TF=['1m','5m','15m','30m','1h','4h'];
  const TFN={'1m':'۱ دقیقه','5m':'۵ دقیقه','15m':'۱۵ دقیقه','30m':'۳۰ دقیقه','1h':'۱ ساعت','4h':'۴ ساعت'};
  const tfc=m=>!m||!m.trend?'<td class="tf muted">—</td>':`<td class="tf" title="RSI ${fa(m.rsi)}"><b class="${m.trend==='up'?'up':'dn'}">${m.trend==='up'?'▲ صعودی':'▼ نزولی'}</b><div class="muted">RSI ${fa(m.rsi)}</div></td>`;
  const all=sc.length?sc[0][1].results:[];const act=all.filter(r=>r.enabled),mon=all.filter(r=>!r.enabled);
  const chip=r=>`<span class="chip ${r.signal?'sig':''}" title="${esc(r.name)}: ${esc(r.why)}">${esc(r.short||r.name)}${r.signal?' ✓':''}</span>`;
  $('#scan').innerHTML=sc.length?`<div style="overflow-x:auto"><table class="scan"><tr><th rowspan="2">بازار</th><th colspan="6" class="grp">جهت روند در هر تایم‌فریم (قیمت نسبت به EMA20)</th>${act.map(k=>`<th rowspan="2">${esc(k.name)}<div><span class="tag on">فعال · معامله می‌کند</span></div></th>`).join('')}<th rowspan="2">سایر استراتژی‌ها<div><span class="tag" style="background:var(--bg);color:var(--ink3)">فقط پایش</span></div></th></tr>
<tr>${TF.map(t=>`<th class="tf">${TFN[t]}</th>`).join('')}</tr>
${sc.map(([sym,v])=>{const R=Object.fromEntries(v.results.map(r=>[r.strategy,r]));return `<tr><td><b>${esc(sym)}</b></td>${TF.map(t=>tfc((v.mtf||{})[t])).join('')}${act.map(k=>{const r=R[k.strategy]||{};return `<td>${r.signal?'<span class="tag on">سیگنال خرید</span>':'<span class="muted">منتظر ماشه</span>'}<div class="muted">${esc(r.why||'')}</div></td>`}).join('')}<td>${v.results.filter(r=>!r.enabled).map(chip).join('')}</td></tr>`}).join('')}</table></div>
<div class="legend">${all.map(k=>`<div><b>${esc(k.name)}</b> ${k.enabled?'<span class="tag on">فعال</span>':'<span class="tag" style="background:var(--bg);color:var(--ink3)">فقط پایش</span>'}<div>${esc(k.rule||'')}</div><div class="muted">بک‌تست: ${esc(k.backtest)}</div></div>`).join('')}</div>`:'<div class="empty">اسکن بعد از اولین کندل ۱۵ دقیقه‌ای نمایش داده می‌شود</div>';
  const syms=d.settings.symbols;if($('#sym').options.length!==syms.length){$('#sym').innerHTML=syms.map(s=>`<option>${s}</option>`).join('');SYM=SYM||syms[0];$('#sym').value=SYM}
  loadPrices();
}
refresh();setInterval(refresh,15000);addEventListener('resize',()=>DATA&&refresh());
</script></body></html>"""
