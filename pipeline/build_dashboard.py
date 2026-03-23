"""
RSR — Live Dashboard Builder with time range filtering (1D/1W/1M/1Y/5Y/ALL)
"""
import json, os, sys, math
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PORTFOLIO_FILE, STARTING_CASH

def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        return {"cash": STARTING_CASH, "holdings": {}, "history": [], "trades": []}
    with open(PORTFOLIO_FILE) as f:
        return json.load(f)

def build_dashboard():
    data     = load_portfolio()
    history  = data.get("history", [])
    trades   = data.get("trades",  [])
    holdings = data.get("holdings", {})
    cash     = data.get("cash", STARTING_CASH)

    if history:
        latest   = history[-1]
        total    = latest["total"]
        ret_pct  = latest["return_pct"]
        peak     = max(h["total"] for h in history)
        drawdown = round((total - peak) / peak * 100, 2) if peak else 0
    else:
        total, ret_pct, drawdown = STARTING_CASH, 0.0, 0.0

    all_dates   = [h["date"]  for h in history]
    all_totals  = [h["total"] for h in history]
    all_cash_h  = [h["cash"]  for h in history]
    all_returns = [round((h["total"] / STARTING_CASH - 1) * 100, 3) for h in history]

    dr_dates  = [h["date"] for h in history[1:]]
    dr_values = [round((history[i]["total"] - history[i-1]["total"]) / history[i-1]["total"] * 100, 3)
                 for i in range(1, len(history))]

    # Trade rows
    trade_rows = ""
    for t in reversed(trades[-50:]):
        action = t.get("action","")
        color  = "#3fb950" if action=="BUY" else "#f78166" if action=="SELL" else "#8b949e"
        trade_rows += f'<tr><td>{t.get("date","")}</td><td>{t.get("ticker","")}</td><td style="color:{color};font-weight:bold">{action}</td><td>{t.get("shares","")}</td><td>${float(t.get("price",0)):.2f}</td><td>${float(t.get("value",0)):,.2f}</td></tr>'

    holding_rows = ""
    for ticker, shares in holdings.items():
        holding_rows += f"<tr><td>{ticker}</td><td>{shares}</td><td>—</td></tr>"
    if not holding_rows:
        holding_rows = '<tr><td colspan="3" style="color:#8b949e;text-align:center">No open positions</td></tr>'

    if len(history) > 2:
        rets   = [(history[i]["total"]-history[i-1]["total"])/history[i-1]["total"] for i in range(1,len(history))]
        mean   = sum(rets)/len(rets)
        std    = math.sqrt(sum((r-mean)**2 for r in rets)/len(rets)) if len(rets)>1 else 1e-9
        sharpe = round((mean/std)*math.sqrt(252),3)
    else:
        sharpe = 0

    buy_count  = sum(1 for t in trades if t.get("action")=="BUY")
    sell_count = sum(1 for t in trades if t.get("action")=="SELL")
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ret_color  = "#3fb950" if ret_pct >= 0 else "#f78166"
    dd_color   = "#f78166" if drawdown < -2 else "#8b949e"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>RSR Trading Bot</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px}}
    header{{background:#161b22;border-bottom:1px solid #30363d;padding:16px 32px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}}
    header h1{{font-size:20px;font-weight:700;color:#58a6ff;letter-spacing:1px}}
    header .sub{{color:#8b949e;font-size:12px;margin-top:2px}}
    .updated{{color:#8b949e;font-size:12px;text-align:right}}
    .dot{{display:inline-block;width:8px;height:8px;background:#3fb950;border-radius:50%;margin-right:6px;animation:pulse 2s infinite}}
    @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
    main{{max-width:1300px;margin:0 auto;padding:24px}}
    .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:28px}}
    .card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px 20px}}
    .card .lbl{{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px}}
    .card .val{{font-size:26px;font-weight:700;color:#e6edf3;letter-spacing:-.5px}}
    .card .s{{font-size:12px;color:#8b949e;margin-top:4px}}
    .cgrid{{display:grid;grid-template-columns:2fr 1fr;gap:20px;margin-bottom:28px}}
    @media(max-width:900px){{.cgrid,.tgrid{{grid-template-columns:1fr}}}}
    .cc{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px}}
    .ch{{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}}
    .cc h3{{font-size:13px;color:#8b949e;text-transform:uppercase;letter-spacing:.8px}}
    .rbtn-group{{display:flex;gap:4px}}
    .rbtn{{background:transparent;border:1px solid #30363d;color:#8b949e;border-radius:6px;padding:4px 10px;font-size:11px;font-weight:700;cursor:pointer;transition:all .15s;letter-spacing:.3px}}
    .rbtn:hover{{border-color:#58a6ff;color:#58a6ff}}
    .rbtn.active{{background:#58a6ff22;border-color:#58a6ff;color:#58a6ff}}
    .cw{{position:relative;height:240px}}
    .tgrid{{display:grid;grid-template-columns:2fr 1fr;gap:20px}}
    .tc{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;overflow:hidden}}
    .tc h3{{font-size:13px;color:#8b949e;text-transform:uppercase;letter-spacing:.8px;margin-bottom:14px}}
    .st{{overflow-y:auto;max-height:320px}}
    table{{width:100%;border-collapse:collapse}}
    th{{position:sticky;top:0;background:#0d1117;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.6px;padding:8px 10px;text-align:left;border-bottom:1px solid #30363d}}
    td{{padding:9px 10px;border-bottom:1px solid #21262d;color:#c9d1d9;font-size:13px}}
    tr:last-child td{{border-bottom:none}}
    tr:hover td{{background:#1c2128}}
    footer{{text-align:center;color:#484f58;font-size:12px;padding:32px 0 16px}}
  </style>
</head>
<body>
<header>
  <div>
    <h1>⚡ RSR Trading Bot</h1>
    <div class="sub">Resilient Signal &amp; Returns — Paper Trading Dashboard</div>
  </div>
  <div class="updated"><span class="dot"></span>Auto-updates every hour<br/>Last run: {updated_at}</div>
</header>
<main>
  <div class="stats">
    <div class="card"><div class="lbl">Portfolio Value</div><div class="val">${total:,.0f}</div><div class="s">Started at ${STARTING_CASH:,.0f}</div></div>
    <div class="card"><div class="lbl">Total Return</div><div class="val" style="color:{ret_color}">{ret_pct:+.2f}%</div><div class="s">${total-STARTING_CASH:+,.0f} P&amp;L</div></div>
    <div class="card"><div class="lbl">Cash Available</div><div class="val">${cash:,.0f}</div><div class="s">{cash/total*100:.0f}% of portfolio</div></div>
    <div class="card"><div class="lbl">Sharpe Ratio</div><div class="val">{sharpe:.3f}</div><div class="s">Annualized (daily)</div></div>
    <div class="card"><div class="lbl">Max Drawdown</div><div class="val" style="color:{dd_color}">{drawdown:.2f}%</div><div class="s">From peak</div></div>
    <div class="card"><div class="lbl">Total Trades</div><div class="val">{len(trades)}</div><div class="s">{buy_count} buys · {sell_count} sells</div></div>
  </div>
  <div class="cgrid">
    <div class="cc">
      <div class="ch">
        <h3>Portfolio Value Over Time</h3>
        <div class="rbtn-group">
          <button class="rbtn" onclick="setRange('1D')">1D</button>
          <button class="rbtn" onclick="setRange('1W')">1W</button>
          <button class="rbtn" onclick="setRange('1M')">1M</button>
          <button class="rbtn" onclick="setRange('1Y')">1Y</button>
          <button class="rbtn" onclick="setRange('5Y')">5Y</button>
          <button class="rbtn active" onclick="setRange('ALL')">ALL</button>
        </div>
      </div>
      <div class="cw"><canvas id="portfolioChart"></canvas></div>
    </div>
    <div class="cc">
      <div class="ch"><h3>Daily Returns (%)</h3></div>
      <div class="cw"><canvas id="returnsChart"></canvas></div>
    </div>
  </div>
  <div class="tgrid">
    <div class="tc">
      <h3>Recent Trades</h3>
      <div class="st">
        <table>
          <thead><tr><th>Date</th><th>Ticker</th><th>Action</th><th>Shares</th><th>Price</th><th>Value</th></tr></thead>
          <tbody>{trade_rows or '<tr><td colspan="6" style="color:#8b949e;text-align:center;padding:20px">No trades yet</td></tr>'}</tbody>
        </table>
      </div>
    </div>
    <div class="tc">
      <h3>Open Positions</h3>
      <div class="st">
        <table>
          <thead><tr><th>Ticker</th><th>Shares</th><th>Value</th></tr></thead>
          <tbody>{holding_rows}</tbody>
        </table>
      </div>
    </div>
  </div>
</main>
<footer>RSR — Resilient Signal &amp; Returns &nbsp;·&nbsp; Paper trading only &nbsp;·&nbsp; Auto-refreshes every hour</footer>
<script>
const AD={json.dumps(all_dates)},AT={json.dumps(all_totals)},AC={json.dumps(all_cash_h)},AR={json.dumps(all_returns)};
const DRD={json.dumps(dr_dates)},DRV={json.dumps(dr_values)};
const GRID="rgba(255,255,255,0.06)";
Chart.defaults.font.family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
Chart.defaults.color="#6e7681";

function filterRange(r){{
  if(!AD.length||r==='ALL')return{{d:AD,t:AT,c:AC}};
  const now=new Date(AD[AD.length-1]),cut=new Date(now);
  if(r==='1D')cut.setDate(cut.getDate()-1);
  else if(r==='1W')cut.setDate(cut.getDate()-7);
  else if(r==='1M')cut.setMonth(cut.getMonth()-1);
  else if(r==='1Y')cut.setFullYear(cut.getFullYear()-1);
  else if(r==='5Y')cut.setFullYear(cut.getFullYear()-5);
  const d=[],t=[],c=[];
  AD.forEach((date,i)=>{{if(new Date(date)>=cut){{d.push(date);t.push(AT[i]);c.push(AC[i]);}}}});
  return{{d,t,c}};
}}

const pChart=new Chart(document.getElementById("portfolioChart"),{{
  type:"line",
  data:{{
    labels:AD,
    datasets:[
      {{label:"Total Value",data:AT,borderColor:"#58a6ff",backgroundColor:"rgba(88,166,255,0.08)",borderWidth:2,fill:true,tension:0.3,pointRadius:AT.length>60?0:3,pointHoverRadius:5}},
      {{label:"Cash",data:AC,borderColor:"#3fb950",backgroundColor:"transparent",borderWidth:1.5,borderDash:[4,4],fill:false,tension:0.3,pointRadius:0}}
    ]
  }},
  options:{{
    responsive:true,maintainAspectRatio:false,
    interaction:{{mode:"index",intersect:false}},
    plugins:{{
      legend:{{labels:{{color:"#8b949e",boxWidth:12,font:{{size:11}}}}}},
      tooltip:{{backgroundColor:"#161b22",borderColor:"#30363d",borderWidth:1,callbacks:{{label:c=>" $"+c.parsed.y.toLocaleString(undefined,{{maximumFractionDigits:0}})}}}}
    }},
    scales:{{
      x:{{grid:{{color:GRID}},ticks:{{maxTicksLimit:8,font:{{size:11}}}}}},
      y:{{grid:{{color:GRID}},ticks:{{font:{{size:11}},callback:v=>"$"+(v/1000).toFixed(0)+"k"}}}}
    }}
  }}
}});

function setRange(r){{
  document.querySelectorAll('.rbtn').forEach(b=>b.classList.toggle('active',b.textContent===r));
  const f=filterRange(r);
  pChart.data.labels=f.d;
  pChart.data.datasets[0].data=f.t;
  pChart.data.datasets[1].data=f.c;
  pChart.data.datasets[0].pointRadius=f.d.length>60?0:3;
  pChart.update();
}}

new Chart(document.getElementById("returnsChart"),{{
  type:"bar",
  data:{{
    labels:DRD,
    datasets:[{{label:"Daily Return %",data:DRV,backgroundColor:DRV.map(v=>v>=0?"rgba(63,185,80,0.75)":"rgba(247,129,102,0.75)"),borderRadius:2}}]
  }},
  options:{{
    responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:"#161b22",borderColor:"#30363d",borderWidth:1,callbacks:{{label:c=>" "+c.parsed.y.toFixed(3)+"%"}}}}}},
    scales:{{
      x:{{grid:{{color:GRID}},ticks:{{maxTicksLimit:6,font:{{size:10}}}}}},
      y:{{grid:{{color:GRID}},ticks:{{font:{{size:11}},callback:v=>v.toFixed(1)+"%"}}}}
    }}
  }}
}});
</script>
</body>
</html>"""

    os.makedirs("dashboard", exist_ok=True)
    with open("dashboard/index.html", "w") as f:
        f.write(html)
    print(f"Dashboard built → dashboard/index.html | ${total:,.2f} | {ret_pct:+.2f}% | {len(trades)} trades")

if __name__ == "__main__":
    build_dashboard()