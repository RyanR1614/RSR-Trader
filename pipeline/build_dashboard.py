"""
RSR — Live Dashboard Builder
Reads portfolio.json and generates a self-contained HTML dashboard
that gets published to GitHub Pages after every run.
Access it at: https://YOUR_USERNAME.github.io/RSR-Trader/
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import PORTFOLIO_FILE, STARTING_CASH, TICKERS


def load_portfolio() -> dict:
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

    # ── Compute summary stats ────────────────────────────────────────────────
    if history:
        latest     = history[-1]
        total      = latest["total"]
        ret_pct    = latest["return_pct"]
        peak       = max(h["total"] for h in history)
        drawdown   = round((total - peak) / peak * 100, 2) if peak else 0
    else:
        total      = STARTING_CASH
        ret_pct    = 0.0
        drawdown   = 0.0

    # Returns series for chart
    dates_js   = json.dumps([h["date"]  for h in history])
    totals_js  = json.dumps([h["total"] for h in history])
    cash_js    = json.dumps([h["cash"]  for h in history])
    returns_js = json.dumps([round((h["total"] / STARTING_CASH - 1) * 100, 3) for h in history])

    # Daily returns for bar chart
    daily_returns = []
    for i in range(1, len(history)):
        prev = history[i-1]["total"]
        curr = history[i]["total"]
        daily_returns.append(round((curr - prev) / prev * 100, 3) if prev else 0)
    daily_ret_dates_js  = json.dumps([h["date"] for h in history[1:]])
    daily_ret_values_js = json.dumps(daily_returns)

    # Recent trades table rows
    recent_trades = trades[-50:][::-1]  # last 50, newest first
    trade_rows = ""
    for t in recent_trades:
        action = t.get("action", "")
        color  = "#3fb950" if action == "BUY" else "#f78166" if action == "SELL" else "#8b949e"
        trade_rows += f"""
        <tr>
          <td>{t.get('date','')}</td>
          <td>{t.get('ticker','')}</td>
          <td style="color:{color};font-weight:bold">{action}</td>
          <td>{t.get('shares','')}</td>
          <td>${float(t.get('price',0)):.2f}</td>
          <td>${float(t.get('value',0)):,.2f}</td>
        </tr>"""

    # Holdings rows
    holding_rows = ""
    for ticker, shares in holdings.items():
        holding_rows += f"""
        <tr>
          <td>{ticker}</td>
          <td>{shares}</td>
          <td>—</td>
        </tr>"""
    if not holding_rows:
        holding_rows = '<tr><td colspan="3" style="color:#8b949e;text-align:center">No open positions</td></tr>'

    # Sharpe ratio (annualized)
    import math
    if len(history) > 2:
        rets  = [(history[i]["total"] - history[i-1]["total"]) / history[i-1]["total"]
                 for i in range(1, len(history))]
        mean  = sum(rets) / len(rets)
        std   = math.sqrt(sum((r - mean)**2 for r in rets) / len(rets)) if len(rets) > 1 else 1e-9
        sharpe = round((mean / std) * math.sqrt(252), 3) if std > 0 else 0
    else:
        sharpe = 0

    total_trade_count = len(trades)
    buy_count  = sum(1 for t in trades if t.get("action") == "BUY")
    sell_count = sum(1 for t in trades if t.get("action") == "SELL")

    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ret_color  = "#3fb950" if ret_pct >= 0 else "#f78166"
    dd_color   = "#f78166" if drawdown < -2 else "#8b949e"

    # ── Build HTML ────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>RSR Trading Bot — Live Dashboard</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0d1117;
      color: #c9d1d9;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }}
    header {{
      background: #161b22;
      border-bottom: 1px solid #30363d;
      padding: 16px 32px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 100;
    }}
    header h1 {{
      font-size: 20px;
      font-weight: 700;
      color: #58a6ff;
      letter-spacing: 1px;
    }}
    header .subtitle {{ color: #8b949e; font-size: 12px; margin-top: 2px; }}
    .updated {{ color: #8b949e; font-size: 12px; text-align: right; }}
    .live-dot {{
      display: inline-block;
      width: 8px; height: 8px;
      background: #3fb950;
      border-radius: 50%;
      margin-right: 6px;
      animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
      0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }}
    }}
    main {{ max-width: 1300px; margin: 0 auto; padding: 24px 24px; }}

    /* ── Stat cards ── */
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 18px 20px;
    }}
    .card .label {{
      color: #8b949e;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-bottom: 8px;
    }}
    .card .value {{
      font-size: 26px;
      font-weight: 700;
      color: #e6edf3;
      letter-spacing: -0.5px;
    }}
    .card .sub {{
      font-size: 12px;
      color: #8b949e;
      margin-top: 4px;
    }}

    /* ── Charts ── */
    .charts-grid {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 20px;
      margin-bottom: 28px;
    }}
    @media (max-width: 900px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
    .chart-card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 20px;
    }}
    .chart-card h3 {{
      font-size: 13px;
      color: #8b949e;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-bottom: 16px;
    }}
    .chart-wrap {{ position: relative; height: 240px; }}

    /* ── Tables ── */
    .tables-grid {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 20px;
    }}
    @media (max-width: 900px) {{ .tables-grid {{ grid-template-columns: 1fr; }} }}
    .table-card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 20px;
      overflow: hidden;
    }}
    .table-card h3 {{
      font-size: 13px;
      color: #8b949e;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-bottom: 14px;
    }}
    .scroll-table {{ overflow-y: auto; max-height: 320px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{
      position: sticky; top: 0;
      background: #0d1117;
      color: #8b949e;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      padding: 8px 10px;
      text-align: left;
      border-bottom: 1px solid #30363d;
    }}
    td {{
      padding: 9px 10px;
      border-bottom: 1px solid #21262d;
      color: #c9d1d9;
      font-size: 13px;
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #1c2128; }}

    /* ── Footer ── */
    footer {{
      text-align: center;
      color: #484f58;
      font-size: 12px;
      padding: 32px 0 16px;
    }}
  </style>
</head>
<body>

<header>
  <div>
    <h1>⚡ RSR Trading Bot</h1>
    <div class="subtitle">Resilient Signal &amp; Returns — Paper Trading Dashboard</div>
  </div>
  <div class="updated">
    <span class="live-dot"></span>Auto-updates every hour
    <br/>Last run: {updated_at}
  </div>
</header>

<main>

  <!-- ── Stats ── -->
  <div class="stats-grid">
    <div class="card">
      <div class="label">Portfolio Value</div>
      <div class="value">${total:,.0f}</div>
      <div class="sub">Started at ${STARTING_CASH:,.0f}</div>
    </div>
    <div class="card">
      <div class="label">Total Return</div>
      <div class="value" style="color:{ret_color}">{ret_pct:+.2f}%</div>
      <div class="sub">${total - STARTING_CASH:+,.0f} P&amp;L</div>
    </div>
    <div class="card">
      <div class="label">Cash Available</div>
      <div class="value">${cash:,.0f}</div>
      <div class="sub">{cash/total*100:.0f}% of portfolio</div>
    </div>
    <div class="card">
      <div class="label">Sharpe Ratio</div>
      <div class="value">{sharpe:.3f}</div>
      <div class="sub">Annualized (daily)</div>
    </div>
    <div class="card">
      <div class="label">Max Drawdown</div>
      <div class="value" style="color:{dd_color}">{drawdown:.2f}%</div>
      <div class="sub">From peak</div>
    </div>
    <div class="card">
      <div class="label">Total Trades</div>
      <div class="value">{total_trade_count}</div>
      <div class="sub">{buy_count} buys · {sell_count} sells</div>
    </div>
  </div>

  <!-- ── Charts ── -->
  <div class="charts-grid">
    <div class="chart-card">
      <h3>Portfolio Value Over Time</h3>
      <div class="chart-wrap">
        <canvas id="portfolioChart"></canvas>
      </div>
    </div>
    <div class="chart-card">
      <h3>Daily Returns (%)</h3>
      <div class="chart-wrap">
        <canvas id="returnsChart"></canvas>
      </div>
    </div>
  </div>

  <!-- ── Tables ── -->
  <div class="tables-grid">
    <div class="table-card">
      <h3>Recent Trades</h3>
      <div class="scroll-table">
        <table>
          <thead>
            <tr>
              <th>Date</th><th>Ticker</th><th>Action</th>
              <th>Shares</th><th>Price</th><th>Value</th>
            </tr>
          </thead>
          <tbody>
            {trade_rows if trade_rows else '<tr><td colspan="6" style="color:#8b949e;text-align:center;padding:20px">No trades yet — run the pipeline first</td></tr>'}
          </tbody>
        </table>
      </div>
    </div>
    <div class="table-card">
      <h3>Open Positions</h3>
      <div class="scroll-table">
        <table>
          <thead>
            <tr><th>Ticker</th><th>Shares</th><th>Value</th></tr>
          </thead>
          <tbody>{holding_rows}</tbody>
        </table>
      </div>
    </div>
  </div>

</main>

<footer>
  RSR — Resilient Signal &amp; Returns &nbsp;·&nbsp; Paper trading only — no real money &nbsp;·&nbsp;
  Auto-refreshes every hour via GitHub Actions
</footer>

<script>
const DATES   = {dates_js};
const TOTALS  = {totals_js};
const CASH    = {cash_js};
const RETURNS = {returns_js};
const DR_DATES  = {daily_ret_dates_js};
const DR_VALUES = {daily_ret_values_js};

const gridColor  = "rgba(255,255,255,0.06)";
const tickColor  = "#6e7681";
const fontFamily = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

Chart.defaults.font.family = fontFamily;
Chart.defaults.color       = tickColor;

// ── Portfolio value chart ──────────────────────────────────────────────────
new Chart(document.getElementById("portfolioChart"), {{
  type: "line",
  data: {{
    labels: DATES,
    datasets: [
      {{
        label: "Total Value",
        data: TOTALS,
        borderColor: "#58a6ff",
        backgroundColor: "rgba(88,166,255,0.08)",
        borderWidth: 2,
        fill: true,
        tension: 0.3,
        pointRadius: TOTALS.length > 60 ? 0 : 3,
        pointHoverRadius: 5,
      }},
      {{
        label: "Cash",
        data: CASH,
        borderColor: "#3fb950",
        backgroundColor: "transparent",
        borderWidth: 1.5,
        borderDash: [4, 4],
        fill: false,
        tension: 0.3,
        pointRadius: 0,
      }},
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    interaction: {{ mode: "index", intersect: false }},
    plugins: {{
      legend: {{ labels: {{ color: "#8b949e", boxWidth: 12, font: {{ size: 11 }} }} }},
      tooltip: {{
        backgroundColor: "#161b22",
        borderColor: "#30363d",
        borderWidth: 1,
        callbacks: {{
          label: ctx => " $" + ctx.parsed.y.toLocaleString(undefined, {{maximumFractionDigits:0}})
        }}
      }}
    }},
    scales: {{
      x: {{ grid: {{ color: gridColor }}, ticks: {{ maxTicksLimit: 8, font: {{ size: 11 }} }} }},
      y: {{
        grid: {{ color: gridColor }},
        ticks: {{
          font: {{ size: 11 }},
          callback: v => "$" + (v/1000).toFixed(0) + "k"
        }}
      }}
    }}
  }}
}});

// ── Daily returns bar chart ────────────────────────────────────────────────
new Chart(document.getElementById("returnsChart"), {{
  type: "bar",
  data: {{
    labels: DR_DATES,
    datasets: [{{
      label: "Daily Return %",
      data: DR_VALUES,
      backgroundColor: DR_VALUES.map(v => v >= 0 ? "rgba(63,185,80,0.75)" : "rgba(247,129,102,0.75)"),
      borderRadius: 2,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor: "#161b22",
        borderColor: "#30363d",
        borderWidth: 1,
        callbacks: {{
          label: ctx => " " + ctx.parsed.y.toFixed(3) + "%"
        }}
      }}
    }},
    scales: {{
      x: {{ grid: {{ color: gridColor }}, ticks: {{ maxTicksLimit: 6, font: {{ size: 10 }} }} }},
      y: {{
        grid: {{ color: gridColor }},
        ticks: {{ font: {{ size: 11 }}, callback: v => v.toFixed(1) + "%" }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    os.makedirs("dashboard", exist_ok=True)
    with open("dashboard/index.html", "w") as f:
        f.write(html)
    print(f"Dashboard built → dashboard/index.html")
    print(f"  Portfolio: ${total:,.2f}  |  Return: {ret_pct:+.2f}%  |  Trades: {total_trade_count}")


if __name__ == "__main__":
    build_dashboard()
