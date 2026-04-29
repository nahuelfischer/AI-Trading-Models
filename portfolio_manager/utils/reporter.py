"""
On-demand reporter.

Call report() at any time to get a clean, human-readable snapshot of the
portfolio.  Output is silent until you ask for it, exactly as specified.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from portfolio.engine import Portfolio


def report(
    portfolio: Portfolio,
    prices:    Optional[dict] = None,
    fmt:       str = "text",            # "text" | "json"
) -> str:
    """
    Generate an on-demand portfolio report.

    Parameters
    ----------
    portfolio : Portfolio
    prices    : current market prices {ticker: price} — used for live P&L.
                If None, uses last known prices from nav_history.
    fmt       : "text" for human-readable, "json" for machine-readable.

    Returns
    -------
    str — the formatted report (also printed to stdout).
    """
    data = portfolio.report(prices=prices)
    out  = _format_text(data) if fmt == "text" else json.dumps(data, indent=2)
    print(out)
    return out


def _format_text(data: dict) -> str:
    sep  = "─" * 52
    pos  = data["positions"]
    held = {k: v for k, v in pos.items() if abs(v["shares"]) > 1e-6}
    flat = {k: v for k, v in pos.items() if abs(v["shares"]) < 1e-6}

    lines = [
        "",
        "╔══════════════════════════════════════════════════╗",
        "║          AI PORTFOLIO MANAGER  —  REPORT         ║",
        f"║  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S'):<46}  ║",
        "╚══════════════════════════════════════════════════╝",
        "",
        f"  Portfolio value   :  ${data['nav']:>14,.2f}",
        f"  Cash              :  ${data['cash']:>14,.2f}",
        f"  Total return      :  {data['total_return_pct']:>+.2f}%",
        f"  Gain / Loss       :  ${data['gain_loss']:>+14,.2f}",
        "",
        sep,
        "  RISK METRICS",
        sep,
        f"  Sharpe ratio      :  {data['sharpe_ratio']:>8.3f}",
        f"  Max drawdown      :  {data['max_drawdown_pct']:>7.2f}%",
        f"  Trading halted    :  {'YES ⚠' if data['halted'] else 'no'}",
        "",
        sep,
        "  TRADING ACTIVITY",
        sep,
        f"  Total trades      :  {data['n_trades']:>8d}",
        f"  Total fees paid   :  ${data['total_fees']:>13,.2f}",
        "",
    ]

    if held:
        lines += [sep, "  OPEN POSITIONS", sep]
        for ticker, p in held.items():
            pct_str = f"{p['unrealised_pct']:>+.2f}%"
            lines.append(
                f"  {ticker:<8}  {p['shares']:>10.4f} sh  "
                f"@ {p['avg_cost']:>8.2f}  "
                f"MV ${p['market_value']:>12,.2f}  "
                f"P&L {pct_str}"
            )
        lines.append("")

    if flat:
        lines += [sep, "  FLAT (no position)", sep]
        lines.append("  " + "  ".join(flat.keys()))
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)
