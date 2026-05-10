"""
AI Portfolio Manager — main entry point.

Usage:
    python main.py backtest                          # single-asset (ZC=F)
    python main.py backtest --universe ZC=F ZS=F ZW=F
    python main.py backtest --strategy ensemble --fees retail
    python main.py backtest --sentiment              # enable Claude sentiment
    python main.py report                            # pretty-print last run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Logging ──────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# ── Imports (after logging setup) ────────────
import config.settings as cfg
from config.settings import (
    DEFAULT_UNIVERSE, FEES_ALPACA, FEES_IBKR_TIERED,
    FEES_RETAIL, FEES_ZERO, INITIAL_CAPITAL, MULTI_ASSET_UNIVERSE,
)
from portfolio.backtester import Backtester
from utils.reporter import report as print_report


FEE_PRESETS = {
    "zero":   FEES_ZERO,
    "alpaca": FEES_ALPACA,
    "ibkr":   FEES_IBKR_TIERED,
    "retail": FEES_RETAIL,
}


def cmd_backtest(args):
    universe = args.universe or DEFAULT_UNIVERSE

    # Override fee schedule at runtime
    fee_schedule = FEE_PRESETS.get(args.fees, FEES_ZERO)
    cfg.ACTIVE_FEE_SCHEDULE = fee_schedule
    log.info(f"Fee schedule: {args.fees}  "
             f"(comm={fee_schedule.commission_pct*100:.3f}%  "
             f"spread={fee_schedule.spread_pct*100:.3f}%  "
             f"slippage={fee_schedule.slippage_pct*100:.3f}%)")

    bt = Backtester(
        universe=universe,
        initial_cash=args.capital,
        strategy_name=args.strategy,
        use_sentiment=args.sentiment,
        retrain=args.retrain,
        sizing=args.sizing,
        augment=args.augment,
        augment_paths=args.augment_paths,
    )
    result = bt.run()

    # Save result
    out_path = Path("last_backtest.json")
    out_path.write_text(json.dumps(result, indent=2))
    log.info(f"Full results saved → {out_path}")

    # Print report
    print_report(bt.portfolio, fmt=args.format)


def cmd_report(args):
    out_path = Path("last_backtest.json")
    if not out_path.exists():
        log.error("No backtest results found. Run 'python main.py backtest' first.")
        sys.exit(1)
    data = json.loads(out_path.read_text())
    _print_from_json(data)


def _print_from_json(data: dict):
    """Pretty-print a saved report dict without re-running the backtest."""
    from utils.reporter import _format_text
    print(_format_text(data))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="portfolio_manager",
        description="AI Portfolio Manager — paper trading engine",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── backtest ──────────────────────────────
    bt = sub.add_parser("backtest", help="Run a full backtest.")
    bt.add_argument(
        "--universe", nargs="+", default=None,
        help="Tickers to trade (default: ZC=F). Example: --universe ZC=F ZS=F ZW=F",
    )
    bt.add_argument("--capital",   type=float, default=INITIAL_CAPITAL,
                    help=f"Starting capital (default: {INITIAL_CAPITAL:,.0f})")
    bt.add_argument("--strategy",  default="fundamental",
                    choices=["momentum", "mean_reversion", "fundamental", "ensemble"],
                    help="Trading strategy")
    bt.add_argument("--sizing",    default="equal",
                    choices=["equal", "kelly", "vol_target"],
                    help="Position sizing method")
    bt.add_argument("--fees",      default="zero",
                    choices=list(FEE_PRESETS.keys()),
                    help="Fee schedule preset (default: zero = pure paper trading)")
    bt.add_argument("--sentiment", action="store_true",
                    help="Enable Claude-powered news sentiment (requires ANTHROPIC_API_KEY)")
    bt.add_argument("--retrain",   action="store_true",
                    help="Force model retraining even if saved weights exist")
    bt.add_argument("--format",    default="text", choices=["text", "json"],
                    help="Output format")
    bt.set_defaults(func=cmd_backtest)
    bt.add_argument("--augment", action="store_true",
                    help="Augment training data with GBM synthetic paths")
    bt.add_argument("--augment-paths", type=int, default=5,
                    help="Number of synthetic GBM paths to add (default: 5)")

    # ── report ────────────────────────────────
    rp = sub.add_parser("report", help="Print report from last backtest.")
    rp.set_defaults(func=cmd_report)

    # ── multi ─────────────────────────────────
    mt = sub.add_parser("multi", help="Shortcut: run multi-asset backtest.")
    mt.add_argument("--strategy", default="ensemble",
                    choices=["momentum", "mean_reversion", "fundamental", "ensemble"])
    mt.add_argument("--fees", default="alpaca",
                    choices=list(FEE_PRESETS.keys()))
    mt.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    mt.add_argument("--format",  default="text", choices=["text", "json"])

    def cmd_multi(args):
        args.universe       = MULTI_ASSET_UNIVERSE
        args.sizing         = "vol_target"
        args.sentiment      = False
        args.retrain        = False
        args.augment        = False
        args.augment_paths  = 5
        cmd_backtest(args)

    mt.set_defaults(func=cmd_multi)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
