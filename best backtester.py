from __future__ import annotations
from datetime import datetime
from typing import List, Optional
import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==== PARAMETRE ====
START_DATE = "2024-01-01"          # Hent historikk nok til 3-dagers retur
END_DATE: Optional[str] = None     # None = i dag
LOOKBACK_DAYS = 3                  # Fast 3-dagers retur
DROP_THRESHOLD = -0.03             # BUY hvis retur <= terskel (f.eks. -3%)
WEBHOOK_TIMEOUT = 12               # sekunder
# ====================

# >>> WEBHOOK FRA SECRET (ingen fallback) <<<
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
if not DISCORD_WEBHOOK_URL:
    raise SystemExit("Mangler DISCORD_WEBHOOK_URL (GitHub secret). Avbryter.")

TICKERS = [
    "PROT.OL",   # Protector
    "GJF.OL",    # Gjensidige
    "STB.OL",    # Storebrand
    "ORK.OL",    # Orkla
    "EPR.OL",    # Europris
    "KID.OL",    # Kid
    "DNB.OL",    # DNB
    "SB1NO.OL",  # SpareBank 1 Sør-Norge (SR-Bank)
    "SBNOR.OL",  # Sparebanken Vest
    "MING.OL",   # SpareBank 1 SMN
    "NONG.OL",   # SpareBank 1 Nord-Norge
    "MORG.OL",   # Sparebanken Møre
    "VEI.OL",    # Veidekke
    "AFG.OL",    # AF Gruppen
]

# ---------- HTTP m/ retry ----------
def _requests_session_with_retry(total: int = 3) -> requests.Session:
    retry = Retry(
        total=total,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

# ---------- Data ----------
def download_adjusted_close(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    """Last adjusted close (auto_adjust=True) til DataFrame [date x ticker]."""
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # Normaliser til én kolonne per ticker (Close er justert når auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            if (t, "Close") in df.columns:
                closes[t] = df[(t, "Close")].rename(t)
        if not closes:
            raise RuntimeError("Ingen prisdata returnert. Sjekk tickere/dato.")
        prices = pd.concat(closes.values(), axis=1).sort_index()
    else:
        # Én ticker-tilfelle
        prices = df["Close"].to_frame()
        prices.columns = tickers

    # Dropp rader som er helt tomme og fyll små hull
    prices = prices.dropna(how="all").ffill()
    return prices

# ---------- Discord webhook ----------
def _is_discord(url: str) -> bool:
    return "discord.com/api/webhooks" in (url or "")

def send_generic_webhook(payload: dict, url: str) -> None:
    s = _requests_session_with_retry()
    try:
        resp = s.post(url, json=payload, timeout=WEBHOOK_TIMEOUT)
        if 200 <= resp.status_code < 300:
            print(f"Webhook (generic): sendt OK ({resp.status_code}).")
        else:
            print(f"Webhook (generic): feilet {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        print(f"Webhook (generic): unntak: {e}")

def _build_table_preview(df: pd.DataFrame, max_rows: int = 15) -> str:
    """Kort tabellvisning i code block for Discord."""
    if df.empty:
        return "Ingen data."
    view = df.copy()
    view["ticker"] = view["ticker"].astype(str)
    txt = view.head(max_rows).to_string(index=False)
    return f"```\n{txt}\n```"

def send_discord_webhook(
    url: str,
    title: str,
    summary: str,
    df: pd.DataFrame,
    as_of: str,
    lookback: int,
    thresh: float,
    missing: list[str],
    username: str = "3-Day Scanner",
    avatar_url: Optional[str] = None,
    attach_csv_path: Optional[str] = None,
) -> None:
    """
    Sender til Discord med embed + valgfritt CSV-vedlegg.
    Hvis CSV er vedlagt, brukes multipart med 'payload_json' + 'files'.
    """
    embed_desc = (
        f"**As of:** {as_of}\n"
        f"**Lookback:** {lookback} d\n"
        f"**Threshold:** {thresh:.2%}\n"
        f"**Tickers:** {len(TICKERS)}\n"
        f"**Rows:** {len(df)}\n"
    )
    if missing:
        embed_desc += f"**Mangler pris:** {', '.join(missing)}\n"

    table_preview = _build_table_preview(df, max_rows=20)

    embed = {
        "title": title,
        "description": embed_desc,
        "fields": [
            {"name": "Summary", "value": summary[:1000] or "-", "inline": False},
            {"name": "Top rows", "value": table_preview[:1900], "inline": False},
        ],
    }

    payload = {
        "username": username,
        **({"avatar_url": avatar_url} if avatar_url else {}),
        "content": None,
        "embeds": [embed],
    }

    s = _requests_session_with_retry()

    # Med vedlegg (CSV)
    if attach_csv_path and os.path.exists(attach_csv_path):
        try:
            with open(attach_csv_path, "rb") as f:
                files = {"file": (os.path.basename(attach_csv_path), f, "text/csv")}
                data = {"payload_json": json.dumps(payload)}
                resp = s.post(url, data=data, files=files, timeout=WEBHOOK_TIMEOUT)
        except Exception as e:
            print(f"Discord webhook: unntak ved filopplasting: {e}")
            return
    else:
        # Uten vedlegg
        try:
            resp = s.post(url, json=payload, timeout=WEBHOOK_TIMEOUT)
        except Exception as e:
            print(f"Discord webhook: unntak ved sending: {e}")
            return

    if 200 <= resp.status_code < 300:
        print(f"Discord webhook: sendt OK ({resp.status_code}).")
    else:
        print(f"Discord webhook: feilet {resp.status_code}: {resp.text[:300]}")

def send_webhook_auto(data: dict, df: pd.DataFrame, csv_path: Optional[str], url: Optional[str]) -> None:
    url = (url or "").strip()
    if not url:
        print("Webhook: ingen URL satt – hopper over sending.")
        return

    if _is_discord(url):
        title = "3-Day Return Scan (BUY/HOLD)"
        send_discord_webhook(
            url=url,
            title=title,
            summary=data.get("summary", ""),
            df=df,
            as_of=data.get("as_of", ""),
            lookback=data.get("lookback_days", LOOKBACK_DAYS),
            thresh=data.get("drop_threshold", DROP_THRESHOLD),
            missing=data.get("missing_last_price", []),
            username="Market Scanner",
            avatar_url=None,
            attach_csv_path=csv_path,
        )
    else:
        send_generic_webhook(data, url)

# ---------- Scan ----------
def run_scan(start: str, end: Optional[str]) -> None:
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    print(f"Laster priser fra {start} til {end} (adjusted close)...")
    prices = download_adjusted_close(TICKERS, start=start, end=end)

    prices = prices.loc[prices.index >= pd.to_datetime(start)]

    if len(prices) < LOOKBACK_DAYS + 1:
        print("For lite data til å beregne 3-dagers retur.")
        payload = {
            "scanner": "3-day return",
            "status": "insufficient_data",
            "start_date": start,
            "end_date": end,
            "lookback_days": LOOKBACK_DAYS,
            "drop_threshold": DROP_THRESHOLD,
            "message": "Too few rows to compute 3-day return.",
        }
        send_webhook_auto(payload, pd.DataFrame(), None, DISCORD_WEBHOOK_URL)
        return

    returns_3d = prices.pct_change(LOOKBACK_DAYS, fill_method=None)

    last_date = prices.index[-1]
    last_close = prices.loc[last_date]
    last_ret = returns_3d.loc[last_date]

    rows = []
    missing = []
    for t in TICKERS:
        r = last_ret.get(t, np.nan)
        p = last_close.get(t, np.nan)
        if np.isnan(p):
            missing.append(t)
            continue
        rec = "BUY" if (not np.isnan(r) and r <= DROP_THRESHOLD) else "HOLD"
        rows.append({
            "ticker": t,
            "last_date": last_date.date().isoformat(),
            "last_close": round(float(p), 4),
            "ret_3d_%": round(float(r) * 100, 2) if not np.isnan(r) else np.nan,
            "recommendation": rec
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["ret_3d_%"], ascending=True)

    print("\n== 3-dagers retur og anbefaling ==")
    if df.empty:
        print("Ingen data å vise.")
    else:
        print(df.to_string(index=False))

    csv_path = "scan_3day.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nLagret til {csv_path}")

    if missing:
        print("\nAdvarsel: Mangler pris på siste dato for:", ", ".join(missing))

    summary_text = (
        f"3-day scan as of {last_date.date().isoformat()} | "
        f"LOOKBACK={LOOKBACK_DAYS} | THRESH={DROP_THRESHOLD:.4f} | "
        f"tickers={len(TICKERS)} | rows={len(df)}"
    )
    payload = {
        "scanner": "3-day return",
        "status": "ok",
        "as_of": last_date.date().isoformat(),
        "start_date": start,
        "end_date": end,
        "lookback_days": LOOKBACK_DAYS,
        "drop_threshold": DROP_THRESHOLD,
        "missing_last_price": missing,
        "summary": summary_text,
        "results": df.to_dict(orient="records"),
    }

    send_webhook_auto(payload, df, csv_path, DISCORD_WEBHOOK_URL)

if __name__ == "__main__":
    run_scan(start=START_DATE, end=END_DATE)
