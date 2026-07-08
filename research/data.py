"""Reproducible market-data layer for research.

Fetches TWSE daily OHLCV via twstock ONCE and caches it to CSV so every
experiment reads identical, offline data. Fetching is network-bound and
rate-limited (~1s/month/stock), so always go through the cache.

Cache lives in research/data_cache/<stock_id>.csv with columns:
    date, open, high, low, close, volume
"""

import os
import time

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")

# A liquid, long-history, sector-diversified TWSE subset drawn from the repo's
# universe (generate_stocks_json.py). Kept small so a full fetch is tractable
# and TWSE-friendly; expand deliberately, not casually.
DEFAULT_UNIVERSE = [
    "2330",  # TSMC (semis)
    "2317",  # Hon Hai (EMS)
    "2454",  # MediaTek (IC design)
    "2308",  # Delta (power)
    "2303",  # UMC (foundry)
    "2382",  # Quanta (notebook ODM)
    "3008",  # Largan (optics)
    "2357",  # Asus (systems)
    "2345",  # Accton (networking)
    "2409",  # AUO (panels)
    "2337",  # Macronix (memory)
    "2379",  # Realtek (IC design) -- may be absent from repo list, still liquid
    "3034",  # Novatek (IC design)
    "2408",  # Nanya (memory)
    "3231",  # Wistron (ODM)
    "2376",  # Gigabyte (systems)
    "2360",  # Chroma (test)
    "6770",  # Powerchip (foundry)
    "0050",  # TW50 ETF (market proxy)
    "0056",  # High-dividend ETF (may be absent, harmless if skipped)
]


# Multi-sector TWSE names (Experiment 3) to break the semis-only concentration
# that made volatility/beta the only axis of cross-sectional dispersion. Liquid,
# long-history large/mid caps across 7 non-semis sectors.
MULTI_SECTOR_NEW = [
    # financials
    "2881", "2882", "2891", "2886", "2884", "2892", "2880", "2801",
    # telecom / utility-like
    "2412", "3045", "4904",
    # plastics / materials / steel / cement
    "1301", "1303", "1326", "1101", "1102", "2002",
    # shipping / transport / airlines
    "2603", "2609", "2610", "2618", "2615",
    # food / consumer staples / retail
    "1216", "2912", "1229", "2903",
    # autos / industrials / textiles
    "2207", "2201", "9910", "1476", "9904",
]

# sector tags for every name we may load (semis from DEFAULT_UNIVERSE + above).
SECTOR_MAP = {
    # semis / electronics (the original universe)
    "2330": "semis", "2317": "electronics", "2454": "semis", "2308": "electronics",
    "2303": "semis", "2382": "electronics", "3008": "optics", "2357": "electronics",
    "2345": "electronics", "2409": "panels", "2337": "semis", "2379": "semis",
    "3034": "semis", "2408": "semis", "3231": "electronics", "2376": "electronics",
    "2360": "electronics", "6770": "semis", "0050": "etf", "0056": "etf",
    # financials
    "2881": "financials", "2882": "financials", "2891": "financials",
    "2886": "financials", "2884": "financials", "2892": "financials",
    "2880": "financials", "2801": "financials",
    # telecom
    "2412": "telecom", "3045": "telecom", "4904": "telecom",
    # materials
    "1301": "materials", "1303": "materials", "1326": "materials",
    "1101": "materials", "1102": "materials", "2002": "materials",
    # shipping / transport
    "2603": "shipping", "2609": "shipping", "2610": "transport",
    "2618": "transport", "2615": "shipping",
    # consumer / retail
    "1216": "consumer", "2912": "consumer", "1229": "consumer", "2903": "retail",
    # autos / industrials
    "2207": "auto", "2201": "auto", "9910": "industrial", "1476": "industrial",
    "9904": "industrial",
}


# Experiment 4 expansion — push toward 100+ names with more sector and size
# breadth (mid/small caps too) so factor exposures (esp. size) are estimable.
EXPAND_NEW = {
    # more semis / electronics
    "2344": "semis", "2351": "semis", "2401": "semis", "3037": "electronics",
    "3443": "semis", "2449": "semis", "2458": "semis", "2377": "electronics",
    "2313": "electronics", "2324": "electronics", "2356": "electronics",
    "2312": "electronics", "3711": "semis", "4938": "electronics",
    "2353": "electronics", "2354": "electronics", "2385": "electronics",
    "3661": "semis", "2451": "electronics", "2368": "electronics",
    # more financials
    "2883": "financials", "2887": "financials", "2888": "financials",
    "2890": "financials", "5880": "financials", "2809": "financials",
    "2845": "financials", "2812": "financials",
    # more materials / traditional industrials
    "1402": "materials", "1434": "materials", "1590": "industrial",
    "2027": "materials", "2015": "materials", "1802": "materials",
    "1503": "industrial", "1513": "industrial", "1519": "industrial",
    "9933": "industrial", "6414": "industrial",
    # more consumer / retail / food
    "1201": "consumer", "1210": "consumer", "1215": "consumer",
    "1227": "consumer", "2723": "consumer", "5903": "retail",
    "2915": "retail", "1231": "consumer",
    # more shipping / transport
    "2606": "shipping", "2634": "transport", "2637": "shipping",
    # autos / bikes
    "2231": "auto", "9914": "auto", "9921": "auto", "9945": "construction",
    # biotech / health
    "1707": "biotech", "6446": "biotech", "1789": "biotech", "4133": "biotech",
    # construction / real estate
    "2542": "construction", "2548": "construction", "5522": "construction",
}

SECTOR_MAP.update(EXPAND_NEW)


def cache_path(stock_id):
    return os.path.join(CACHE_DIR, f"{stock_id}.csv")


def load_cached(stock_id):
    """Return the cached DataFrame for a stock, or None if not cached."""
    path = cache_path(stock_id)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def fetch_and_cache(stock_id, from_year=2018, from_month=1, force=False,
                    verbose=True):
    """Fetch OHLCV for a stock and cache it. Returns the DataFrame (or None on
    failure). Cached stocks are returned without hitting the network."""
    if not force:
        cached = load_cached(stock_id)
        if cached is not None:
            return cached

    import twstock  # imported lazily so offline analysis needs no network dep

    try:
        t0 = time.time()
        data = twstock.Stock(stock_id).fetch_from(from_year, from_month)
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[{stock_id}] fetch failed: {type(e).__name__}: {e}")
        return None

    if not data:
        if verbose:
            print(f"[{stock_id}] no data returned")
        return None

    df = pd.DataFrame({
        "date": [d.date for d in data],
        "open": [d.open for d in data],
        "high": [d.high for d in data],
        "low": [d.low for d in data],
        "close": [d.close for d in data],
        "volume": [d.capacity for d in data],
    }).dropna().reset_index(drop=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(cache_path(stock_id), index=False)
    if verbose:
        print(f"[{stock_id}] cached {len(df)} rows "
              f"({df['date'].min().date()}..{df['date'].max().date()}) "
              f"in {time.time() - t0:.0f}s")
    return df


def build_cache(universe=None, from_year=2018, from_month=1):
    """Fetch+cache the whole universe. Resumable: cached stocks are skipped."""
    universe = universe or DEFAULT_UNIVERSE
    ok = 0
    for sid in universe:
        df = fetch_and_cache(sid, from_year, from_month)
        if df is not None and len(df) > 0:
            ok += 1
    print(f"\nCache ready: {ok}/{len(universe)} stocks in {CACHE_DIR}")
    return ok


def load_universe(universe=None):
    """Load all cached stocks into {stock_id: DataFrame}. Skips uncached ones."""
    universe = universe or DEFAULT_UNIVERSE
    out = {}
    for sid in universe:
        df = load_cached(sid)
        if df is not None and len(df) > 0:
            out[sid] = df
    return out


if __name__ == "__main__":
    build_cache()
