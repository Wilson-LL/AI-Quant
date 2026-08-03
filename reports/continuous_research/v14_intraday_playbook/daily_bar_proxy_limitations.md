# v14 Daily-Bar Proxy — Limitations (Task 3, mandatory reading)

Everything in this sprint's empirical work rests on daily OHLCV. That
means, structurally and without remedy:

1. **Path ambiguity.** A daily bar does not reveal whether the high or the
   low occurred first. Any stop-loss / take-profit rule is unverifiable:
   a day hitting both −2% and +3% from open cannot be classified. We
   report bounded best/worst cases only; no first-passage labels exist.
2. **No time axis.** 09:30/10:00/11:00 checkpoint rules cannot be tested
   at all — there is no observation between open and close. Only OPEN
   (gap known) and CLOSE (outcome) are real decision points.
3. **No fills, no slippage.** Open/close prints are auction prices; a real
   order does not necessarily fill there. Day-trade cost assumption used
   (30 bps round trip: reduced day-trade tax 0.15% + fees) is an
   ASSUMPTION, not calibrated.
4. **No VWAP, no intraday volume shape, no index co-movement within the
   day.**
5. **Survivorship-biased universe** (curated liquid names) — intraday
   dislocations of delisted/distressed names are absent; short-side proxy
   results are optimistic.
6. **Short mechanics unmodeled.** TWSE uptick/quota/borrow rules and
   day-trade eligibility flags are not in the data.

**Consequences (binding):** every proxy output row carries
`data_quality=DAILY_BAR_PROXY_ONLY` and `live_trading_allowed=false`; the
rule search is restricted to gap→open-to-close space; no verdict above
DAILY_BAR_PROXY_ONLY_NOT_DECISION_GRADE / PLAYBOOK_FRAMEWORK_READY can be
issued from this sprint; production readiness claims are prohibited until
true intraday data (collector, months of accumulation) exists.
