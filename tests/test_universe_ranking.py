"""Tests for the full-universe ranking layer (decision support only).

Covers the acceptance list of the feature request: completeness,
determinism, coverage transparency, portfolio invariance (the validated
strategy is READ-ONLY here), holdings-first prioritization, short-side
constraints, agreement semantics, rank persistence, and report layout.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research"))
sys.path.insert(0, os.path.join(REPO, "research", "intraday_advisor"))

import universe_ranking as ur  # noqa: E402
import simplified_reports as sr  # noqa: E402

DATE = "2026-02-13"
PREV = "2026-02-12"

RANKED = [f"{i:04d}" for i in range(1101, 1121)]  # 20 ranked symbols
# +3 data-unready +1 real ETF (model-scope exclusion, sector rule)
CONFIGURED = RANKED + ["7777", "8888", "9999", "0056"]


def _hash(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def make_root(tmp, holdings=True):
    pt = os.path.join(tmp, "reports", "paper_trading")
    tg = os.path.join(tmp, "reports", "transformer_gpu")
    dc = os.path.join(tmp, "research", "data_cache")
    for d in (pt, tg, dc):
        os.makedirs(d, exist_ok=True)

    n = len(RANKED)
    uni = pd.DataFrame({
        "symbol": RANKED,
        "tf_score": np.linspace(0.9, -0.9, n).round(5),
        "seed_score_std": np.linspace(0.01, 0.20, n).round(5),
        "momentum": np.linspace(0.5, -0.5, n).round(5),
        "z_tf": np.linspace(2, -2, n).round(4),
        "z_momentum": np.linspace(2, -2, n).round(4),
        "blend_score": np.linspace(2.0, -2.0, n).round(4),
        "confidence": (["high"] * 7 + ["medium"] * 7 + ["low"] * 6),
        "rank": range(1, n + 1),
        "sector": "semis", "signal_date": DATE,
    })
    uni.to_csv(os.path.join(pt, f"{DATE}_blend50_universe_scores.csv"),
               index=False)

    # book: 1101 BUY (held SHORT), 1102 HOLD (held LONG), 1103 BUY
    # (unheld), 1104 WATCH (unheld), 1106 SELL (held LONG)
    book = pd.DataFrame([
        ("1101", "BUY", 0.25, 0.00, 1), ("1102", "HOLD", 0.30, 0.30, 2),
        ("1103", "BUY", 0.30, 0.00, 3), ("1104", "WATCH", 0.0, 0.0, 4),
        ("1106", "SELL", 0.0, 0.10, 6),
    ], columns=["symbol", "action", "target_weight", "previous_weight",
                "rank"])
    book["model_score"] = 1.0
    book.to_csv(os.path.join(
        pt, f"{DATE}_blend50_band10_decision_book.csv"), index=False)

    preds = pd.DataFrame({"stock": RANKED + ["7777"], "score": 0.1})
    preds.to_csv(os.path.join(tg, f"{DATE}_predictions.csv"), index=False)

    # exclusion-reason fixtures: 7777 scored but momentum-short cache;
    # 8888 no cache at all; 9999 stale cache
    dts = pd.date_range("2025-01-01", periods=300, freq="B")
    pd.DataFrame({"date": dts[:100], "close": 50.0}).to_csv(
        os.path.join(dc, "7777.csv"), index=False)
    pd.DataFrame({"date": dts[:200], "close": 50.0}).to_csv(
        os.path.join(dc, "9999.csv"), index=False)

    if holdings:
        with open(os.path.join(tmp, "my_holdings.csv"), "w",
                  encoding="utf-8") as f:
            f.write("symbol,side,shares,avg_cost,current_price,"
                    "current_value,account,notes\n"
                    "1101,SHORT,1000,120,118,118000,a,\n"
                    "1102,LONG,1000,60,60,30000,a,\n"
                    "1106,LONG,2000,35,35,70000,a,\n"
                    "0099,LONG,100,10,10,1000,a,outside universe\n")
    return tmp


class TestUniverseRanking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="uniranktest_")
        cls.root = make_root(cls.tmp)
        cls.out = os.path.join(cls.tmp, "reports", "user_actions")
        cls.df, cls.meta = ur.build_ranking(
            cls.root, holdings_path=os.path.join(cls.root,
                                                 "my_holdings.csv"),
            asof=DATE, out_dir=cls.out, configured=CONFIGURED)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # 1. every eligible name exactly once (held rows may add unranked)
    def test_all_eligible_exactly_once(self):
        ranked = self.df[pd.notna(self.df["universe_rank"])]
        self.assertEqual(sorted(ranked["symbol"]), sorted(RANKED))
        self.assertTrue(ranked["symbol"].is_unique)

    # 2. deterministic output
    def test_deterministic(self):
        h1 = _hash(os.path.join(self.out, "latest_universe_ranking.csv"))
        ur.build_ranking(self.root,
                         holdings_path=os.path.join(self.root,
                                                    "my_holdings.csv"),
                         asof=DATE, out_dir=self.out,
                         configured=CONFIGURED)
        self.assertEqual(
            h1, _hash(os.path.join(self.out,
                                   "latest_universe_ranking.csv")))

    # 3. rank 1 = strongest bullish production signal
    def test_rank1_strongest(self):
        ranked = self.df[pd.notna(self.df["universe_rank"])]
        top = ranked.loc[ranked["universe_rank"].idxmin()]
        self.assertEqual(int(top["universe_rank"]), 1)
        self.assertEqual(float(top["model_score"]),
                         float(ranked["model_score"].max()))

    # 4/5. three universe scopes + visible exclusions with reasons
    def test_coverage_and_exclusions(self):
        m = self.meta
        self.assertEqual(m["configured_universe_count"], 24)
        self.assertEqual(m["model_eligible_count"], 23)  # 0056 = etf
        self.assertEqual(m["scored_count"], 20)
        self.assertAlmostEqual(m["configured_coverage_ratio"], 20 / 24,
                               places=3)
        self.assertAlmostEqual(m["model_scored_coverage_ratio"], 20 / 23,
                               places=3)
        ex = dict(m["excluded"])
        self.assertEqual(ex["0056"], "OUTSIDE_MODEL_SCOPE")
        self.assertEqual(ex["8888"], "NO_CACHE")
        self.assertEqual(ex["9999"], "STALE_DATA")
        self.assertEqual(ex["7777"], "INSUFFICIENT_HISTORY")
        with open(os.path.join(self.out, "latest_universe_ranking.md"),
                  encoding="utf-8") as f:
            md = f.read()
        for s in ("8888", "9999", "7777", "0056"):
            self.assertIn(s, md)
        self.assertIn("模型範圍外", md)      # scope != data failure
        self.assertIn("99% EOD", md)         # explicit gate disclaimer

    # model-scope exclusion is classified by rule, never by ticker, and
    # never labeled a validation failure
    def test_scope_exclusion_not_error(self):
        self.assertTrue(ur.outside_model_scope("0056"))
        self.assertTrue(ur.outside_model_scope("0050"))
        self.assertFalse(ur.outside_model_scope("2330"))
        ex = dict(self.meta["excluded"])
        self.assertNotEqual(ex["0056"], "OTHER_VALIDATION_FAILURE")

    # a healthy scoring session with scope exclusions must never look
    # like a failed 99% EOD publication event
    def test_scope_exclusions_never_look_like_gate_failure(self):
        tmp2 = tempfile.mkdtemp(prefix="unirankgate_")
        try:
            root2 = make_root(tmp2, holdings=False)
            out2 = os.path.join(tmp2, "reports", "user_actions")
            _, m = ur.build_ranking(
                root2, holdings_path=None, asof=DATE, out_dir=out2,
                configured=RANKED + ["0056"])  # only a scope exclusion
            self.assertEqual(m["model_scored_coverage_ratio"], 1.0)
            self.assertLess(m["configured_coverage_ratio"], 1.0)
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
        # the production publication gate keeps its own contract:
        # threshold unchanged, and neither gate module imports this layer
        import user_next_session_plan as unsp
        import pipeline_gate
        self.assertEqual(unsp.PARTIAL_COVERAGE_MIN, 0.99)
        with open(pipeline_gate.__file__.replace(".pyc", ".py"),
                  encoding="utf-8") as f:
            self.assertNotIn("universe_ranking", f.read())

    # 6/7/8. portfolio actions, membership, weights untouched
    def test_portfolio_untouched(self):
        bp = os.path.join(self.root, "reports", "paper_trading",
                          f"{DATE}_blend50_band10_decision_book.csv")
        before = _hash(bp)
        ur.build_ranking(self.root, holdings_path=None, asof=DATE,
                         out_dir=self.out, configured=CONFIGURED)
        self.assertEqual(before, _hash(bp))  # book file byte-identical
        book = pd.read_csv(bp, dtype={"symbol": str})
        for _, b in book.iterrows():
            row = self.df[(self.df["symbol"] == b["symbol"])]
            self.assertTrue((row["model_action"] == b["action"]).all())
            for tw in row["target_weight"]:
                self.assertAlmostEqual(tw, b["target_weight"])
        members = set(self.df[self.df["portfolio_member"]]["symbol"])
        self.assertEqual(members,
                         set(book[book["target_weight"] > 0]["symbol"]))

    # 9. ranking alone never creates a BUY
    def test_rank_never_creates_buy(self):
        unowned_nonbook = self.df[(~self.df["is_actual_holding"])
                                  & (self.df["model_action"] == "")]
        self.assertGreater(len(unowned_nonbook), 0)
        self.assertFalse(unowned_nonbook["user_action"]
                         .isin(ur.ENTRY_ACTIONS).any())
        # even the best-ranked unowned non-book symbol
        cand = ur.top_nonportfolio(self.df)
        self.assertNotIn("OPEN_LONG_NEW_SIGNAL", set(cand.columns))

    # 10. WATCH semantics unchanged
    def test_watch_semantics(self):
        w = self.df[self.df["watch_status"] == "WATCH"]
        self.assertEqual(set(w["symbol"]), {"1104"})
        self.assertTrue((w["model_action"] == "WATCH").all())
        self.assertTrue((w["target_weight"] == 0).all())

    # 11. actual holding requiring action outranks unowned candidates
    def test_holding_priority(self):
        d = self.df.set_index(["symbol", "holding_side"])
        exit_long = d.loc[("1106", "LONG")]
        self.assertEqual(exit_long["user_action"], "EXIT_LONG")
        self.assertEqual(int(exit_long["priority_tier"]), 1)
        rank1_unowned = self.df[(self.df["symbol"] == "1103")].iloc[0]
        self.assertGreaterEqual(int(rank1_unowned["priority_tier"]), 3)
        # sorted output: tier-1 rows come first
        self.assertEqual(int(self.df.iloc[0]["priority_tier"]), 1)

    # 12/13. shorts supported; no short-entry action can exist
    def test_short_supported_no_short_entry(self):
        s = self.df[(self.df["symbol"] == "1101")
                    & (self.df["holding_side"] == "SHORT")]
        self.assertEqual(len(s), 1)
        s = s.iloc[0]
        self.assertEqual(s["user_action"], "BUY_TO_COVER")
        self.assertEqual(int(s["priority_tier"]), 1)
        for bad in ("OPEN_SHORT", "SELL_SHORT", "WATCH_SHORT"):
            self.assertFalse((self.df["user_action"] == bad).any())
            self.assertNotIn(bad, ur.ranking_md(self.df, self.meta))

    # 14. agreement is descriptive, never probability
    def test_agreement_descriptive(self):
        vals = set(self.df["model_agreement"]) - {""}
        self.assertTrue(vals <= {"HIGH", "NORMAL", "LOW"})
        md = ur.ranking_md(self.df, self.meta)
        for bad in ("probability", "獲利機率保證", "win rate", "勝率"):
            self.assertNotIn(bad, md)
        self.assertIn("非獲利機率", md)

    # 15/16. previous-session rank lookup; missing history unavailable
    def test_missing_history_unavailable(self):
        self.assertTrue(self.df["previous_rank"].isna().all())
        self.assertTrue(self.df["rank_change_1d"].isna().all())
        self.assertFalse((self.df["rank_change_1d"] == 0).any())

    def test_previous_rank_lookup(self):
        tmp2 = tempfile.mkdtemp(prefix="unirank2_")
        try:
            root2 = make_root(tmp2)
            out2 = os.path.join(tmp2, "reports", "user_actions")
            hist = os.path.join(out2, "history", PREV[:7])
            os.makedirs(hist, exist_ok=True)
            pd.DataFrame({"symbol": ["1101", "1115"],
                          "universe_rank": [15, 2]}).to_csv(
                os.path.join(hist, f"{PREV}_universe_ranking.csv"),
                index=False)
            df, meta = ur.build_ranking(
                root2, holdings_path=None, asof=DATE, out_dir=out2,
                configured=CONFIGURED)
            self.assertEqual(meta["prev_signal_date"], PREV)
            r = df.set_index("symbol")
            self.assertEqual(int(r.loc["1101", "previous_rank"]), 15)
            self.assertEqual(int(r.loc["1101", "rank_change_1d"]), 14)
            self.assertEqual(int(r.loc["1115", "rank_change_1d"]), -13)
            self.assertTrue(pd.isna(r.loc["1110", "previous_rank"]))
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    # 17/18. layout: dated in history/YYYY-MM, stable latest paths
    def test_report_layout(self):
        self.assertTrue(os.path.isfile(os.path.join(
            self.out, "history", DATE[:7],
            f"{DATE}_universe_ranking.csv")))
        self.assertTrue(os.path.isfile(os.path.join(
            self.out, "latest_universe_ranking.csv")))
        self.assertTrue(os.path.isfile(os.path.join(
            self.out, "latest_universe_ranking.md")))
        # nothing dated leaks into the user_actions root
        for f in os.listdir(self.out):
            if os.path.isfile(os.path.join(self.out, f)):
                self.assertTrue(f.startswith("latest_"), f)

    # held symbol outside the scored universe stays visible
    def test_held_outside_universe_visible(self):
        r = self.df[self.df["symbol"] == "0099"]
        self.assertEqual(len(r), 1)
        self.assertTrue(pd.isna(r.iloc[0]["universe_rank"]))
        self.assertEqual(r.iloc[0]["user_action"], "NO_MODEL_OPINION")

    # NO_MODEL_OPINION = manual review, never action-required (item 3)
    def test_no_model_opinion_priority_semantics(self):
        row = self.df[self.df["symbol"] == "0099"].iloc[0]
        # tier 2: prominent (above every unowned research name) ...
        self.assertEqual(int(row["priority_tier"]), 2)
        unowned = self.df[~self.df["is_actual_holding"]]
        self.assertTrue((unowned["priority_tier"] >= 3).all())
        # ... but never worded as a trade requirement
        self.assertIn("模型未涵蓋", row["priority_reason"])
        self.assertIn("人工檢視", row["priority_reason"])
        self.assertNotIn("需要減碼", row["priority_reason"])
        self.assertNotIn("需要交易", row["priority_reason"])
        # a validated EXIT/REDUCE holding outranks it
        exit_row = self.df[(self.df["symbol"] == "1106")].iloc[0]
        self.assertLess(int(exit_row["priority_tier"]),
                        int(row["priority_tier"]))
        # unowned rank #1-class candidate cannot outrank held risk rows
        best_unowned = unowned[pd.notna(unowned["universe_rank"])]
        best_unowned = best_unowned.loc[
            best_unowned["universe_rank"].idxmin()]
        self.assertGreater(int(best_unowned["priority_tier"]),
                           int(exit_row["priority_tier"]))
        # no new trading-action enums were invented anywhere
        self.assertTrue(set(self.df["user_action"])
                        <= set(__import__("holdings").USER_ACTIONS))

    # 19. simplified summary stays compact and clearly disclaimed
    def test_simplified_teaser_compact(self):
        top = ur.top_nonportfolio(self.df)
        self.assertLessEqual(len(top), 8)
        self.assertNotIn("1103", set(top["symbol"]))  # entry action
        self.assertNotIn("1101", set(top["symbol"]))  # held
        plan = pd.DataFrame([{
            "symbol": "1103", "user_action": "OPEN_LONG_NEW_SIGNAL",
            "ideal_zone_low": 98.5, "ideal_zone_high": 100.0,
            "acceptable_ceiling": 101.0}])
        meta = {"intended_execution_date": "2026-02-16",
                "signal_date": DATE}
        md = sr.night_summary_md(plan, meta, universe_top=top)
        self.assertIn("全市場強勢候選摘要", md)
        self.assertIn("尚非正式買進訊號", md)
        self.assertIn("latest_universe_ranking.md", md)
        self.assertLessEqual(len(md.splitlines()), 45)
        # and without ranking data the summary is unchanged in shape
        md0 = sr.night_summary_md(plan, meta)
        self.assertNotIn("全市場強勢候選摘要", md0)

    # signal-strength bands are pure rank-percentile mappings
    def test_signal_strength_bands(self):
        r = self.df.set_index("symbol")
        self.assertEqual(r.loc["1101", "signal_strength"], "TOP_TIER")
        self.assertEqual(r.loc["1104", "signal_strength"], "STRONG")
        self.assertEqual(r.loc["1110", "signal_strength"], "POSITIVE")
        self.assertEqual(r.loc["1120", "signal_strength"], "WEAK")


class TestLiveRankingContext(unittest.TestCase):

    def test_context_from_latest_csv(self):
        import refresh_execution_prices as rp
        tmp = tempfile.mkdtemp(prefix="unirankctx_")
        try:
            root = make_root(tmp)
            out = os.path.join(tmp, "reports", "user_actions")
            ur.build_ranking(root, holdings_path=os.path.join(
                root, "my_holdings.csv"), asof=DATE, out_dir=out,
                configured=CONFIGURED)
            plan = pd.DataFrame({"symbol": ["1103"]})
            plan_path = os.path.join(out, "plan.csv")
            plan.to_csv(plan_path, index=False)
            ctx = rp._ranking_context(plan_path, plan, states={})
            self.assertLessEqual(len(ctx), 5)
            syms = {c["symbol"] for c in ctx}
            self.assertNotIn("1103", syms)          # already in plan
            self.assertNotIn("1101", syms)          # held
            self.assertNotIn("1102", syms)          # portfolio member
            for c in ctx:
                self.assertIsNone(c["live_price"])  # no collector data
            # absent ranking file -> empty context, never an error
            ctx0 = rp._ranking_context(
                os.path.join(tmp, "nowhere", "plan.csv"), plan, {})
            self.assertEqual(ctx0, [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
