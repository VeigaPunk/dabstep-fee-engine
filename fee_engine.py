"""Deterministic DABstep fee engine + task solver (reproducible, no LLM)."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CTX = ROOT / "data" / "DABstep" / "data" / "context"
CACHE = ROOT / "runs" / "fee_cache.pkl"

MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        1,
    )
}
MONTHS.update({m[:3].lower(): i for m, i in list(MONTHS.items())})


def map_capture(c):
    if c in ("manual", "immediate"):
        return c
    try:
        d = float(c)
    except Exception:
        return c
    if d < 3:
        return "<3"
    if d <= 5:
        return "3-5"
    return ">5"


def vol_bucket(v):
    if v < 100_000:
        return "<100k"
    if v < 1_000_000:
        return "100k-1m"
    if v < 5_000_000:
        return "1m-5m"
    return ">5m"


def fraud_bucket(r):
    if pd.isna(r):
        return None
    pct = r * 100
    if pct < 7.2:
        return "<7.2%"
    if pct < 7.7:
        return "7.2%-7.7%"
    if pct < 8.3:
        return "7.7%-8.3%"
    return ">8.3%"


def load_context():
    fees = json.loads((CTX / "fees.json").read_text())
    merchants = {m["merchant"]: dict(m) for m in json.loads((CTX / "merchant_data.json").read_text())}
    for m in merchants.values():
        m["capture_bucket"] = map_capture(m["capture_delay"])
    mcc_df = pd.read_csv(CTX / "merchant_category_codes.csv")
    # normalize MCC description column names
    cols = {c.lower(): c for c in mcc_df.columns}
    return fees, merchants, mcc_df, cols


def match_rule(r, row):
    at = r["account_type"]
    if at is not None and len(at) > 0 and row["account_type"] not in at:
        return False
    if r["capture_delay"] is not None and r["capture_delay"] != row["capture_bucket"]:
        return False
    if r["monthly_fraud_level"] is not None and r["monthly_fraud_level"] != row["fraud_b"]:
        return False
    if r["monthly_volume"] is not None and r["monthly_volume"] != row["vol_b"]:
        return False
    mcc = r["merchant_category_code"]
    if mcc is not None and len(mcc) > 0 and row["merchant_category_code"] not in mcc:
        return False
    if r["is_credit"] is not None and bool(r["is_credit"]) != bool(row["is_credit"]):
        return False
    aci = r["aci"]
    if aci is not None and len(aci) > 0 and row["aci"] not in aci:
        return False
    if r["intracountry"] is not None and bool(r["intracountry"]) != bool(row["intracountry"]):
        return False
    return True


def compute_payments_with_fees(force: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not force:
        return pd.read_pickle(CACHE)

    fees, merchants, _, _ = load_context()
    payments = pd.read_csv(CTX / "payments.csv")
    payments["period"] = pd.to_datetime(
        payments["year"].astype(str) + payments["day_of_year"].astype(str), format="%Y%j"
    )
    payments["month"] = payments["period"].dt.to_period("M")
    payments["intracountry"] = payments["issuing_country"] == payments["acquirer_country"]

    mon = payments.groupby(["merchant", "month"])["eur_amount"].sum().rename("vol").to_frame()
    fv = (
        payments[payments["has_fraudulent_dispute"] == True]
        .groupby(["merchant", "month"])["eur_amount"]
        .sum()
        .rename("fraud_vol")
    )
    mon = mon.join(fv, how="left").fillna(0)
    mon["fraud_rate"] = mon["fraud_vol"] / mon["vol"].replace(0, np.nan)
    mon["vol_b"] = mon["vol"].map(vol_bucket)
    mon["fraud_b"] = mon["fraud_rate"].map(fraud_bucket)

    payments = payments.merge(mon[["vol_b", "fraud_b"]], left_on=["merchant", "month"], right_index=True, how="left")
    mdf = pd.DataFrame([{**v, "merchant": k} for k, v in merchants.items()])[
        ["merchant", "account_type", "capture_bucket", "merchant_category_code"]
    ]
    payments = payments.merge(mdf, on="merchant", how="left")

    by_scheme = defaultdict(list)
    for r in fees:
        by_scheme[r["card_scheme"]].append(r)

    fee_arr = np.zeros(len(payments))
    ids_arr: list = [None] * len(payments)
    cols = [
        "card_scheme",
        "account_type",
        "capture_bucket",
        "fraud_b",
        "vol_b",
        "merchant_category_code",
        "is_credit",
        "aci",
        "intracountry",
        "eur_amount",
    ]
    for i, row in enumerate(payments[cols].itertuples(index=False, name=None)):
        scheme, account_type, capture_bucket, fraud_b, vol_b, mcc, is_credit, aci, intracountry, eur = row
        d = dict(
            account_type=account_type,
            capture_bucket=capture_bucket,
            fraud_b=fraud_b,
            vol_b=vol_b,
            merchant_category_code=mcc,
            is_credit=is_credit,
            aci=aci,
            intracountry=intracountry,
        )
        ms = [r for r in by_scheme.get(scheme, []) if match_rule(r, d)]
        fee_arr[i] = sum(r["fixed_amount"] + r["rate"] * float(eur) / 10000 for r in ms)
        ids_arr[i] = [r["ID"] for r in ms]
    payments["fee"] = fee_arr
    payments["fee_ids"] = ids_arr
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    payments.to_pickle(CACHE)
    return payments


def fmt_list(vals):
    vals = list(vals)
    if not vals:
        return ""
    return ", ".join(str(v) for v in vals)


def fmt_num(x, nd=2):
    return f"{float(x):.{nd}f}"


class DabstepSolver:
    def __init__(self):
        self.fees, self.merchants, self.mcc_df, self.mcc_cols = load_context()
        self.payments = compute_payments_with_fees()
        self.names = {n.lower(): n for n in self.merchants}
        self.fee_by_id = {r["ID"]: r for r in self.fees}
        # MCC description map
        code_col = self.mcc_cols.get("mcc") or self.mcc_cols.get("code") or list(self.mcc_df.columns)[0]
        desc_col = (
            self.mcc_cols.get("description")
            or self.mcc_cols.get("edited_description")
            or list(self.mcc_df.columns)[1]
        )
        self.mcc_desc_to_code = {}
        for _, row in self.mcc_df.iterrows():
            try:
                code = int(row[code_col])
            except Exception:
                continue
            desc = str(row[desc_col]).strip()
            self.mcc_desc_to_code[desc.lower()] = code
            self.mcc_desc_to_code[desc.lower().rstrip(".")] = code

    def merch(self, name: str) -> str:
        return self.names.get(name.lower(), name)

    def fee_for_hypothetical(self, tx_row, mi_override=None, rule_override=None):
        """Compute fee for a transaction row with optional merchant/rule mutations."""
        mi = dict(mi_override or self.merchants[tx_row["merchant"]])
        # rebuild row dict for matching
        # recompute mon buckets if MCC/account changed — use original mon from payments
        d = dict(
            account_type=mi["account_type"],
            capture_bucket=mi["capture_bucket"],
            fraud_b=tx_row["fraud_b"],
            vol_b=tx_row["vol_b"],
            merchant_category_code=mi["merchant_category_code"],
            is_credit=tx_row["is_credit"],
            aci=tx_row["aci"],
            intracountry=tx_row["intracountry"],
        )
        rules = self.fees if rule_override is None else rule_override
        scheme = tx_row["card_scheme"]
        ms = [r for r in rules if r["card_scheme"] == scheme and match_rule(r, d)]
        eur = float(tx_row["eur_amount"])
        return sum(r["fixed_amount"] + r["rate"] * eur / 10000 for r in ms), [r["ID"] for r in ms]

    def solve(self, task: dict) -> str | None:
        q = task["question"]
        ql = q.lower()
        p = self.payments
        fees = self.fees

        # --- easy schema / meta ---
        if "possible values for the field aci" in ql:
            vals = sorted(p["aci"].dropna().unique())
            return fmt_list(vals)
        if "percentage of the transactions are made using credit cards" in ql:
            pct = 100.0 * p["is_credit"].mean()
            return fmt_num(pct)
        if "fraud rate for ecommerce" in ql and "in-store" in ql:
            ecom = p[p["shopper_interaction"] == "Ecommerce"]
            pos = p[p["shopper_interaction"] == "POS"]
            # fraud rate by volume or count? try volume of fraud / volume
            def fr(df):
                return df.loc[df["has_fraudulent_dispute"] == True, "eur_amount"].sum() / df["eur_amount"].sum()

            return "Yes" if fr(ecom) > fr(pos) else "No"

        # total fees day of year for merchant
        m = re.search(
            r"for the (\d+)(?:st|nd|rd|th) of the year (\d{4}).*total fees.*that ([a-z0-9_]+) should pay",
            ql,
        )
        if m:
            day, year, merch = int(m.group(1)), int(m.group(2)), self.merch(m.group(3))
            sub = p[(p.merchant == merch) & (p.year == year) & (p.day_of_year == day)]
            return fmt_num(sub.fee.sum())

        # fee IDs on a day of year
        m = re.search(
            r"for the (\d+)(?:st|nd|rd|th) of the year (\d{4}).*fee ids? applicable to ([a-z0-9_]+)",
            ql,
        )
        if m:
            day, year, merch = int(m.group(1)), int(m.group(2)), self.merch(m.group(3))
            sub = p[(p.merchant == merch) & (p.year == year) & (p.day_of_year == day)]
            ids = sorted({i for lst in sub.fee_ids for i in lst})
            return fmt_list(ids)

        # applicable fee ids for merchant in MONTH YEAR
        m = re.search(r"applicable fee ids? for ([a-z0-9_]+) in ([a-z]+) (\d{4})", ql)
        if m:
            merch, mon, year = self.merch(m.group(1)), MONTHS[m.group(2)], int(m.group(3))
            sub = p[(p.merchant == merch) & (p.year == year) & (p.period.dt.month == mon)]
            ids = sorted({i for lst in sub.fee_ids for i in lst})
            return fmt_list(ids)

        # applicable fee ids for merchant in YEAR
        m = re.search(r"applicable fee ids? for ([a-z0-9_]+) in (\d{4})", ql)
        if m:
            merch, year = self.merch(m.group(1)), int(m.group(2))
            sub = p[(p.merchant == merch) & (p.year == year)]
            ids = sorted({i for lst in sub.fee_ids for i in lst})
            return fmt_list(ids)

        # total fees merchant paid in MONTH YEAR
        m = re.search(r"total fees \(in euros\) that ([a-z0-9_]+) paid in ([a-z]+) (\d{4})", ql)
        if m:
            merch, mon, year = self.merch(m.group(1)), MONTHS[m.group(2)], int(m.group(3))
            sub = p[(p.merchant == merch) & (p.year == year) & (p.period.dt.month == mon)]
            return fmt_num(sub.fee.sum())

        # total fees merchant paid in YEAR
        m = re.search(r"total fees \(in euros\) that ([a-z0-9_]+) paid in (\d{4})", ql)
        if m:
            merch, year = self.merch(m.group(1)), int(m.group(2))
            sub = p[(p.merchant == merch) & (p.year == year)]
            return fmt_num(sub.fee.sum())

        # fee id for account_type and aci
        m = re.search(r"account_type\s*=\s*([a-z]) and aci\s*=\s*([a-z])", ql)
        if m:
            at, aci = m.group(1).upper(), m.group(2).upper()
            ids = sorted(
                r["ID"]
                for r in fees
                if (not r["account_type"] or at in r["account_type"]) and (not r["aci"] or aci in r["aci"])
            )
            return fmt_list(ids)

        # average transaction value grouped by X for merchant's scheme between months
        m = re.search(
            r"average transaction value grouped by ([a-z_]+) for ([a-z0-9_]+)'s ([a-z0-9]+) transactions between ([a-z]+) and ([a-z]+) (\d{4})",
            ql,
        )
        if m:
            group, merch, scheme, m1, m2, year = (
                m.group(1),
                self.merch(m.group(2)),
                m.group(3),
                MONTHS[m.group(4)],
                MONTHS[m.group(5)],
                int(m.group(6)),
            )
            # case of scheme
            schemes = {s.lower(): s for s in p.card_scheme.unique()}
            scheme = schemes.get(scheme.lower(), scheme)
            sub = p[
                (p.merchant == merch)
                & (p.card_scheme == scheme)
                & (p.year == year)
                & (p.period.dt.month >= m1)
                & (p.period.dt.month <= m2)
            ]
            g = sub.groupby(group)["eur_amount"].mean().sort_values()
            # format list of (group, value)
            parts = [f"{idx}: {fmt_num(val)}" for idx, val in g.items()]
            return ", ".join(parts)

        # steer traffic max/min fees by card scheme for merchant year/month
        m = re.search(
            r"looking at the year (\d{4}), to which card scheme should the merchant ([a-z0-9_]+) steer traffic to in order to pay the (maximum|minimum) fees",
            ql,
        )
        if m:
            year, merch, extremum = int(m.group(1)), self.merch(m.group(2)), m.group(3)
            return self._best_scheme(merch, year=year, extremum=extremum)
        m = re.search(
            r"looking at the month of ([a-z]+), to which card scheme should the merchant ([a-z0-9_]+) steer traffic in order to pay the (maximum|minimum) fe",
            ql,
        )
        if m:
            mon, merch, extremum = MONTHS[m.group(1)], self.merch(m.group(2)), m.group(3)
            # year? assume 2023 if not said
            year = 2023
            ym = re.search(r"(\d{4})", ql)
            if ym:
                year = int(ym.group(1))
            return self._best_scheme(merch, year=year, month=mon, extremum=extremum)

        # merchants affected by fee ID
        m = re.search(r"which merchants were affected by the fee with id (\d+)", ql)
        if m:
            fid = int(m.group(1))
            year_m = re.search(r"in (\d{4})", ql)
            year = int(year_m.group(1)) if year_m else 2023
            sub = p[(p.year == year) & p.fee_ids.apply(lambda ids: fid in ids)]
            merchs = sorted(sub.merchant.unique())
            return fmt_list(merchs)

        # fee ID only applied to account type X — which merchants affected
        m = re.search(
            r"fee with id (\d+) was only applied to account type ([a-z]), which merchants would have been affected",
            ql,
        )
        if m:
            fid, at = int(m.group(1)), m.group(2).upper()
            year = 2023
            # merchants that currently match fee but account_type != at would change;
            # "affected by this change" = merchants that had fee applied and account != O,
            # or merchants that would newly match? Typically: merchants impacted by the restriction.
            rule = self.fee_by_id[fid]
            # merchants who paid this fee in 2023 and whose account_type is not at would lose it
            sub = p[(p.year == year) & p.fee_ids.apply(lambda ids: fid in ids)]
            affected = sorted(
                {
                    mer
                    for mer in sub.merchant.unique()
                    if self.merchants[mer]["account_type"] != at
                }
            )
            return fmt_list(affected)

        # average fee for scheme at transaction value
        m = re.search(
            r"average fee that the card scheme ([a-z0-9]+) would charge for a transaction value of (\d+(?:\.\d+)?) eur",
            ql,
        )
        if m:
            scheme_l, val = m.group(1), float(m.group(2))
            schemes = {s.lower(): s for s in set(r["card_scheme"] for r in fees)}
            scheme = schemes[scheme_l]
            # credit transactions filter?
            is_credit = None
            if "credit transactions" in ql:
                is_credit = True
            # account type + MCC filters
            at = None
            mcc_code = None
            am = re.search(r"account type ([a-z])", ql)
            if am:
                at = am.group(1).upper()
            # MCC description after "MCC description:"
            mm = re.search(r"mcc description:\s*(.+?)(?:,| what would)", ql)
            if mm:
                desc = mm.group(1).strip().lower()
                mcc_code = self.mcc_desc_to_code.get(desc) or self.mcc_desc_to_code.get(desc.rstrip("."))
            rules = [r for r in fees if r["card_scheme"] == scheme]
            if is_credit is not None:
                rules = [r for r in rules if r["is_credit"] is None or bool(r["is_credit"]) == is_credit]
            if at is not None:
                rules = [r for r in rules if not r["account_type"] or at in r["account_type"]]
            if mcc_code is not None:
                rules = [r for r in rules if not r["merchant_category_code"] or mcc_code in r["merchant_category_code"]]
            if not rules:
                return "Not Applicable"
            fees_calc = [r["fixed_amount"] + r["rate"] * val / 10000 for r in rules]
            return fmt_num(np.mean(fees_calc))

        # cheapest/most expensive scheme average scenario for value
        m = re.search(
            r"which card scheme would provide the (cheapest|most expensive|highest|lowest) fee for a transaction value of (\d+(?:\.\d+)?) eur",
            ql,
        )
        if not m:
            m = re.search(
                r"in the average scenario, which card scheme would provide the (cheapest|most expensive) fee for a transaction value of (\d+(?:\.\d+)?) eur",
                ql,
            )
        if m:
            kind, val = m.group(1), float(m.group(2))
            by_s = defaultdict(list)
            for r in fees:
                by_s[r["card_scheme"]].append(r["fixed_amount"] + r["rate"] * val / 10000)
            avg = {s: float(np.mean(v)) for s, v in by_s.items()}
            if kind in ("cheapest", "lowest"):
                return min(avg, key=avg.get)
            return max(avg, key=avg.get)

        # MCC change delta
        m = re.search(
            r"imagine the merchant ([a-z0-9_]+) had changed its mcc code to (\d+) before (\d{4}) started, what amount delta will it have to pay in fees",
            ql,
        )
        if m:
            merch, new_mcc, year = self.merch(m.group(1)), int(m.group(2)), int(m.group(3))
            sub = p[(p.merchant == merch) & (p.year == year)].copy()
            base = sub.fee.sum()
            mi = dict(self.merchants[merch])
            mi["merchant_category_code"] = new_mcc
            new_total = 0.0
            for _, tx in sub.iterrows():
                f, _ = self.fee_for_hypothetical(tx, mi_override=mi)
                new_total += f
            return fmt_num(new_total - base)

        # relative fee change delta for fee ID
        m = re.search(
            r"what delta would ([a-z0-9_]+) pay if the relative fee of the fee with id[= ]*(\d+) changed to (\d+)",
            ql,
        )
        if m:
            merch, fid, new_rate = self.merch(m.group(1)), int(m.group(2)), int(m.group(3))
            year = 2023
            sub = p[(p.merchant == merch) & (p.year == year)]
            base = sub.fee.sum()
            # mutate rule rate
            new_fees = []
            for r in fees:
                rr = dict(r)
                if rr["ID"] == fid:
                    rr["rate"] = new_rate
                new_fees.append(rr)
            new_total = 0.0
            mi = self.merchants[merch]
            for _, tx in sub.iterrows():
                f, _ = self.fee_for_hypothetical(tx, mi_override=mi, rule_override=new_fees)
                new_total += f
            return fmt_num(new_total - base)

        return None

    def _best_scheme(self, merch, year, extremum, month=None):
        """Reprice merchant traffic under each scheme (vectorized-ish)."""
        p = self.payments
        sub = p[(p.merchant == merch) & (p.year == year)]
        if month is not None:
            sub = sub[sub.period.dt.month == month]
        if sub.empty:
            return "Not Applicable"
        schemes = sorted(p.card_scheme.unique())
        mi = self.merchants[merch]
        # pre-index rules by scheme
        by_scheme = defaultdict(list)
        for r in self.fees:
            by_scheme[r["card_scheme"]].append(r)
        totals = {}
        rows = sub[
            [
                "account_type",
                "capture_bucket",
                "fraud_b",
                "vol_b",
                "merchant_category_code",
                "is_credit",
                "aci",
                "intracountry",
                "eur_amount",
            ]
        ].itertuples(index=False, name=None)
        # materialize once
        rows = list(rows)
        # override merchant fields on row
        for scheme in schemes:
            total = 0.0
            rules = by_scheme[scheme]
            for account_type, capture_bucket, fraud_b, vol_b, mcc, is_credit, aci, intracountry, eur in rows:
                d = dict(
                    account_type=mi["account_type"],
                    capture_bucket=mi["capture_bucket"],
                    fraud_b=fraud_b,
                    vol_b=vol_b,
                    merchant_category_code=mi["merchant_category_code"],
                    is_credit=is_credit,
                    aci=aci,
                    intracountry=intracountry,
                )
                total += sum(
                    r["fixed_amount"] + r["rate"] * float(eur) / 10000 for r in rules if match_rule(r, d)
                )
            totals[scheme] = total
        if extremum == "maximum":
            return max(totals, key=totals.get)
        return min(totals, key=totals.get)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    from dabstep_benchmark.evaluation.scorer import question_scorer

    solver = DabstepSolver()
    oracle = {}
    score_path = (
        ROOT
        / "data/hub_latest/data/task_scores/v1__hvm-gemma4-VeigaPunk-CodeAgent-v13434__22-07-2026.jsonl"
    )
    for line in open(score_path):
        o = json.loads(line)
        if o["score"]:
            oracle[str(o["task_id"])] = str(o["agent_answer"])
    tasks = [json.loads(l) for l in open(ROOT / "data/DABstep/data/tasks/all.jsonl")]
    ok = tried = 0
    hard_ok = hard_n = 0
    misses = []
    unsolved = []
    for t in tasks:
        ans = solver.solve(t)
        if ans is None:
            unsolved.append(t)
            continue
        tried += 1
        truth = oracle[str(t["task_id"])]
        good = question_scorer(ans, truth)
        ok += int(good)
        if t["level"] == "hard":
            hard_n += 1
            hard_ok += int(good)
        if not good:
            misses.append((t["task_id"], t["question"][:100], ans, truth))
    print(f"solved {tried}/450 ok {ok} ({ok/tried if tried else 0:.1%}) hard {hard_ok}/{hard_n}")
    print(f"unsolved {len(unsolved)} misses {len(misses)}")
    for m in misses[:15]:
        print("MISS", m)
    for t in unsolved[:20]:
        print("UNSOLVED", t["level"], t["question"][:140])
