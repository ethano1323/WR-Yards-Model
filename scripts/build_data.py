#!/usr/bin/env python3
"""
Builds data/data.json for the Receiving Yards Projection Model.

Free sources only:
  - nflverse (via nflreadpy): player/team stats, schedules, spreads/totals, rosters, injuries
  - The Odds API (free key, optional): receiving-yards props (1 credit per game)

Also fits, from historical nflverse data:
  - outcome distribution (for P(Over) and fair line)
  - pass-attempt model from team average, spread and total
  - wind efficiency factors
  - backup-QB efficiency prior

Run locally:   pip install nflreadpy requests numpy && python scripts/build_data.py
Env vars:      ODDS_API_KEY (optional), FETCH_ODDS=1/0 (default 1)
"""
import datetime as dt
import json
import os
import re
import sys

import numpy as np
import polars as pl
import nflreadpy as nfl
import requests

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "data.json")
FIT_START = 2018          # first historical season used for model fits
QB_SHRINK_ATT = 200       # attempts of "prior" blended into each QB's Y/A
WIND_BUCKETS = [(0, 10, "0-9 mph"), (10, 15, "10-14 mph"), (15, 20, "15-19 mph"), (20, 99, "20+ mph")]
DIST_BUCKETS = [(0, 45, "<45"), (45, 70, "45-70"), (70, 999, "70+")]
ADOT_BUCKETS = [(-99, 8, "short (aDOT <8)"), (8, 12, "mid (aDOT 8-12)"), (12, 99, "deep (aDOT 12+)")]
CB_POS = {"CB", "CB/R", "DB", "DB/L"}
S_POS = {"S", "SS", "FS"}
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
STATIC_FITS = os.path.join(os.path.dirname(__file__), "..", "data", "fits_static.json")
ODDS_SPORT = "americanfootball_nfl"
ODDS_MARKET = "player_reception_yds"
CREDIT_RESERVE = 25       # stop pulling props when fewer credits than this remain

TEAM_NAMES = {  # The Odds API full names -> nflverse abbreviations
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def now_utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def log(*a):
    print("[build]", *a, flush=True)


def norm_name(s):
    s = (s or "").lower().replace(".", "").replace("'", "").replace("’", "").replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def r(x, n=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


# --------------------------------------------------------------------------------------
# Historical fits
# --------------------------------------------------------------------------------------
def fit_distribution(hist_ps):
    """Ratio of actual yards to expected (prior-games mean), WR/TE, normalized so mean = 1.
    Bucketed by expected yards and by player type (prior-games aDOT) -> #12."""
    d = (hist_ps.filter(pl.col("position").is_in(["WR", "TE"]) & (pl.col("season_type") == "REG"))
         .sort(["player_id", "season", "week"])
         .with_columns([pl.col("receiving_yards").fill_null(0).cast(pl.Float64).alias("y"),
                        pl.col("receiving_air_yards").fill_null(0).cast(pl.Float64).alias("ay"),
                        pl.col("targets").fill_null(0).cast(pl.Float64).alias("t")])
         .with_columns([
             pl.col("y").cum_sum().shift(1).over(["player_id", "season"]).alias("cum"),
             pl.col("y").cum_count().shift(1).over(["player_id", "season"]).alias("n"),
             pl.col("ay").cum_sum().shift(1).over(["player_id", "season"]).alias("cay"),
             pl.col("t").cum_sum().shift(1).over(["player_id", "season"]).alias("ct"),
         ])
         .filter(pl.col("n") >= 4)
         .with_columns([(pl.col("cum") / pl.col("n")).alias("exp"),
                        pl.when(pl.col("ct") >= 10).then(pl.col("cay") / pl.col("ct")).otherwise(None).alias("adot")])
         .filter(pl.col("exp") >= 20)
         .with_columns((pl.col("y") / pl.col("exp")).alias("ratio")))

    def pack(x, **meta):
        x = x / x.mean()
        q = np.quantile(x, np.linspace(0, 1, 101))
        return {**meta, "n": int(len(x)), "q": [r(v, 4) for v in q]}

    out = []
    for lo, hi, label in DIST_BUCKETS:
        mb = d.filter((pl.col("exp") >= lo) & (pl.col("exp") < hi))
        out.append(pack(mb["ratio"].to_numpy(), lo=lo, hi=hi, label=label, adLo=None, adHi=None, adLabel="all types"))
        for alo, ahi, alab in ADOT_BUCKETS:
            x = mb.filter((pl.col("adot") >= alo) & (pl.col("adot") < ahi))["ratio"].to_numpy()
            if len(x) >= 300:
                e = pack(x, lo=lo, hi=hi, label=label, adLo=alo, adHi=ahi, adLabel=alab)
                out.append(e)
                q = e["q"]
                log(f"dist {label} {alab}: n={e['n']} median={q[50]:.3f} p10={q[10]:.2f} p90={q[90]:.2f}")
    return out


def fit_def_weight(hist_ts):
    """How much of a defense's season-to-date Y/A-allowed edge carries into the next game.
    Slope of next-game ratio on prior-games ratio (both vs league), by games of sample."""
    lg = hist_ts.group_by("season").agg((pl.col("passing_yards").sum() / pl.col("attempts").sum()).alias("lg"))
    d = (hist_ts.filter(pl.col("season_type") == "REG").join(lg, on="season")
         .sort(["opponent_team", "season", "week"])
         .with_columns([
             pl.col("passing_yards").cast(pl.Float64).cum_sum().shift(1).over(["opponent_team", "season"]).alias("cy"),
             pl.col("attempts").cast(pl.Float64).cum_sum().shift(1).over(["opponent_team", "season"]).alias("ca"),
             pl.col("attempts").cum_count().shift(1).over(["opponent_team", "season"]).alias("n")])
         .filter((pl.col("n") >= 3) & (pl.col("attempts") >= 10))
         .with_columns([((pl.col("cy") / pl.col("ca")) / pl.col("lg") - 1).alias("x"),
                        ((pl.col("passing_yards") / pl.col("attempts")) / pl.col("lg") - 1).alias("y")]))
    out = []
    for lo, hi in [(0, 6), (6, 10), (10, 99)]:
        b = d.filter((pl.col("n") >= max(lo, 3)) & (pl.col("n") < hi))
        x, y, w = b["x"].to_numpy(), b["y"].to_numpy(), b["attempts"].to_numpy()
        slope = float(np.sum(w * x * y) / np.sum(w * x * x))
        out.append({"minG": lo, "maxG": hi - 1, "w": r(slope), "n": int(len(x))})
        log(f"defense weight games {lo}-{hi - 1}: {slope:.3f} (n={len(x)})")
    return out


def fit_db_injuries(hist_snaps, hist_ts):
    """Effect on opponent passing Y/A when a starting CB or S misses a game (#10)."""
    sn = (hist_snaps.filter(pl.col("game_type") == "REG")
          .with_columns(pl.col("team").replace(TEAM_FIX))
          .with_columns(pl.when(pl.col("position").is_in(list(CB_POS))).then(pl.lit("CB"))
                        .when(pl.col("position").is_in(list(S_POS))).then(pl.lit("S")).otherwise(None).alias("grp"))
          .filter(pl.col("grp").is_not_null()))
    played = sn.filter(pl.col("defense_snaps") > 0)
    starters = (played.group_by(["pfr_player_id", "season", "team", "grp"])
                .agg([pl.len().alias("gp"), pl.col("defense_pct").mean().alias("pct"),
                      pl.col("week").min().alias("w0"), pl.col("week").max().alias("w1")])
                .filter((pl.col("gp") >= 6) & (pl.col("pct") >= 0.6)))
    tg = hist_ts.filter(pl.col("season_type") == "REG").select(
        ["season", "week", pl.col("opponent_team").alias("team")]).unique()  # weeks each defense played
    exp_ = starters.join(tg, on=["season", "team"]).filter((pl.col("week") >= pl.col("w0")) & (pl.col("week") <= pl.col("w1")))
    miss = (exp_.join(played.select(["pfr_player_id", "season", "week"]).with_columns(pl.lit(1).alias("p")),
                      on=["pfr_player_id", "season", "week"], how="left")
            .filter(pl.col("p").is_null())
            .group_by(["season", "week", "team"]).agg([
                (pl.col("grp") == "CB").sum().alias("mCB"), (pl.col("grp") == "S").sum().alias("mS")]))
    t = hist_ts.filter(pl.col("season_type") == "REG")
    lg = t.group_by("season").agg((pl.col("passing_yards").sum() / pl.col("attempts").sum()).alias("lg"))
    dya = t.group_by(["season", "opponent_team"]).agg((pl.col("passing_yards").sum() / pl.col("attempts").sum()).alias("dya"))
    oya = t.group_by(["season", "team"]).agg((pl.col("passing_yards").sum() / pl.col("attempts").sum()).alias("oya"))
    d = (t.filter(pl.col("attempts") >= 10).join(lg, on="season").join(dya, on=["season", "opponent_team"])
         .join(oya, on=["season", "team"])
         .join(miss.rename({"team": "opponent_team"}), on=["season", "week", "opponent_team"], how="left")
         .with_columns([pl.col("mCB").fill_null(0), pl.col("mS").fill_null(0),
                        ((pl.col("passing_yards") / pl.col("attempts")) / (pl.col("dya") * pl.col("oya") / pl.col("lg"))).log().alias("y")]))
    X = np.column_stack([np.ones(d.height), d["mCB"].to_numpy(), d["mS"].to_numpy()])
    w = d["attempts"].to_numpy().astype(float)
    W = np.sqrt(w)
    b, *_ = np.linalg.lstsq(X * W[:, None], d["y"].to_numpy() * W, rcond=None)
    res = {"cb": r(np.exp(b[1]), 4), "s": r(np.exp(b[2]), 4), "n": int(d.height),
           "gamesMissingCB": int((d["mCB"] > 0).sum()), "gamesMissingS": int((d["mS"] > 0).sum())}
    log(f"DB injuries: per missing starting CB x{res['cb']}, per S x{res['s']} (n={res['n']})")
    return res


def team_games(hist_ts, hist_sched):
    """Team-game rows joined to schedule lines."""
    s = hist_sched.filter(pl.col("game_type") == "REG").select(
        ["game_id", "home_team", "away_team", "spread_line", "total_line", "roof", "wind"])
    t = hist_ts.filter(pl.col("season_type") == "REG").select(
        ["game_id", "season", "week", "team", "attempts", "passing_yards"])
    g = t.join(s, on="game_id", how="inner").with_columns(
        pl.when(pl.col("team") == pl.col("home_team")).then(pl.col("spread_line"))
          .otherwise(-pl.col("spread_line")).alias("team_spread"))
    return g


def fit_pass_attempts(g):
    """OLS: attempts ~ prior-games avg attempts + team spread (+ = favored) + game total.
    Fitted separately by how many prior games the average is built on (more games = more trust)."""
    d = (g.sort(["team", "season", "week"])
          .with_columns([
              pl.col("attempts").cast(pl.Float64).cum_sum().shift(1).over(["team", "season"]).alias("cum"),
              pl.col("attempts").cum_count().shift(1).over(["team", "season"]).alias("n"),
          ])
          .filter((pl.col("n") >= 3) & pl.col("spread_line").is_not_null() & pl.col("total_line").is_not_null())
          .with_columns((pl.col("cum") / pl.col("n")).alias("avg")))
    out = []
    for lo, hi in [(0, 6), (6, 10), (10, 99)]:
        s = d.filter((pl.col("n") >= max(lo, 3)) & (pl.col("n") < hi))
        X = np.column_stack([np.ones(s.height), s["avg"].to_numpy(), s["team_spread"].to_numpy(), s["total_line"].to_numpy()])
        y = s["attempts"].to_numpy().astype(float)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ b
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        log(f"pass att games {lo}-{hi - 1}: b0={b[0]:.2f} avg={b[1]:.3f} spread={b[2]:.3f} total={b[3]:.3f} R2={r2:.3f} n={len(y)}")
        out.append({"minG": lo, "maxG": hi - 1, "b0": r(b[0]), "bAvg": r(b[1], 4), "bSpread": r(b[2], 4),
                    "bTotal": r(b[3], 4), "r2": r(r2), "sd": r(np.std(y - pred), 2), "n": int(len(y))})
    return out


def fit_wind(g):
    """Passing yards per attempt vs. the team's own season Y/A, by wind bucket (outdoor games)."""
    season_ya = g.group_by(["team", "season"]).agg(
        (pl.col("passing_yards").sum() / pl.col("attempts").sum()).alias("sya"))
    d = g.join(season_ya, on=["team", "season"]).with_columns(
        (pl.col("attempts") * pl.col("sya")).alias("expy"))
    overall = d["passing_yards"].sum() / d["expy"].sum()
    out = {"indoor": None, "buckets": []}
    ind = d.filter(pl.col("roof").is_in(["dome", "closed"]))
    out["indoor"] = {"factor": r(ind["passing_yards"].sum() / ind["expy"].sum() / overall), "n": ind.height}
    outd = d.filter(pl.col("roof").is_in(["outdoors", "open"]) & pl.col("wind").is_not_null())
    for lo, hi, label in WIND_BUCKETS:
        b = outd.filter((pl.col("wind") >= lo) & (pl.col("wind") < hi))
        f = b["passing_yards"].sum() / b["expy"].sum() / overall
        out["buckets"].append({"lo": lo, "hi": hi, "label": label, "factor": r(f), "n": b.height})
        log(f"wind {label}: factor={f:.3f} n={b.height}")
    return out


def fit_qb_prior(hist_ps):
    """Aggregate Y/A of QB-seasons with 20-199 attempts (i.e., backups / spot starters)."""
    q = (hist_ps.filter(pl.col("position") == "QB")
         .group_by(["player_id", "season"]).agg([pl.col("attempts").sum(), pl.col("passing_yards").sum()])
         .filter((pl.col("attempts") >= 20) & (pl.col("attempts") < 200)))
    ya = q["passing_yards"].sum() / q["attempts"].sum()
    log(f"backup QB prior Y/A={ya:.2f} (n={q.height} QB-seasons)")
    return r(ya, 2)


# --------------------------------------------------------------------------------------
# Odds
# --------------------------------------------------------------------------------------
def fetch_odds(key, week_games):
    base = f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}"
    ev = requests.get(f"{base}/events", params={"apiKey": key}, timeout=30)
    ev.raise_for_status()
    wanted = {(g["away"], g["home"]) for g in week_games}
    props, remaining = {}, None
    for e in ev.json():
        pair = (TEAM_NAMES.get(e["away_team"]), TEAM_NAMES.get(e["home_team"]))
        if pair not in wanted:
            continue
        if remaining is not None and float(remaining) < CREDIT_RESERVE:
            log(f"odds: stopping, only {remaining} credits left (reserve {CREDIT_RESERVE})")
            break
        resp = requests.get(f"{base}/events/{e['id']}/odds", timeout=30, params={
            "apiKey": key, "regions": "us", "markets": ODDS_MARKET, "oddsFormat": "american"})
        remaining = resp.headers.get("x-requests-remaining", remaining)
        if resp.status_code in (401, 429):   # out of credits or bad key: stop, don't waste calls
            log("odds stopped:", resp.status_code, resp.text[:200])
            break
        if resp.status_code != 200:
            log("odds error", resp.status_code, resp.text[:200])
            continue
        for bk in resp.json().get("bookmakers", []):
            for m in bk.get("markets", []):
                if m["key"] != ODDS_MARKET:
                    continue
                sides = {}
                for o in m["outcomes"]:
                    k = (norm_name(o.get("description")), o.get("point"))
                    sides.setdefault(k, {})[o["name"].lower()] = o["price"]
                for (nm, pt), s in sides.items():
                    if pt is None or "over" not in s or "under" not in s:
                        continue
                    props.setdefault(nm, []).append(
                        {"book": bk["title"], "line": pt, "over": s["over"], "under": s["under"]})
    log(f"odds: {len(props)} players, credits remaining={remaining}")
    return props, remaining


# --------------------------------------------------------------------------------------
# Main build
# --------------------------------------------------------------------------------------
def main():
    season = nfl.get_current_season()
    log("season", season)

    # ---- schedule & current week
    sched = nfl.load_schedules(season)
    upcoming = sched.filter(pl.col("result").is_null())
    week = int(upcoming["week"].min()) if upcoming.height else int(sched["week"].max())
    wk = upcoming.filter(pl.col("week") == week)
    games = [{
        "id": g["game_id"], "away": g["away_team"], "home": g["home_team"], "gameday": g["gameday"],
        "gametime": g["gametime"], "spread": g["spread_line"], "total": g["total_line"],
        "roof": g["roof"], "location": g["location"], "stadium": g["stadium"],
    } for g in wk.iter_rows(named=True)]
    log(f"week {week}: {len(games)} upcoming games")

    # ---- current-season stats
    ps = nfl.load_player_stats(season).filter(pl.col("season_type").is_in(["REG", "POST"]))
    ts = nfl.load_team_stats(season).filter(pl.col("season_type").is_in(["REG", "POST"]))
    league_ya = ts["passing_yards"].sum() / ts["attempts"].sum()

    team_off = ts.group_by("team").agg([
        pl.len().alias("g"), pl.col("attempts").mean().alias("attPg"),
        (pl.col("passing_yards").sum() / pl.col("attempts").sum()).alias("ya")])
    team_def = ts.group_by("opponent_team").agg(
        (pl.col("passing_yards").sum() / pl.col("attempts").sum()).alias("oppYA"))
    teams = {}
    for t in team_off.iter_rows(named=True):
        teams[t["team"]] = {"g": t["g"], "attPg": r(t["attPg"], 1), "ya": r(t["ya"], 2), "qbs": [], "catchers": []}
    for t in team_def.iter_rows(named=True):
        teams.setdefault(t["opponent_team"], {})["oppYA"] = r(t["oppYA"], 2)

    # ---- injuries (current week)
    status = {}
    try:
        inj = nfl.load_injuries(season).filter(pl.col("week") == week)
        for row in inj.iter_rows(named=True):
            if row["report_status"]:
                status[row["gsis_id"]] = row["report_status"]
        log(f"injuries: {len(status)} with a game status")
    except Exception as e:  # injuries can lag early in the week
        log("injuries unavailable:", e)

    # ---- roster status: IR/PUP/NFI/suspended ("RES") and released players never appear on the
    #      weekly injury report, so take them from the roster and let them override the report.
    ROSTER_OUT = {"RES": "IR", "CUT": "Released", "RLS": "Released", "RET": "Released", "EXE": "Released"}
    try:
        ra = nfl.load_rosters(season)
        if "week" in ra.columns:
            ra = ra.sort("week").group_by("gsis_id").agg(pl.all().last())
        n_res = 0
        for x in ra.iter_rows(named=True):
            lab = ROSTER_OUT.get(x["status"])
            if lab and x["gsis_id"]:
                status[x["gsis_id"]] = lab
                n_res += 1
        log(f"roster: {n_res} players on IR/reserve or released")
    except Exception as e:
        log("roster status unavailable:", e)

    # ---- games played from snap counts (so 0-catch games count), dropbacks from play-by-play
    idmap = nfl.load_players().select(["gsis_id", "pfr_id"]).drop_nulls()
    snaps = nfl.load_snap_counts(season).filter(pl.col("game_type").is_in(["REG", "WC", "DIV", "CON", "SB"]))
    pbp = nfl.load_pbp(season).filter(pl.col("season_type").is_in(["REG", "POST"]))
    dbk = (pbp.filter(pl.col("posteam").is_not_null()).group_by(["game_id", "posteam"])
              .agg(pl.col("qb_dropback").fill_null(0).sum().alias("team_db")).rename({"posteam": "team"}))
    for t in dbk.group_by("team").agg(pl.col("team_db").mean()).iter_rows(named=True):
        if t["team"] in teams:
            teams[t["team"]]["dbPg"] = r(t["team_db"], 1)
    sched_wk = sched.select(["game_id", "week", "home_team", "away_team"])
    snap_games = (snaps.filter(pl.col("offense_snaps") > 0)
                  .join(idmap, left_on="pfr_player_id", right_on="pfr_id")
                  .select([pl.col("gsis_id").alias("player_id"), "game_id", "team"]))
    stat_games = ps.select(["player_id", "game_id", "team"])
    pg = (pl.concat([stat_games, snap_games], how="vertical_relaxed").unique(["player_id", "game_id"], keep="first")
          .join(ps.select(["player_id", "game_id", "targets", "receiving_yards", "receiving_air_yards"]),
                on=["player_id", "game_id"], how="left")
          .join(ts.select(["game_id", "team", pl.col("targets").alias("team_tgt")]), on=["game_id", "team"], how="left")
          .join(dbk, on=["game_id", "team"], how="left")
          .join(sched_wk, on="game_id", how="left")
          .with_columns(pl.when(pl.col("team") == pl.col("home_team")).then(pl.col("away_team"))
                        .otherwise(pl.col("home_team")).alias("opp"))
          .with_columns([pl.col(c).fill_null(0) for c in ["targets", "receiving_yards", "receiving_air_yards", "team_tgt", "team_db"]])
          .sort(["player_id", "week"]))
    meta = (ps.sort("week").group_by("player_id").agg([pl.col("player_display_name").last(), pl.col("position").last(),
                                                       pl.col("targets").fill_null(0).sum().alias("tot_tgt")]))
    meta = {m["player_id"]: m for m in meta.iter_rows(named=True)
            if m["position"] in ("WR", "TE") or m["tot_tgt"] > 0}
    # weeks each team has played (to measure how much of a player's absence is already in teammates' stats)
    team_weeks = {x["team"]: sorted(x["week"]) for x in ts.group_by("team").agg(pl.col("week")).iter_rows(named=True)}
    players = {}
    for (pid,), grp in pg.filter(pl.col("player_id").is_in(list(meta))).group_by(["player_id"]):
        grp = grp.sort("week")
        m, last, l4 = meta[pid], grp.tail(1).row(0, named=True), grp.tail(4)
        tg, ttg = grp["targets"].sum(), grp["team_tgt"].sum()
        l4tg, l4ttg = l4["targets"].sum(), l4["team_tgt"].sum()
        p = {
            "name": m["player_display_name"], "pos": m["position"], "team": last["team"],
            "g": grp.height, "ypg": r(grp["receiving_yards"].mean(), 1), "l4ypg": r(l4["receiving_yards"].mean(), 1),
            "ts": r(tg / ttg * 100 if ttg else 0, 1), "l4ts": r(l4tg / l4ttg * 100 if l4ttg else 0, 1),
            "tgtPg": r(tg / grp.height, 1), "adot": r(grp["receiving_air_yards"].sum() / tg, 1) if tg >= 5 else None,
            "teamDb": int(grp["team_db"].sum()), "status": status.get(pid),
            "missed": 0, "missedL4": 0,
            "log": [[int(x["week"]), x["opp"], int(x["targets"]), int(x["receiving_yards"])]
                    for x in grp.tail(6).iter_rows(named=True)],
        }
        tw = team_weeks.get(p["team"], [])
        mine = set(grp.filter(pl.col("team") == p["team"])["week"].to_list())
        p["missed"] = sum(1 for w in tw if w not in mine) if mine else 0
        p["missedL4"] = sum(1 for w in tw[-4:] if w not in mine) if mine else 0
        players[pid] = p
        if tg > 0 and p["team"] in teams:
            teams[p["team"]]["catchers"].append(
                {"id": pid, "name": p["name"], "pos": p["pos"], "ts": p["ts"], "l4ts": p["l4ts"], "status": p["status"],
                 "missed": p["missed"], "missedL4": p["missedL4"]})
    for t in teams.values():
        t["catchers"] = sorted(t.get("catchers", []), key=lambda c: -c["ts"])[:12]
    log(f"players: {len(players)}")

    # ---- defensive starters (CB/S) with injury status (#10)
    pfr2gsis = {x["pfr_id"]: x["gsis_id"] for x in idmap.iter_rows(named=True)}
    dsn = (snaps.with_columns(pl.when(pl.col("position").is_in(list(CB_POS))).then(pl.lit("CB"))
                              .when(pl.col("position").is_in(list(S_POS))).then(pl.lit("S")).otherwise(None).alias("grp"))
           .filter(pl.col("grp").is_not_null() & (pl.col("defense_snaps") > 0)))
    team_games_n = {t: v["g"] for t, v in teams.items()}
    dstat = (dsn.group_by(["pfr_player_id", "player", "team", "grp"])
             .agg([pl.len().alias("gp"), pl.col("defense_pct").mean().alias("pct"), pl.col("week").max().alias("lastwk")])
             .filter((pl.col("gp") >= 2) & (pl.col("pct") >= 0.5)).sort("pct", descending=True))
    for t in teams:
        rows = dstat.filter(pl.col("team") == t).head(7)
        teams[t]["dbs"] = [{"name": x["player"], "pos": x["grp"], "pct": r(x["pct"] * 100, 0), "gp": x["gp"],
                            "missed": max(0, team_games_n.get(t, x["gp"]) - x["gp"]),
                            "status": status.get(pfr2gsis.get(x["pfr_player_id"]))} for x in rows.iter_rows(named=True)]

    # ---- historical data for fits
    hist_seasons = list(range(FIT_START, season))
    hist_ps = nfl.load_player_stats(hist_seasons)
    hist_ts = nfl.load_team_stats(hist_seasons)
    hist_sched = nfl.load_schedules(hist_seasons)
    g = team_games(hist_ts, hist_sched)
    qb_prior = fit_qb_prior(hist_ps)
    hist_snaps = nfl.load_snap_counts(hist_seasons)
    fits = {"dist": fit_distribution(hist_ps), "passAtt": fit_pass_attempts(g), "wind": fit_wind(g),
            "qbPrior": qb_prior, "defWeight": fit_def_weight(hist_ts), "dbInj": fit_db_injuries(hist_snaps, hist_ts),
            "seasons": f"{FIT_START}-{season - 1}"}
    if os.path.exists(STATIC_FITS):   # coverage-split shrinkage (#3), fitted once by fit_split_reliability.py
        fits.update(json.load(open(STATIC_FITS)))

    # ---- QBs: roster QBs, Y/A over last 3 seasons shrunk toward backup prior
    q3 = (pl.concat([hist_ps.filter(pl.col("season") >= season - 2).select(ps.columns), ps], how="vertical_relaxed")
            .filter(pl.col("attempts").fill_null(0) > 0)
            .group_by("player_id").agg([pl.col("attempts").sum(), pl.col("passing_yards").sum()]))
    q3 = {x["player_id"]: x for x in q3.iter_rows(named=True)}
    cur_att = (ps.filter(pl.col("attempts").fill_null(0) > 0).group_by(["team", "player_id"])
                 .agg(pl.col("attempts").sum()).sort("attempts", descending=True))
    rost = nfl.load_rosters(season).filter((pl.col("position") == "QB") & (pl.col("status") == "ACT"))
    for t, tm in teams.items():
        ids = rost.filter(pl.col("team") == t)["gsis_id"].to_list()
        starters = cur_att.filter(pl.col("team") == t)
        inc = starters["player_id"][0] if starters.height else None
        if inc and inc not in ids:
            ids.append(inc)
        qbs = []
        for qid in ids:
            s = q3.get(qid, {"attempts": 0, "passing_yards": 0})
            name = rost.filter(pl.col("gsis_id") == qid)["full_name"]
            name = name[0] if name.len() else players.get(qid, {}).get("name") or next(
                (x["player_display_name"] for x in ps.filter(pl.col("player_id") == qid).head(1).iter_rows(named=True)), qid)
            ya = (s["passing_yards"] + QB_SHRINK_ATT * qb_prior) / (s["attempts"] + QB_SHRINK_ATT)
            qbs.append({"id": qid, "name": name, "att3y": int(s["attempts"]), "ya": r(ya, 2),
                        "status": status.get(qid), "incumbent": qid == inc})
        tm["qbs"] = sorted(qbs, key=lambda x: (not x["incumbent"], -x["att3y"]))

    # ---- odds (optional)
    prev = {}
    if os.path.exists(OUT_PATH):
        try:
            prev = json.load(open(OUT_PATH))
        except Exception:
            prev = {}
    key = os.environ.get("ODDS_API_KEY", "").strip()
    fetch = os.environ.get("FETCH_ODDS", "1") == "1"
    same_week = prev.get("meta", {}).get("week") == week and prev.get("meta", {}).get("season") == season
    odds = dict(prev.get("odds", {})) if same_week else {}       # start from this week's earlier pull
    odds_meta = dict(prev.get("meta", {}).get("odds", {})) if same_week else {}
    if key and fetch:
        try:
            new, remaining = fetch_odds(key, games)
            odds.update(new)                                      # fresh lines overwrite, missing players keep earlier lines
            if new:
                odds_meta["updated"] = now_utc()
            if remaining is not None:
                odds_meta["creditsRemaining"] = remaining
        except Exception as e:
            log("odds fetch failed:", e)
    log(f"odds: {len(odds)} players available")

    data = {
        "meta": {"generated": now_utc(), "season": season,
                 "week": week, "odds": odds_meta, "source": "nflverse (CC-BY), The Odds API"},
        "league": {"ya": r(league_ya, 2)},
        "fits": fits, "games": games, "teams": teams, "players": players, "odds": odds,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    log(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024:.0f} KB)")


if __name__ == "__main__":
    sys.exit(main())
