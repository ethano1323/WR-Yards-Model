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
    """Ratio of actual yards to expected (prior-games mean), WR/TE, normalized so mean = 1."""
    d = (hist_ps.filter(pl.col("position").is_in(["WR", "TE"]) & (pl.col("season_type") == "REG"))
         .sort(["player_id", "season", "week"])
         .with_columns([
             pl.col("receiving_yards").fill_null(0).cast(pl.Float64).alias("y"),
         ])
         .with_columns([
             pl.col("y").cum_sum().shift(1).over(["player_id", "season"]).alias("cum"),
             pl.col("y").cum_count().shift(1).over(["player_id", "season"]).alias("n"),
         ])
         .filter(pl.col("n") >= 4)
         .with_columns((pl.col("cum") / pl.col("n")).alias("exp"))
         .filter(pl.col("exp") >= 20)
         .with_columns((pl.col("y") / pl.col("exp")).alias("ratio")))
    out = []
    for lo, hi, label in DIST_BUCKETS:
        x = d.filter((pl.col("exp") >= lo) & (pl.col("exp") < hi))["ratio"].to_numpy()
        x = x / x.mean()
        q = np.quantile(x, np.linspace(0, 1, 101))
        out.append({"lo": lo, "hi": hi, "label": label, "n": int(len(x)), "q": [r(v, 4) for v in q]})
        log(f"dist {label}: n={len(x)} median={q[50]:.3f} P(0 yds)={np.mean(x == 0):.3f}")
    return out


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

    # ---- pass catchers (all positions with targets; WR/TE selectable)
    tt = ts.select(["game_id", "team", pl.col("targets").alias("team_tgt")])
    pc = (ps.filter((pl.col("targets").fill_null(0) > 0) | pl.col("position").is_in(["WR", "TE"]))
            .join(tt, on=["game_id", "team"], how="left")
            .sort(["player_id", "week"]))
    players = {}
    for (pid,), grp in pc.group_by(["player_id"]):
        grp = grp.sort("week")
        last = grp.tail(1).row(0, named=True)
        l4 = grp.tail(4)
        tg, ttg = grp["targets"].fill_null(0).sum(), grp["team_tgt"].fill_null(0).sum()
        l4tg, l4ttg = l4["targets"].fill_null(0).sum(), l4["team_tgt"].fill_null(0).sum()
        p = {
            "name": last["player_display_name"], "pos": last["position"], "team": last["team"],
            "g": grp.height, "ypg": r(grp["receiving_yards"].fill_null(0).mean(), 1),
            "l4ypg": r(l4["receiving_yards"].fill_null(0).mean(), 1),
            "ts": r(tg / ttg * 100 if ttg else 0, 1), "l4ts": r(l4tg / l4ttg * 100 if l4ttg else 0, 1),
            "tgtPg": r(tg / grp.height, 1), "status": status.get(pid),
            "log": [[int(x["week"]), x["opponent_team"], int(x["targets"] or 0), int(x["receiving_yards"] or 0)]
                    for x in grp.tail(6).iter_rows(named=True)],
        }
        players[pid] = p
        if tg > 0 and p["team"] in teams:
            teams[p["team"]]["catchers"].append(
                {"id": pid, "name": p["name"], "pos": p["pos"], "ts": p["ts"], "l4ts": p["l4ts"], "status": p["status"]})
    for t in teams.values():
        t["catchers"] = sorted(t.get("catchers", []), key=lambda c: -c["ts"])[:12]
    log(f"players: {len(players)}")

    # ---- historical data for fits
    hist_seasons = list(range(FIT_START, season))
    hist_ps = nfl.load_player_stats(hist_seasons)
    hist_ts = nfl.load_team_stats(hist_seasons)
    hist_sched = nfl.load_schedules(hist_seasons)
    g = team_games(hist_ts, hist_sched)
    qb_prior = fit_qb_prior(hist_ps)
    fits = {"dist": fit_distribution(hist_ps), "passAtt": fit_pass_attempts(g), "wind": fit_wind(g),
            "qbPrior": qb_prior, "seasons": f"{FIT_START}-{season - 1}"}

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
