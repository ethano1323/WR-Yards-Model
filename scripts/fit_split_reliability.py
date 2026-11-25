#!/usr/bin/env python3
"""
One-time fit (not run by the workflow): how many routes before a coverage YPRR split is
half signal / half noise? Uses free nflverse participation (FTN charting) + play-by-play.
Method: for each WR-season, randomly split routes into two halves, compute the split's
deviation from the player's overall YPRR in each half, correlate halves across players
(split-half reliability r), then k = n_half * (1 - r) / r  (Spearman-Brown).
Writes data/fits_static.json, which build_data.py merges into data.json.
"""
import json, os
import numpy as np, polars as pl, nflreadpy as nfl

SEASONS = [2023, 2024, 2025]
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "fits_static.json")
SINGLE = {"COVER_0", "COVER_1", "COVER_3"}
TWO = {"COVER_2", "2_MAN", "COVER_4", "COVER_6"}

part = nfl.load_participation(SEASONS).select(
    ["nflverse_game_id", "play_id", "offense_players", "offense_positions", "defense_man_zone_type", "defense_coverage_type"])
pbp = nfl.load_pbp(SEASONS).filter((pl.col("qb_dropback") == 1) & (pl.col("season_type") == "REG")).select(
    ["game_id", "play_id", "season", "defteam", "receiver_player_id", "receiving_yards"])
d = pbp.join(part, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="inner")
print("dropbacks with charting:", d.height)
print(d["defense_man_zone_type"].value_counts())
print(d["defense_coverage_type"].value_counts())

# team-level coverage tendencies: how far apart defenses are (for weighting Man/Zone vs Shell)
tm = (d.with_columns([
        pl.col("defense_man_zone_type").str.to_uppercase().str.contains("MAN").alias("man"),
        pl.col("defense_man_zone_type").is_in(["MAN_COVERAGE", "ZONE_COVERAGE"]).alias("mzknown"),
        pl.col("defense_coverage_type").is_in(list(SINGLE)).alias("single"),
        (pl.col("defense_coverage_type").is_in(list(SINGLE)) | pl.col("defense_coverage_type").is_in(list(TWO))).alias("shk")])
      .group_by(["season", "defteam"]).agg([
        (pl.col("man").filter(pl.col("mzknown")).mean()).alias("man_rate"),
        (pl.col("single").filter(pl.col("shk")).mean()).alias("single_rate")]))
team_rates = {
    "simple": {"mean": round(tm["man_rate"].mean(), 3), "sd": round(tm["man_rate"].std(), 3)},
    "shell": {"mean": round(tm["single_rate"].mean(), 3), "sd": round(tm["single_rate"].std(), 3)},
    "corr": round(float(np.corrcoef(tm["man_rate"], tm["single_rate"])[0, 1]), 3),
    "teamSeasons": tm.height,
}
print("team rates:", team_rates)

# explode to one row per WR on field (proxy for a route)
d = d.with_columns([pl.col("offense_players").str.split(";").alias("ids"),
                    pl.col("offense_positions").str.split(";").alias("pos")]).explode(["ids", "pos"])
d = d.filter(pl.col("pos") == "WR").with_columns(
    pl.when(pl.col("receiver_player_id") == pl.col("ids")).then(pl.col("receiving_yards").fill_null(0)).otherwise(0).alias("yds"))
d = d.with_columns([
    pl.col("defense_man_zone_type").str.to_uppercase().str.contains("MAN").alias("man"),
    pl.col("defense_coverage_type").is_in(list(SINGLE)).alias("single"),
    (pl.col("defense_coverage_type").is_in(list(SINGLE)) | pl.col("defense_coverage_type").is_in(list(TWO))).alias("shellknown"),
])
print("WR routes:", d.height)

def reliability(df, flag, valid=None, seeds=20, min_routes=300, min_split=20):
    rs, ns = [], []
    rng = np.random.default_rng(7)
    g = df if valid is None else df.filter(pl.col(valid))
    keep = g.group_by(["ids", "season"]).len().filter(pl.col("len") >= min_routes).select(["ids", "season"])
    g = g.join(keep, on=["ids", "season"])
    for s in range(seeds):
        h = g.with_columns(pl.Series("half", rng.integers(0, 2, g.height)))
        agg = h.group_by(["ids", "season", "half"]).agg([
            pl.col("yds").sum().alias("y"), pl.len().alias("n"),
            pl.col("yds").filter(pl.col(flag)).sum().alias("ys"), pl.col(flag).sum().alias("ns")])
        agg = agg.filter(pl.col("ns") >= min_split).with_columns(((pl.col("ys") / pl.col("ns")) - (pl.col("y") / pl.col("n"))).alias("dev"))
        w = agg.pivot(on="half", index=["ids", "season"], values=["dev", "ns"]).drop_nulls()
        if "dev_0" not in w.columns or "dev_1" not in w.columns or w.height < 30:
            return {"r_half": None, "routes_per_half": None, "k": None, "players": int(w.height)}
        r = np.corrcoef(w["dev_0"], w["dev_1"])[0, 1]
        rs.append(r); ns.append(float(np.mean(np.concatenate([w["ns_0"], w["ns_1"]]))))
    r, n = float(np.mean(rs)), float(np.mean(ns))
    k = n * (1 - r) / r if r > 0 else None
    return {"r_half": round(r, 3), "routes_per_half": round(n, 1), "k": round(k) if k else None, "players": int(w.height)}

out = {
    "man": reliability(d, "man"),
    "zone": reliability(d.with_columns((~pl.col("man")).alias("zone")), "zone"),
    "single": reliability(d, "single", valid="shellknown"),
    "two": reliability(d.with_columns((~pl.col("single")).alias("two")), "two", valid="shellknown"),
}
for cov in ["COVER_0", "COVER_1", "COVER_2", "2_MAN", "COVER_3", "COVER_4", "COVER_6"]:
    out[cov] = reliability(d.with_columns((pl.col("defense_coverage_type") == cov).alias("c")), "c", min_split=8)
ks = [out[x]["k"] for x in ["man", "zone"] if out[x]["k"]]
mz = d.filter(pl.col("defense_man_zone_type").is_in(["MAN_COVERAGE", "ZONE_COVERAGE"]))
sk = d.filter(pl.col("shellknown"))
ct = d["defense_coverage_type"].value_counts()
tot = ct["count"].sum()
cshare = {row["defense_coverage_type"]: row["count"] / tot for row in ct.iter_rows(named=True) if row["defense_coverage_type"]}
named = ["COVER_0", "COVER_1", "COVER_2", "2_MAN", "COVER_3", "COVER_4", "COVER_6"]
out["shares"] = {
    "simple": [round(mz["man"].mean(), 3), round(1 - mz["man"].mean(), 3)],
    "shell": [round(sk["single"].mean(), 3), round(1 - sk["single"].mean(), 3)],
    "detailed": [round(cshare.get(c, 0), 3) for c in named] + [round(1 - sum(cshare.get(c, 0) for c in named), 3)],
}
out["teamRates"] = team_rates
out["recommended_k"] = int(round(np.median(ks), -1))
out["seasons"] = f"{SEASONS[0]}-{SEASONS[-1]}"
print(json.dumps(out, indent=1))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"splitShrink": out}, open(OUT, "w"), indent=1)
