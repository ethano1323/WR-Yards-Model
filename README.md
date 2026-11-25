# Receiving Yards Projection Model (WR/TE), v3

A static web app (GitHub Pages) plus a free scheduled data job (GitHub Actions). It has two views:
- **Single player:** a full breakdown with every input editable.
- **Slate:** every WR/TE on the week's slate in one sortable table, ranked by EV.

## What's automated vs. manual

| Input | Source | Refresh |
|---|---|---|
| Season and last-4 yards per game, target shares, aDOT, games played (from snap counts) | nflverse | Tue / Thu / Sat / Sun |
| Team dropbacks and pass attempts; team dropbacks in each player's games | nflverse play-by-play | Tue / Thu / Sat / Sun |
| Teammates + injury status (#13), QB options (#7), opponent's starting secondary + injury status (#10) | nflverse rosters, injuries, snap counts | Tue / Thu / Sat / Sun |
| Spread, total, projected attempts (#11), opponent and league Y/A | nflverse + fitted model | Tue / Thu / Sat / Sun |
| Kickoff weather (#8) | Open-Meteo, fetched live in your browser | Each time you load a player or open the slate |
| Line and odds | The Odds API (free key), or "My own line" | Thu / Sat / Sun or manual |
| **Coverage rates, YPRR splits, routes vs. each coverage, total routes, overall YPRR** | **You, from FTN Stats IQ** | Remembered per defense (weekly) and per player |

## Model

```
Efficiency   = defense × DB injuries × QB × weather
YPG method   = Baseline YPG (season/L4 blend) × TS factor × (proj. attempts ÷ avg) × coverage × efficiency
Route method = Proj. dropbacks × route % × matchup YPRR × TS factor × efficiency          (#5)
                 route % = your total routes ÷ team dropbacks in his games (automated)
                 proj. dropbacks = proj. attempts × team dropbacks/attempts
Mean         = (1 − w) × YPG method + w × route method     (w = 50% default; route method needs total routes)
Coverage     = Σ rate × shrunk split YPRR ÷ overall YPRR                                      (#3)
                 shrunk split = (routes × split + k × overall) ÷ (routes + k)
Defense      = 1 + carry-over × (opp Y/A allowed ÷ league Y/A − 1)    (carry-over fitted by games played)
Fair line, P(Over) = historical outcome distribution by mean AND player type (aDOT)          (#1, #12)
```

**"Both" coverage view (default):** enter man/zone *and* single/two-high splits.
1. Each view gets its own shrunk coverage effect.
2. The two effects are blended by how unusual this defense is on each, measured in standard deviations (SD): weight ∝ |SD from league avg| + 0.1.
   - Team spread, 2023–25 FTN: man rate 40.2% ± 11.1 points; single-high rate 58.2% ± 7.0 points.
   - The two rates are correlated (r = 0.54), which is why they're blended rather than added.
   - Example: a 60%-man defense with an average shell is weighted 94% man/zone and 6% shell.
3. Once you've entered 16+ defenses for the week, their average becomes the league baseline.
4. Enter one side of each pair and the other fills in (zone = 100 − man). With total routes plus routes vs. one split, the other split's routes fill in too.

**Coverage weighting rule** (applies within each view):
1. If you entered routes for each split, those routes are used.
2. If you only entered total routes, routes per split are estimated from the league coverage mix (41% man / 59% zone, 58% single-high / 42% two-high).
3. If you entered no routes at all, the fallback weight (25%) is used.

## What the data says (fits refresh each run; split reliability fitted once)

| Finding | Number | Source |
|---|---|---|
| Routes needed before a split is 50% signal (k) | Man 608 · Zone 773 · Single-high 2,044 · Two-high 2,552 | FTN charting via nflverse, 2023–25, 244 WRs |
| Cover 0–6 splits | Cover 1 ≈ 609, Cover 2 ≈ 645, Cover 4 ≈ 489. Cover 0, 2-Man, 3 and 6 showed **no** detectable signal (treated as overall YPRR) | Same |
| What that means in practice | A WR with 150 routes vs. man gets about 20% weight on his man split | Same |
| Defense Y/A carry-over | 7% (3–5 games) → 20% (6–9) → 27% (10+) | nflverse 2018–25 |
| Missing starting CB / S | Opponent Y/A ×1.012 per CB, ×1.003 per S (small) | 4,252 games, 2018–25 |
| Fair line below mean | about 20% (<45 yds) · 12% (45–70) · 5–12% (70+, depending on player type) | 11,500 WR/TE games |
| Spread vs. pass attempts | About 0 once the game total and team average are known | 3,486 team games |
| Wind (Y/A) | 10–14 mph ×0.97 · 15–19 ×0.94 · 20+ ×0.93 · indoors ×1.02 | 2,646 games |

## Repo layout (must match exactly)

```
index.html
README.md
requirements.txt
data/data.json                      (rebuilt by the workflow)
data/fits_static.json               (coverage-split reliability; keep it)
scripts/build_data.py
scripts/fit_split_reliability.py    (one-time fit; the workflow doesn't run it)
.github/workflows/update-data.yml
```

## Weekly workflow

1. **Thursday onward:** check the top bar on the site for the current week and the props timestamp.
2. **Slate tab, enter coverage once per defense:** fill in man % and single-high % for each defense in "Defense coverage rates" (the other side of each pair auto-fills).
3. **Enter each player's FTN data in his row:** overall YPRR, his Man/Zone/1-Hi/2-Hi YPRR, total routes, and routes vs. Man and vs. 1-Hi (the other routes auto-fill).
4. **Sort by EV/$100.** Blue rows are +EV with complete FTN data. ⚠ marks an edge flagged as unusually large.
5. **Open a player** to check his teammates, defenders, QB and weather in detail.
6. **Save the bets:** click **Save +EV bets to log** on the slate, or **Save to log** on a single player.
7. **After the games:** enter actual yards in the log.

## Troubleshooting

| Problem | Fix |
|---|---|
| No "Update model data" in the Actions tab | The workflow must be at exactly `.github/workflows/update-data.yml` at the top level. Create it with **Add file → Create new file**. |
| Fails at **Check repo layout** | The red message names the missing file. Upload it at the path shown in the repo layout above. |
| Fails at **Install dependencies** | Re-upload `requirements.txt` from the zip. |
| Fails at **Build data/data.json** | Copy the last red lines from the step log. Usually it's a temporary download error, so re-run once. |
| Fails at **Commit updated data** (403) | Go to **Settings → Actions → General → Workflow permissions → Read and write permissions**. |
| "No data file found: manual mode" | Run the workflow, then hard-refresh. |
| Slate shows 0 players | Untick "Only players with a line" (props may not be posted yet) or lower "Min season YPG". |
| Props stopped late in the month | The credit reserve was reached. Use "My own line" until credits reset on the 1st. |

Data: nflverse (CC-BY 4.0; FTN charting data via nflverse, CC-BY-SA 4.0), The Odds API, Open-Meteo (CC-BY 4.0).
