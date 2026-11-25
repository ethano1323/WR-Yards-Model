# Receiving Yards Projection Model (WR/TE)

A single-page web app that projects a WR/TE's receiving yards from inputs you enter yourself. It needs no paid data, no server and no build step: one `index.html` file.

## Formula

```
Projection = Season YPG
           × (Projected target share ÷ Season target share)
           × (Projected team pass attempts ÷ Season team pass attempts)      [optional]
           × [1 + w × (Matchup YPRR ÷ Overall YPRR − 1)]
           × [1 + w × (Opp yards/attempt allowed ÷ League yards/attempt − 1)] [optional]
```

- **Coverage views (pick one per projection):** Man/Zone, Single-high/Two-high, or Cover 0–6. Don't combine views, because they overlap and would double count.
- **Matchup YPRR** = Σ (defense coverage rate × player YPRR vs. that coverage). Coverage rates are scaled to 100%. A blank coverage row uses the player's overall YPRR.
- **w** is the *Matchup confidence* slider (default 50%). It shrinks the matchup effects toward neutral, because YPRR splits come from small samples.

## Inputs and free sources

| Input | Required? | Free source |
|---|---|---|
| Defense coverage rates (man/zone or single/two-high) | Yes | FTN Stats IQ (backup: Sharp Football Analysis → NFL Coverage Schemes) |
| Player YPRR vs. each coverage | Yes | FTN Stats IQ |
| Player overall YPRR | Yes | FTN Stats IQ (backup: SumerSports) |
| Season receiving YPG | Yes | Pro-Football-Reference player page |
| Season target share | Yes* | Player targets ÷ team pass attempts (both on Pro-Football-Reference) |
| Projected target share | Yes* | Your judgment (injuries, returns, depth chart) |
| Team pass attempts (season / projected) | Optional | Pro-Football-Reference team page; adjust the projection for the spread and total |
| Opp. Y/A allowed and league Y/A | Optional | Pro-Football-Reference → Team Defense → Passing (the league average row) |
| Sportsbook line | Yes | Any sportsbook |

\* Leave both target-share fields blank to skip the target-share adjustment.

## Deploy to GitHub Pages (about 5 minutes)

1. Go to github.com and click **New repository**. Name it, for example, `wr-yards-model`, set it to **Public**, and click **Create**.
2. On the repo page, click **Add file → Upload files**, drag in `index.html` and `README.md`, then click **Commit changes**.
3. Go to **Settings → Pages**. Under *Build and deployment*, set Source to **Deploy from a branch**, Branch to **main**, and folder to **/ (root)**, then click **Save**.
4. Wait 1–2 minutes and refresh. Your site is live at `https://<your-username>.github.io/wr-yards-model/`.
5. To update the site later, edit `index.html` on GitHub (pencil icon) or upload a new version. Pages redeploys automatically.

Saved inputs and the projection log are stored in your browser's localStorage. They stay on that device and browser, so use **Export CSV** to back them up.

## Calibrating

1. Save every projection to the log, then enter actual yards after the game.
2. Watch the **mean absolute error** and **bias** figures. If bias stays positive (the model runs low) over 20+ games, the baseline is too conservative.
3. Tune the **matchup confidence** slider. If raising it lowers your error over a full sample, the coverage splits are carrying real signal.
