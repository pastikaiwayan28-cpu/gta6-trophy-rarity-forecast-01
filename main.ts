// gta6-trophy-rarity-forecast-01 entry (Node 22, standard library only): a dilution-aware
// hierarchical log-time forecaster for Exophase trophy earn rates.
//
// The rate of a trophy is earners / players-tracked. When Exophase refreshes, the tracked
// player base jumps (launch night: 638 -> 3,868) and every rate but the 100% one falls, because
// the new players have not earned as much yet. So the program is built around that dilution:
//
//   * players: a robust (Theil-Sen) line of log(players) against ln(1 + hours since release),
//     read at the target, so the denominator grows the way the history says it grows;
//   * the median rate: a robust (Theil-Sen) line of logit(rate) against ln(1 + hours
//     since release), shrunk toward its tier's mean slope, read at the target. The rate
//     itself already carries the dilution (earners over players), and fitting its logit
//     directly is steadier than forecasting earners and players apart: the two growth
//     curves bend the same way in log-time, so their separate extrapolations drift
//     apart while the rate's own line stays put. The players trend is still fitted, to
//     size the dilution term of the spread.
//
// When the history is flat (every read identical, the page not yet refreshed), no trend can be
// fit, so a dilution prior takes over: the tracked base is stale and a refresh is overdue, so
// the program assumes it grows by a launch catch-up factor over the stale span plus the
// horizon, while earners grow slower (a trophy's earner growth exponent rises with its rate:
// common trophies are earned by newcomers almost as fast as by the first players, rare ones
// barely at all). The median rate is pulled down by that dilution, and the spread is widened
// so the quantiles stay honest about a refresh jump.
//
// The spread is honest by construction: a fit-residual term, a random-walk term that grows
// with the square root of the horizon, a dilution term that grows with the forecast player
// growth, and a staleness term when the history is flat. Quantiles are the median shifted on
// the logit scale by the standard normal's quantiles times that spread.
//
// The spread is ASYMMETRIC: wider below the median than above it. A rate is far more likely
// to land below the median than above it — the tracked base keeps growing, so dilution can
// always pull a rate down further, while a rate above the median needs earners to outgrow
// players, which the first week rarely shows. So the lower quantiles use 1.15x the spread
// and the upper quantiles 0.60x: the band stays honest about a refresh jump without paying
// for width where the answer almost never goes.
//
// It never crashes and never emits a bad shape: any failure in a case falls back to the
// persistence forecast (last read's rates, random-walk spread). Deterministic: no randomness,
// fixed key order, fixed rounding, so two runs write byte-identical files.
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Random-walk spread, logit units per square-root day (the house baseline's term). */
const SIGMA_HORIZON = 0.5;
/** Dilution spread per log unit of forecast player growth. */
const SIGMA_DILUTION = 0.25;
/** Extra spread when the history is flat (a refresh jump is unmodelled). */
const SIGMA_STALE = 0.7;
/** Prior doubling time, hours, of the tracked base during launch catch-up. */
const PRIOR_DOUBLING_HOURS = 2;
/** Cap on the prior player-growth factor. */
const PRIOR_GROWTH_CAP = 12;
/** Spread multiplier below the median (dilution can always pull a rate down further). */
const SPREAD_LO = 1.15;
/** Spread multiplier above the median (earners rarely outgrow players in week one). */
const SPREAD_HI = 0.6;
/** Shrinkage strength: a trophy's own slope weighs n / (n + K) against its tier's. */
const SHRINK_K = 3;

interface Trophy {
  id: number;
  name: string;
  tier: string;
  points: number;
  secret: boolean;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { release_date: string };
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const clampPct = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
/** Median of a list of numbers. */
const median = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/**
 * Theil-Sen slope and intercept for y against x: the median pairwise slope, and the median
 * of y - slope * x. Robust to the flat stretches and step jumps of a page that only changes
 * when Exophase refreshes. Returns null when fewer than 2 distinct x values carry data.
 */
function theilSen(xs: number[], ys: number[]): { slope: number; intercept: number } | null {
  const pts: Array<[number, number]> = [];
  for (let i = 0; i < xs.length; i++) {
    if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) pts.push([xs[i], ys[i]]);
  }
  if (pts.length < 2) return null;
  const slopes: number[] = [];
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      const dx = pts[j][0] - pts[i][0];
      if (dx !== 0) slopes.push((pts[j][1] - pts[i][1]) / dx);
    }
  }
  if (slopes.length === 0) return null;
  const slope = median(slopes);
  const intercept = median(pts.map(([x, y]) => y - slope * x));
  return { slope, intercept };
}

/** Hours from an ISO instant to another (b - a), never negative. */
const hoursBetween = (a: string, b: string): number =>
  Math.max(0, (Date.parse(b) - Date.parse(a)) / 3_600_000);

/** A trophy's earner-growth exponent: how fast its earners grow relative to players. */
function earnerBeta(ratePct: number, tier: string): number {
  const base = 0.25 + 0.65 * (ratePct / (ratePct + 2));
  const penalty = tier === 'platinum' ? 0.15 : tier === 'gold' ? 0.1 : tier === 'silver' ? 0.05 : 0;
  return clamp(base - penalty, 0.15, 0.95);
}

/** Forecast one case into {"trophies":[{"id","q":[q05,q25,q50,q75,q95]}, ...]}. */
function forecast(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  const release = c.game?.release_date ? `${c.game.release_date}T00:00:00Z` : c.history[0]?.at ?? c.at;
  const horizonDays = c.horizon_minutes / 1440;
  const xTarget = Math.log(1 + hoursBetween(release, c.target));

  const n = c.trophies.length;
  // Last non-null rate per trophy, and the last read's players.
  const lastRate: number[] = new Array(n).fill(NaN);
  let lastPlayers = NaN;
  for (let j = c.history.length - 1; j >= 0; j--) {
    const h = c.history[j];
    if (Number.isFinite(h.players_tracked) && h.players_tracked > 0 && !Number.isFinite(lastPlayers)) {
      lastPlayers = h.players_tracked;
    }
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(lastRate[i]) && h.rates[i] !== null && Number.isFinite(h.rates[i])) {
        lastRate[i] = h.rates[i] as number;
      }
    }
  }
  if (!Number.isFinite(lastPlayers)) lastPlayers = 638;
  // Tier fallback rates when a trophy has no read at all.
  const tierFallback = (tier: string): number =>
    tier === 'platinum' ? 0.5 : tier === 'gold' ? 1 : 1;
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(lastRate[i])) lastRate[i] = tierFallback(c.trophies[i].tier);
  }

  // Collapse the history into distinct states: consecutive reads of an unchanged page carry
  // no information, and their flat stretches would drag every pairwise slope to zero. A
  // state is the players count plus the whole rate vector, kept at its first observed time.
  interface State { x: number; players: number; rates: Array<number | null> }
  const states: State[] = [];
  for (const h of c.history) {
    const x = Math.log(1 + hoursBetween(release, h.at));
    const prev = states[states.length - 1];
    const same =
      prev !== undefined &&
      prev.players === h.players_tracked &&
      prev.rates.length === h.rates.length &&
      prev.rates.every((r, i) => r === h.rates[i]);
    if (!same) states.push({ x, players: h.players_tracked, rates: h.rates });
  }
  const flat = states.length <= 1;

  // --- Players at the target. ---
  let playersTarget = lastPlayers;
  let logGrowth = 0;
  if (!flat) {
    const fit = theilSen(
      states.map((s) => s.x),
      states.map((s) => Math.log(Math.max(1, s.players))),
    );
    if (fit !== null) {
      const slope = 0.9 * fit.slope; // weak shrinkage toward no growth
      playersTarget = Math.exp(fit.intercept + slope * xTarget);
    }
  }
  playersTarget = clamp(playersTarget, lastPlayers, lastPlayers * 200);
  logGrowth = Math.log(playersTarget / lastPlayers);

  // --- Logit-rate trend at the target, per trophy, with tier shrinkage. ---
  const logitSlope: number[] = new Array(n).fill(0);
  const logitIntercept: number[] = new Array(n).fill(0);
  const logitHasFit: boolean[] = new Array(n).fill(false);
  if (!flat) {
    for (let i = 0; i < n; i++) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const s of states) {
        const r = s.rates[i];
        if (r !== null && Number.isFinite(r)) {
          xs.push(s.x);
          ys.push(logit(r));
        }
      }
      const fit = theilSen(xs, ys);
      if (fit !== null) {
        logitSlope[i] = fit.slope;
        logitIntercept[i] = fit.intercept;
        logitHasFit[i] = true;
      } else {
        logitIntercept[i] = logit(lastRate[i]);
      }
    }
    // Tier mean slopes, shrunk toward the global mean slope.
    const tiers = ['platinum', 'gold', 'silver', 'bronze'];
    const globalSlope = median(logitSlope.filter((_, i) => logitHasFit[i]));
    const tierSlope: Record<string, number> = {};
    for (const tier of tiers) {
      const ss = logitSlope.filter((_, i) => logitHasFit[i] && c.trophies[i].tier === tier);
      const m = ss.length > 0 ? median(ss) : globalSlope;
      tierSlope[tier] = (ss.length * m + 2 * globalSlope) / (ss.length + 2);
    }
    for (let i = 0; i < n; i++) {
      if (!logitHasFit[i]) continue;
      const w = states.length / (states.length + SHRINK_K);
      logitSlope[i] = w * logitSlope[i] + (1 - w) * tierSlope[c.trophies[i].tier];
    }
  }

  // --- Fit residual spread (pooled logit residuals around the fitted trends). ---
  let fitSd = 0;
  if (!flat) {
    const residuals: number[] = [];
    for (let i = 0; i < n; i++) {
      if (!logitHasFit[i]) continue;
      for (const s of states) {
        const r = s.rates[i];
        if (r === null || !Number.isFinite(r)) continue;
        residuals.push(logit(r) - (logitIntercept[i] + logitSlope[i] * s.x));
      }
    }
    if (residuals.length >= 10) {
      const m = median(residuals);
      fitSd = clamp(Math.sqrt(median(residuals.map((r) => (r - m) ** 2)) * 1.4826 ** 2), 0, 0.6);
    }
  }

  // --- Median rate and spread per trophy. ---
  const rows: Array<{ id: number; q: number[] }> = [];
  const stalenessH = c.history.length > 0 ? hoursBetween(c.history[0].at, c.at) : 0;
  for (let i = 0; i < n; i++) {
    const t = c.trophies[i];
    const r = lastRate[i];
    let medianRate: number;
    let spread: number;
    if (flat) {
      // Dilution prior: the stale tracked base catches up over the stale span plus the
      // horizon, earners growing slower than players by the trophy's exponent.
      // This branch also covers a saturated trophy (r = 100): every tracked player has
      // it, so its rate can only fall as new players join, and r * growth ** (beta - 1)
      // is exactly that dilution. A trophy pinned at 100% must never be forecast at
      // 100% — the moment the base grows, its rate is below 100%.
      const growth = Math.min(PRIOR_GROWTH_CAP, 2 ** ((stalenessH + c.horizon_minutes / 60) / PRIOR_DOUBLING_HOURS));
      const beta = earnerBeta(r, t.tier);
      medianRate = r * growth ** (beta - 1);
      spread = Math.sqrt(
        (SIGMA_HORIZON * Math.sqrt(horizonDays)) ** 2 +
          (SIGMA_DILUTION * Math.log(growth)) ** 2 +
          SIGMA_STALE ** 2,
      );
    } else {
      // Trend median: the fitted logit line read at the target.
      medianRate = invLogitPct(logitIntercept[i] + logitSlope[i] * xTarget);
      if (r >= 100) {
        // Saturated trophy: its logit history is pinned at the clamp, so the fitted
        // line is flat at ~100% and says nothing about the future. The rate can only
        // fall as the base grows, so take the dilution model instead: the forecast
        // player growth pulled down by the trophy's earner exponent.
        const beta = earnerBeta(100, t.tier);
        medianRate = 100 * Math.exp(logGrowth * (beta - 1));
      }
      spread = Math.sqrt(
        fitSd ** 2 +
          (SIGMA_HORIZON * Math.sqrt(horizonDays)) ** 2 +
          (SIGMA_DILUTION * Math.abs(logGrowth)) ** 2,
      );
    }
    medianRate = clamp(medianRate, 0.01, 99.99);
    spread = clamp(spread, 0.35, 2.0);
    const l = logit(medianRate);
    // Asymmetric spread: wider below the median, narrower above it.
    const q = Z.map((z) =>
      round4(clamp(invLogitPct(l + z * spread * (z < 0 ? SPREAD_LO : SPREAD_HI)), 0.01, 100)),
    );
    // Enforce non-decreasing after rounding.
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    rows.push({ id: t.id, q });
  }
  return { trophies: rows };
}

/** Persistence fallback: last read's rates with a random-walk spread (always valid). */
function fallback(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  const last = c.history[c.history.length - 1];
  const spread = SIGMA_HORIZON * Math.sqrt(c.horizon_minutes / 1440);
  const trophies = c.trophies.map((t, i) => {
    const r = last?.rates[i] ?? 1;
    const l = logit(r);
    const q = Z.map((z) =>
      round4(clamp(invLogitPct(l + z * spread * (z < 0 ? SPREAD_LO : SPREAD_HI)), 0.01, 100)),
    );
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    return { id: t.id, q };
  });
  return { trophies };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  let out: { trophies: Array<{ id: number; q: number[] }> };
  try {
    if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
      throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}`);
    }
    if (!c.history || c.history.length === 0) throw new Error(`${file} has no history`);
    out = forecast(c);
  } catch {
    out = fallback(c);
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
