# Fast-reaction lead investigation

## Decision

Do not change the production moment estimator on this evidence. A two-parameter
chat-only candidate improves the 13 labeled moments in leave-one-out evaluation,
but a **50 ms shift in chat timestamps makes a slow-reaction clip start six
seconds after its event**. Leave-one-stream-out evaluation provides no gain.
The apparent improvement is not robust enough to deploy.

Base: requested main revision `4dce234`. At fetch time `origin/main` had advanced
to `fc38abb`; its detector and detector tests were identical to the requested
revision. No production files, dependencies, or shared configuration changed.

## Data and method

- Used all 13 timed rows from `~/eval/moment_truth.json`: eight impulseSV2 and
  five ItsFunneh. The four null-T rows were not treated as timing successes.
- Preserved the supplied T values and their uncertainty; did not relabel events
  to improve results. This is exploratory analysis on a very small cohort, with
  both near-zero-lag labels coming from the same stream.
- Extracted features at the existing selected peaks by instrumenting a copy of
  the production detector outside the repository. No video, transcript, T,
  stream ID, or absolute timestamp is an input to a candidate estimator.
- Rise is the existing contiguous baseline walk-back, not the returned estimated
  moment. Reaction-message ages use individually classified reaction messages in
  the preceding 30 seconds plus the peak bin, sorted by timestamp. These messages
  can react to unrelated earlier content. The second-message age requires a
  second distinct, casefolded author, not two messages from one author.
- Tested five threshold-rule families: rise age, normalized smoothed rise
  steepness, peak/pre-rise signal ratio, second-reacting-author age, and
  laughter-only first-reaction age. Each family fits a threshold and short delay
  (0, 5, or 10 seconds); otherwise it keeps the current estimator. Thresholds are
  on discrete grids, not individual timestamp cutpoints. Both inequality
  directions and the unchanged detector are candidates (217 configurations).
- In each leave-one-out (LOO) fold, repeated family, direction, threshold, and
  delay selection using only the other 12 rows. Ranked first by avoiding events
  before clip start, then 15–20s compliance, 12–25s compliance, fewer >25s leads,
  and absolute error from 18s. Leave-one-stream-out repeats selection using only
  the other stream. No stream-specific parameters are used.
- These are peak-matched timing folds, not an independent new-video benchmark.
  Subsequent full four-stream replay checks candidate selection and boundaries.
  The deployed 23s fallback is frozen, not recalibrated in each fold; its original
  calibration used these same 13 labels. LOO isolates selection of the additional
  fast-reaction rule, not the original fallback's calibration.

## Extracted chat-only features

All ages and absolute times are seconds. A negative reaction age means the
message arrives after the five-second peak bin's starting timestamp. Peak ratio
is raw peak reaction weight divided by max(0.75, mean weight in peak−90..−35s).
Steepness is `(smoothed_peak − smoothed_previous) / max(0.75, smoothed_peak)`.

| Stream | Peak | Rise | Peak−rise | Peak weight | Pre-rise mean | Ratio | Steepness | First reaction age | Second author age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Funneh | 635 | 610 | 25 | 30.75 | 12.48 | 2.46 | .34 | 29.918 | 29.835 |
| Funneh | 1435 | 1420 | 15 | 23 | 6.46 | 3.56 | .02 | 29.656 | 28.563 |
| Funneh | 1680 | 1680 | 0 | 23.5 | 7.54 | 3.12 | .36 | 26.941 | 24.763 |
| Funneh | 2020 | 1995 | 25 | 16 | 7.58 | 2.11 | .01 | 29.948 | 28.993 |
| Funneh | 2905 | 2895 | 10 | 23 | 9.48 | 2.43 | .18 | 29.316 | 29.097 |
| impulse | 1210 | 1205 | 5 | 2 | .08 | 2.67 | .50 | −2.324 | −2.390 |
| impulse | 2145 | 2145 | 0 | 2 | .08 | 2.67 | .40 | 24.111 | 7.963 |
| impulse | 4180 | 4180 | 0 | 2 | .17 | 2.67 | .58 | 18.005 | −.932 |
| impulse | 5015 | 5005 | 10 | 2 | .17 | 2.67 | .29 | 8.826 | 1.902 |
| impulse | 7335 | 7330 | 5 | 1 | 0 | 1.33 | 0 | 3.578 | 1.904 |
| impulse | 7400 | 7400 | 0 | 2 | .42 | 2.67 | .50 | 27.875 | 20.150 |
| impulse | 7820 | 7815 | 5 | 1 | .08 | 1.33 | 0 | 3.914 | 2.035 |
| impulse | 9645 | 9635 | 10 | 3 | .06 | 4 | .10 | 5.645 | 3.574 |

Abruptness alone does not identify fast video-to-chat reactions. The fast ads
quip and slow shard announcement have identical raw seven-bin reaction curves
`[0, 0, 0, 0, 0, 2, 1]`, equal rise age, equal peak ratio, and equal steepness,
despite true peak lags of −1s and 24s respectively.

## Best candidate — not deployed

If the second independently authored reaction in the look-back window arrives
0–2 seconds before the peak bin starts, estimate moment = peak (short delay 0).
Otherwise retain main's bounded `min(peak−23, rise_onset)` estimator. Thus the two
fitted parameters are the 2s threshold and the 0s fast delay. Do not additionally
clamp the fast branch to the old rise onset: that would retain over-leading.

The two parameters and family are identical in all 13 LOO fits. That stability
does not protect against bin-phase sensitivity.

### Per timed pick

Absolute timestamps and actual leads (`T − clip_start`) below are seconds.
"New" refers to the discarded candidate, not a shipped detector change.

| Stream | True T | Peak | Rise | Main moment | Candidate moment | Main lead | New lead | LOO lead | Stream-held-out lead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| impulse | 1178 | 1210 | 1205 | 1187 | 1187 | 9 | 9 | 9 | 9 |
| impulse | 2126 | 2145 | 2145 | 2122 | 2122 | 22 | 22 | 22 | 22 |
| impulse | 4156 | 4180 | 4180 | 4157 | 4157 | 17 | 17 | 17 | 17 |
| impulse | 5014 | 5015 | 5005 | 4992 | 5015 | 40 | 17 | 17 | 40 |
| impulse | 7336 | 7335 | 7330 | 7312 | 7335 | 42 | 19 | 19 | 42 |
| impulse | 7385 | 7400 | 7400 | 7377 | 7377 | 26 | 26 | 26 | 26 |
| impulse | 7796 | 7820 | 7815 | 7797 | 7797 | 17 | 17 | 17 | 17 |
| impulse | 9624 | 9645 | 9635 | 9622 | 9622 | 20 | 20 | 20 | 20 |
| Funneh | 615 | 635 | 610 | 610 | 610 | 23 | 23 | 23 | 23 |
| Funneh | 1412 | 1435 | 1420 | 1412 | 1412 | 18 | 18 | 18 | 18 |
| Funneh | 1652 | 1680 | 1680 | 1657 | 1657 | 13 | 13 | 13 | 13 |
| Funneh | 1983 | 2020 | 1995 | 1995 | 1995 | 6 | 6 | 6 | 6 |
| Funneh | 2877 | 2905 | 2895 | 2882 | 2882 | 13 | 13 | 13 | 13 |

### Compliance

Counts use inclusive ranges. "Before" means T < start, not an insufficient lead.

| Estimator/evaluation | Stream | n | 15–20s | 12–25s | >25s | Before | Median |
|---|---|---:|---:|---:|---:|---:|---:|
| Main | impulse | 8 | 3 | 4 | 3 | 0 | 21s |
| Main | Funneh | 5 | 1 | 4 | 0 | 0 | 13s |
| Main | Overall | 13 | 4 | 8 | 3 | 0 | 18s |
| Candidate in-sample | impulse | 8 | 5 | 6 | 1 | 0 | 18s |
| Candidate in-sample | Funneh | 5 | 1 | 4 | 0 | 0 | 13s |
| Candidate in-sample | Overall | 13 | 6 | 10 | 1 | 0 | 17s |
| Candidate LOO | impulse | 8 | 5 | 6 | 1 | 0 | 18s |
| Candidate LOO | Funneh | 5 | 1 | 4 | 0 | 0 | 13s |
| Candidate LOO | Overall | 13 | 6 | 10 | 1 | 0 | 17s |
| Candidate stream-held-out | Overall | 13 | 4 | 8 | 3 | 0 | 18s |

Other families' in-sample / LOO 15–20s counts: rise age 4/4, steepness 4/3,
peak ratio 4/3, laughter-only first age 6/5. Steepness and peak-ratio selection
each produce one event-before-start error in LOO. The unchanged detector is the
best full-data fit for the first three families under the safety-first objective.

## Disproving check: timestamp sensitivity

Reran the actual candidate detector, including binning, gates, outro filtering,
ranking, and clip selection, on cached chat with every timestamp increased by
0.05s. Video T was held fixed. The shard-announcement peak remains **7820**:

| Input | Second-author message | Age relative to peak | Estimated moment | Start | T−start |
|---|---:|---:|---:|---:|---:|
| Original chat | 7817.965 | 2.035 | 7797 | 7779 | 17s |
| Chat +0.05s | 7818.015 | 1.985 | 7820 | 7802 | **−6s** |

The truth uncertainty for this event is 7794.5–7796.5, so the miss is not explained
by event-time rounding. A +0.1s replay also misses this event. The fast examples'
second-author ages are 1.902 and 1.904: the classifier is separating opposite lag
regimes by just 0.131s of bin-relative phase. This is not a robust lag signal.

## Cached video checks

Reviewed existing video contact sheets plus their cached small.en transcripts;
this was not continuous audiovisual playback or a new frame-accurate annotation.
No downloads, credentials, or new transcription were needed.

- **5015:** `~/eval/dyrjzioAtNA/verify/5015.png` and matching transcript. Setup
  "I could totally make it" at 5008–5012, exclamation at 5013.88, "almost made
  it" from 5016.66. Retained T=5014. Candidate [4997,5035] gives 17s lead and
  contains the annotated payoff through 5023, plus peak+20. Cached video covers
  [4987,5035].
- **7335:** `~/eval/dyrjzioAtNA/verify/7335.png` and matching transcript. Mostly
  static tunnel while discussing internet routing; ads "not just fake news,
  actually helpful" is in 7332–7343. Retained T=7336. Candidate [7317,7355]
  gives 19s lead, contains payoff through 7340, and ends at peak+20. Cached
  video covers [7312,7355].
- **7820 counterexample:**
  `~/eval/dyrjzioAtNA/verify/moment_truth/7820.png` and matching transcript.
  Shard announcement at 7792.66–7797.26 precedes the later Warden joke. The
  jittered candidate's 7802 start misses this announcement. The [7760,7840]
  cached video covers both original and candidate windows.

Only 5015 and 7335 move by >10s in the unperturbed four-stream candidate replay;
both were checked above. No production estimate changes.

## Replay and reproducibility

Main and the discarded candidate each return 24 clips across all four streams,
with identical selected peaks (Speed 8, Funneh 8, impulse 8, Rivera 0). Every
clip satisfies start <= moment <= peak < end <= duration, start=max(0,moment−18),
duration 30–60s, and non-overlap. Both changed fast windows end at peak+tail.

Persistent offline artifacts under `~/eval/`:

- `fast_reaction_features.py` / `.json`: feature extraction and 13 feature rows.
- `fast_reaction_validate.py`, `fast_reaction_validation.json`: parameter grids,
  all fitting/tie-breaking code, per-fold models, timing tables and summaries.
- `fast_reaction_candidate.py`: isolated candidate detector; never imported by
  production code.
- `fast_reaction_sensitivity.py` / `.json`: full-detector timestamp perturbations.
- `fast_reaction_main.json` / `.txt`, `fast_reaction_candidate_eval.json` / `.txt`:
  full four-stream evaluations, including every window and existing proxy metrics.

Example replay command (from the repository virtualenv):

```
.venv/bin/python /home/ubuntu/eval/run_eval.py --detector fast_reaction_candidate:detect_spikes --output /home/ubuntu/eval/fast_reaction_candidate_eval.json
```

Next useful evidence is independently annotated fast and slow moments from more
streams, including similar-looking chat curves and different bin phases. A new
candidate should pass stream-held-out and timestamp-perturbation checks before
changing production timing. The owner's universal 15–20s lead requirement remains
unsolved; the existing 23s estimator is unchanged rather than replaced by a
fragile in-sample improvement.

Full baseline suite on the requested main revision (no production changes):

```
........................................................................ [ 19%]
........................................................................ [ 39%]
........................................................................ [ 58%]
........................................................................ [ 78%]
........................................................................ [ 97%]
........                                                                 [100%]
368 passed in 8.40s
```
