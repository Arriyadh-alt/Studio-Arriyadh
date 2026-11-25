"""Editorial estimates and optional published-performance calibration."""

import json
import math
import os
from pathlib import Path


DEFAULT_WEIGHTS = dict(hook=.20, story=.20, emotion=.20, visual=.15, payoff=.15, retention=.10)
DEFAULT_PENALTIES = dict(missing_trigger=9, misleading_hook=12, incomplete_conversation=8,
                         static_footage=8, repetitive=10, poor_framing=8, abrupt_ending=8,
                         low_value_scene=12)
SCORE_FIELDS = ("scene", "trigger", "reaction", "needs_manual_review", "hooks", "selected_hook",
                "hook_anchors", "hook_quote_spans", "score_breakdown", "score_penalties", "score_explanation", "score_tier")


def score_tier(score):
    if score >= 85:
        return "high_priority"
    if score >= 70:
        return "review"
    if score >= 55:
        return "needs_improvement"
    return "hidden"


def _number(value, maximum):
    return type(value) in (int, float) and 0 <= value <= maximum and math.isfinite(value)


def _weights(values, defaults):
    if not isinstance(values, dict):
        return defaults
    merged = {k: values.get(k, v) for k, v in defaults.items()}
    if not all(_number(v, 100) for v in merged.values()) or not sum(merged.values()):
        return defaults
    total = sum(merged.values())
    return {k: v / total for k, v in merged.items()}


def scoring_config(data_dir=None):
    root = Path(data_dir or os.environ.get("CLIPPER_DATA_DIR", "~/clipper-data")).expanduser()
    try:
        config = json.loads((root / "viral-scoring.json").read_text())
    except (OSError, ValueError):
        config = {}
    if not isinstance(config, dict):
        config = {}
    weights = _weights(config.get("weights"), DEFAULT_WEIGHTS)
    overrides = config.get("penalties")
    penalties = dict(DEFAULT_PENALTIES)
    if isinstance(overrides, dict):
        penalties.update({k: v for k, v in overrides.items() if k in penalties and _number(v, 100)})
    try:
        from clipper_studio.feedback import calibrated_weights

        weights = _weights(calibrated_weights(dict(weights)), weights)
    except Exception:
        pass
    return weights, penalties


def editorial_score(components, flags, explanation, config=None):
    weights, penalties = config or scoring_config()
    breakdown = {k: round(max(0, min(100, components.get(k, 0)))) for k in DEFAULT_WEIGHTS}
    deductions = {k: penalties[k] for k in penalties if k in flags and penalties[k]}
    score = max(0, min(100, round(sum(weights[k] * v for k, v in breakdown.items()) - sum(deductions.values()))))
    return dict(score=score, score_breakdown=breakdown, score_penalties=deductions,
                score_explanation=explanation, score_tier=score_tier(score))


def review_hook_fit(moment, data_dir=None):
    anchor = moment.get("hook_anchors", {}).get(moment.get("selected_hook"))
    if anchor is None:
        return
    penalties = moment["score_penalties"]
    key = "misleading_hook"
    if not moment["start"] <= anchor < moment["end"]:
        _, config = scoring_config(data_dir)
        if config[key]:
            original = moment.setdefault("score_before_review", moment["score"])
            penalties[key] = config[key]
            moment["score"] = max(0, round(original - config[key]))
    elif key in penalties and "score_before_review" in moment:
        penalties.pop(key)
        moment["score"] = moment.pop("score_before_review")
    moment["score_tier"] = score_tier(moment["score"])
