"""Local publication records and conservative calibration from measured retention."""

import csv
import io
import json
import os
import re
import threading
from datetime import date, datetime
from pathlib import Path
from statistics import StatisticsError, correlation, median
from typing import Annotated, Literal
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator


Count = Annotated[int, Field(ge=0, le=2**53 - 1)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
ClipId = Annotated[str, Field(pattern=r"^[a-zA-Z0-9]{1,64}$")]
HookStyle = Literal["question", "curiosity gap", "direct statement", "source quote", "commentary", "no hook", "other"]
MIN_SAMPLE = 5
OUTCOME = "average_percentage_viewed"
COMPONENTS = ("hook", "story", "emotion", "visual", "payoff", "retention")
DIRECTOR_COMPONENTS = ("hook", "pacing", "emotion", "dialogue", "chat")


def confirm_published(value):
    if value is not True:
        raise ValueError("Confirm the video was already published outside Studio")
    return value


PublicationConfirmation = Annotated[Literal[True], BeforeValidator(confirm_published)]


class RetentionPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position_percent: Annotated[Nonnegative, Field(le=100)]
    retained_percent: Nonnegative


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    views: Count | None = None
    engaged_views: Count | None = None
    viewed_percent: Annotated[Nonnegative, Field(le=100)] | None = None
    swiped_away_percent: Annotated[Nonnegative, Field(le=100)] | None = None
    average_view_duration: Nonnegative | None = None
    average_percentage_viewed: Nonnegative | None = None
    retention_graph: list[RetentionPoint] | None = Field(default=None, max_length=1000)
    likes: Count | None = None
    comments: Count | None = None
    shares: Count | None = None
    subscribers_gained: Count | None = None

    @field_validator("*", mode="before")
    @classmethod
    def reject_booleans(cls, value):
        if isinstance(value, bool):
            raise ValueError("Enter a measured number, not a boolean")
        return value

    @field_validator("retention_graph")
    @classmethod
    def order_points(cls, points):
        if points is None:
            return None
        if len({point.position_percent for point in points}) != len(points):
            raise ValueError("Retention graph positions must be unique")
        return sorted(points, key=lambda point: point.position_percent)


def youtube_id(value: str) -> str:
    try:
        url = urlsplit(value.strip())
        if (url.scheme not in {"https", "http"} or url.username is not None or url.password is not None
                or url.port not in (None, 80 if url.scheme == "http" else 443)):
            raise ValueError
        parts = url.path.strip("/").split("/")
        video = ""
        if url.hostname == "youtu.be" and len(parts) == 1:
            video = parts[0]
        elif url.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            if parts == ["watch"]:
                video = parse_qs(url.query).get("v", [""])[0]
            elif len(parts) == 2 and parts[0] in {"shorts", "live"}:
                video = parts[1]
        if re.fullmatch(r"[\w-]{11}", video, re.ASCII):
            return video
    except ValueError:
        pass
    raise ValueError("Enter a published YouTube video or Shorts URL")


class PublicationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    publish_url: str | None = Field(default=None, max_length=2048)
    publish_date: date | None = None
    title: str | None = Field(default=None, max_length=200)
    hook: str | None = Field(default=None, max_length=200)
    hook_style: HookStyle | None = None
    metrics: Metrics = Field(default_factory=Metrics)

    @field_validator("publish_url")
    @classmethod
    def canonical_url(cls, value):
        return f"https://www.youtube.com/watch?v={youtube_id(value)}" if value is not None else None


class Publication(PublicationUpdate):
    job_id: ClipId
    clip_id: ClipId
    publish_url: str = Field(max_length=2048)
    confirmed_published: PublicationConfirmation


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["csv", "json"]
    content: str = Field(min_length=1, max_length=1_000_000)
    job_id: ClipId | None = None
    clip_id: ClipId | None = None
    confirmed_published: PublicationConfirmation


def header(value):
    return re.sub(r"[^a-z0-9]", "", value.casefold())


CSV_HEADERS = {header(name): name for name in Metrics.model_fields}
CSV_HEADERS.update({
    "stayedtowatch": "viewed_percent",
    "viewed": "viewed_percent",
    "viewedvsswipedaway": "viewed_percent",
    "swipedaway": "swiped_away_percent",
    "subscribers": "subscribers_gained",
    "averageviewdurationseconds": "average_view_duration",
    "averageviewdurationsec": "average_view_duration",
    "audienceretentiongraph": "retention_graph",
})
CSV_META = {
    "jobid": "job_id", "clipid": "clip_id", "publishurl": "publish_url",
    "publisheddate": "publish_date", "publishdate": "publish_date", "videopublishtime": "publish_date",
    "videotitle": "title", "title": "title", "hook": "hook", "hookstyle": "hook_style",
}


def csv_value(key, value):
    if key == "retention_graph":
        return json.loads(value)
    if key == "average_view_duration" and re.fullmatch(r"(?:\d+:)?\d+:\d{2}(?:\.\d+)?", value):
        seconds = 0
        for part in value.split(":"):
            seconds = seconds * 60 + float(part)
        return seconds
    value = value.removesuffix("%").strip()
    if "," in value and re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", value):
        return value.replace(",", "")
    return value


def parse_csv(content):
    reader = csv.DictReader(io.StringIO(content.lstrip("\ufeff"), newline=""), strict=True)
    rows = []
    for row in reader:
        if None in row:
            raise ValueError(f"CSV row {reader.line_num} has extra columns")
        cells = {header(key): value.strip() for key, value in row.items() if value and value.strip()}
        video = next((cells[key] for key in ("content", "videoid", "video") if key in cells), "")
        if not cells or video.casefold() == "total" or not video and cells.get("title", "").casefold() == "total":
            continue
        item = {CSV_META[key]: value for key, value in cells.items() if key in CSV_META}
        if not item.get("publish_url") and re.fullmatch(r"[\w-]{11}", video, re.ASCII):
            item["publish_url"] = f"https://www.youtube.com/watch?v={video}"
        value = item.get("publish_date")
        if value:
            for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%dT%H:%M:%S%z"):
                try:
                    item["publish_date"] = datetime.strptime(value, fmt).date().isoformat()
                    break
                except ValueError:
                    continue
        item["metrics"] = {
            CSV_HEADERS[key]: csv_value(CSV_HEADERS[key], value)
            for key, value in cells.items() if key in CSV_HEADERS and value not in {"—", "-", "–"}
        }
        rows.append(item)
    return rows


def import_publications(spec, existing):
    rows = parse_csv(spec.content) if spec.format == "csv" else json.loads(spec.content.lstrip("\ufeff"))
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 500 or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Import must contain 1–500 per-video records")
    known = {record["publish_url"]: record for record in existing}
    result = []
    for index, row in enumerate(rows, 1):
        row = dict(row)
        if isinstance(row.get("publish_url"), str):
            url = PublicationUpdate(publish_url=row["publish_url"]).publish_url
            matched = known.get(url)
            if matched:
                row.setdefault("job_id", matched["job_id"])
                row.setdefault("clip_id", matched["clip_id"])
        if len(rows) == 1 and spec.job_id and spec.clip_id:
            row.setdefault("job_id", spec.job_id)
            row.setdefault("clip_id", spec.clip_id)
        if not row.get("job_id") or not row.get("clip_id"):
            raise ValueError(f"Record {index}: link the published URL to a highlight first or include job_id and clip_id")
        result.append(Publication.model_validate({**row, "confirmed_published": spec.confirmed_published}))
    if len(result) != len({entry.publish_url for entry in result}):
        raise ValueError("Import contains the same published video more than once")
    return result


class FeedbackStore:
    def __init__(self, data_dir):
        self.path = Path(data_dir).expanduser() / "feedback" / "publications.json"
        self.lock = threading.RLock()

    def records(self):
        with self.lock:
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return []

    def save(self, records):
        with self.lock:
            self.path.parent.mkdir(exist_ok=True, parents=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(records, allow_nan=False, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)


def numeric(value):
    return type(value) in (int, float)


def _correlation(pairs):
    if len(pairs) < MIN_SAMPLE:
        return None
    try:
        return round(correlation(*zip(*pairs)), 4)
    except StatisticsError:
        return None


def _pairs(records, metric, score):
    return [(score(record), record["metrics"][metric]) for record in records
            if record["metrics"].get(metric) is not None and numeric(score(record))]


def _percentile(value, values):
    return 100 * (sum(point < value for point in values) + (sum(point == value for point in values) - 1) / 2) / (len(values) - 1)


def comparison(records, metric=OUTCOME):
    pairs = _pairs(records, metric, lambda record: record["editorial_score"])
    corr = _correlation(pairs)
    ranks = {}
    if corr is not None:
        scores, measured = zip(*pairs)
        for record in records:
            score, value = record["editorial_score"], record["metrics"].get(metric)
            if numeric(score) and value is not None:
                ranks[record["id"]] = round(_percentile(value, measured) - _percentile(score, scores), 1)
    director_pairs = _pairs(records, metric, lambda record: (record.get("director_score") or {}).get("overall"))
    return {"metric": metric, "sample_size": len(pairs), "correlation": corr,
            "components": {key: _correlation(_pairs(records, metric, lambda record: record["score_breakdown"].get(key)))
                           for key in COMPONENTS}, "percentile_gaps": ranks,
            "director": {"sample_size": len(director_pairs), "correlation": _correlation(director_pairs),
                         "components": {key: _correlation(_pairs(
                             records, metric, lambda record: (record.get("director_score") or {}).get(key)))
                             for key in DIRECTOR_COMPONENTS}}}


def _duration_bucket(seconds):
    if seconds is None:
        return None
    if seconds < 15:
        return "Under 15s"
    if seconds > 60:
        return "Over 60s"
    if seconds < 25:
        return "15–25s"
    if seconds < 40:
        return "25–40s"
    return "40–55s" if seconds < 55 else "55–60s"


def _groups(records, value):
    grouped = {}
    for record in records:
        measured, key = record["metrics"].get(OUTCOME), value(record)
        if measured is not None and key:
            grouped.setdefault(key, []).append(measured)
    return sorted(({"label": key, "sample_size": len(samples), "median_percentage_viewed": median(samples)}
                   for key, samples in grouped.items() if len(samples) >= MIN_SAMPLE),
                  key=lambda group: (-group["median_percentage_viewed"], -group["sample_size"], group["label"]))


def _records():
    return FeedbackStore(os.environ.get("CLIPPER_DATA_DIR", "~/clipper-data")).records()


def calibration_summary(records=None):
    records = _records() if records is None else records
    return {"metric": OUTCOME, "minimum_sample": MIN_SAMPLE, "record_count": len(records),
            "sample_size": sum(record["metrics"].get(OUTCOME) is not None for record in records),
            "component_sample_size": sum(record["metrics"].get(OUTCOME) is not None
                                         and any(numeric(v) for v in record["score_breakdown"].values())
                                         for record in records),
            "durations": _groups(records, lambda record: _duration_bucket(record["duration"])),
            "hook_styles": _groups(records, lambda record: record["hook_style"]),
            "editing_modes": _groups(records, lambda record: record["editing_mode"])}


def calibrated_weights(default_weights: dict[str, float]) -> dict[str, float]:
    records = _records()
    correlations = {
        key: _correlation(_pairs(records, OUTCOME, lambda record: record["score_breakdown"].get(key)))
        for key in default_weights
    }
    # Compare supported components only. A convex blend caps any shift at five percentage points.
    supported = [key for key, corr in correlations.items() if corr is not None and default_weights[key] > 0]
    total = sum(default_weights[key] for key in supported)
    if len(supported) < 2 or not total:
        return default_weights.copy()
    factors = {key: default_weights[key] * (1 + .5 * correlations[key]) for key in supported}
    denominator = sum(factors.values())
    return {key: .95 * weight + .05 * total * factors[key] / denominator if key in factors else weight
            for key, weight in default_weights.items()}
