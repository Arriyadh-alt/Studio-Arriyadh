"""Persistent analysis status and artifact helpers."""

import json
import uuid
from typing import Literal

Stage = Literal["source_validation", "video_analysis", "audio_analysis", "transcript_generation",
                "chat_analysis", "viral_moment_detection", "ai_director", "final_rendering"]
ANALYSIS_STAGES = ("video_analysis", "audio_analysis", "transcript_generation")
RENDER_STAGES = ("ai_director", "final_rendering")
RETRYABLE = ("failed", "partially_completed")


def stage_state(status="pending", progress=0, attempts=0):
    return dict(status=status, progress=progress, attempts=attempts, error=None, artifacts=[])


def stage_states(job):
    states = job.setdefault("stages", {})
    for name in Stage.__args__:
        if name not in states:
            done = ((name == "source_validation" and job.get("metadata")) or
                    (name == "viral_moment_detection" and job.get("analysis_complete")))
            states[name] = stage_state("completed", 100, 1) if done else stage_state()
    return states


def mark_stage(job, name, status, progress=None, error=None, artifacts=()):
    state = stage_states(job)[name]
    if status == "processing" and state["status"] != "processing":
        state["attempts"] += 1
    if progress is None:
        progress = 100 if status in ("completed", "skipped") else state["progress"]
    state.update(status=status, progress=round(max(0, min(100, progress))), error=error)
    state["artifacts"] = list(dict.fromkeys([*state["artifacts"], *artifacts]))
    return state


class Checkpoints:
    def __init__(self, root):
        self.root = root

    def file(self, name):
        return self.root / name

    def read(self, name):
        try:
            return json.loads(self.file(name).read_text())
        except (FileNotFoundError, ValueError):
            return None

    def write(self, name, value):
        target = self.file(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(value, allow_nan=False))
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def media(self, name, fetch, *, refresh=False):
        value = None if refresh else self.read(name)
        if value is not None:
            paths = tuple(self.file(item) if item else None for item in value)
            if all(path is None or path.is_file() for path in paths):
                return paths
        paths = tuple(fetch())
        if all(path is None or (path.is_file() and path.is_relative_to(self.root)) for path in paths):
            local = [str(path.relative_to(self.root)) if path else None for path in paths]
            if any(local):
                self.write(name, local)
        return paths


class AnalysisTracker:
    def __init__(self, cache, publish, check, frozen=()):
        self.cache, self.publish, self.check = cache, publish, check
        self.expected = {}
        self.done = {name: {} for name in ANALYSIS_STAGES}
        self.errors = {name: set() for name in ANALYSIS_STAGES}
        self.started = set()
        self.frozen = frozen

    def expect(self, stage, count):
        self.expected[stage] = count

    def progress(self, stage):
        expected = self.expected.get(stage, 1)
        return min(99, 100 * len(self.done[stage]) / max(1, expected))

    def run(self, stage, key, work):
        self.check()
        name = f"{stage}/{key}.json"
        value = self.cache.read(name)
        if value is None:
            if stage in self.frozen:
                raise RuntimeError("This input is not available in the saved checkpoint")
            try:
                if stage not in self.started:
                    self.publish(stage, "processing")
                    self.started.add(stage)
                value = work()
                self.check()
                self.cache.write(name, value)
            except Exception:
                self.check()
                self.errors[stage].add(key)
                raise
        self.done[stage][key] = f"checkpoints/{name}"
        if stage in self.started:
            self.publish(stage, "processing", self.progress(stage), artifacts=(f"checkpoints/{name}",))
        return value

    def finish(self, required_error=None):
        for stage in ANALYSIS_STAGES:
            if stage in self.frozen:
                continue
            done, errors, expected = self.done[stage], self.errors[stage], self.expected.get(stage)
            if errors or (done and expected is not None and len(done) < expected):
                self.publish(stage, "partially_completed" if done else "failed", self.progress(stage),
                             "Some inputs could not be processed." if done else "This stage could not be completed.",
                             done.values())
            elif done:
                self.publish(stage, "completed", artifacts=done.values())
            elif stage in self.started or (stage == "audio_analysis" and required_error):
                self.publish(stage, "failed", error="Audio was unavailable for AI analysis.")
            else:
                self.publish(stage, "skipped")
