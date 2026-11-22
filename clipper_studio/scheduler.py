"""Independent analysis workers and priority/round-robin render workers."""

import logging
import threading
from collections import OrderedDict, deque


class Scheduler:
    def __init__(self, analyze, render, analysis_workers=3, render_workers=2, preflight=None, metadata_workers=3):
        if not all(1 <= count <= 32 for count in (analysis_workers, render_workers, metadata_workers)):
            raise ValueError("Worker counts must be between 1 and 32")
        self.analyze = analyze
        self.render = render
        self.preflight = preflight
        self.metadata_workers = metadata_workers
        self.analysis_workers = analysis_workers
        self.render_workers = render_workers
        self.condition = threading.Condition()
        self.analysis = deque()
        self.checks = deque()
        self.priority = deque()
        self.renders = OrderedDict()
        self.outstanding = 0
        self.stopping = False
        self.threads = []

    def put(self, item, priority=False, preflight=False):
        jid, cid, generation = item
        with self.condition:
            if self.stopping:
                return
            if preflight:
                self.checks.append(item)
            elif cid is None:
                self.analysis.append(item)
            elif priority:
                self.priority.append(item)
            else:
                self.renders.setdefault(jid, deque()).append(item)
            self.outstanding += 1
            self.condition.notify_all()

    def _next(self, kind):
        with self.condition:
            while not self.stopping:
                if kind == "metadata" and self.checks:
                    return self.checks.popleft()
                if kind == "analysis" and self.analysis:
                    return self.analysis.popleft()
                if kind == "render":
                    if self.priority:
                        return self.priority.popleft()
                    if self.renders:
                        jid, clips = self.renders.popitem(last=False)
                        item = clips.popleft()
                        if clips:
                            self.renders[jid] = clips
                        return item
                self.condition.wait()
            return None

    def _worker(self, kind):
        while (item := self._next(kind)) is not None:
            try:
                if kind == "metadata":
                    self.preflight(item[0], item[2])
                elif kind == "analysis":
                    self.analyze(item[0], item[2])
                else:
                    self.render(*item)
            except Exception:
                logging.exception("Unhandled scheduler task error")
            finally:
                with self.condition:
                    self.outstanding -= 1
                    self.condition.notify_all()

    def positions(self):
        with self.condition:
            analyses = {item[0]: i for i, item in enumerate(self.checks, 1)}
            analyses.update({item[0]: i for i, item in enumerate(self.analysis, 1)})
            ordered = list(self.priority)
            rounds = [deque(clips) for clips in self.renders.values()]
            while rounds:
                for clips in rounds:
                    ordered.append(clips.popleft())
                rounds = [clips for clips in rounds if clips]
            renders = {(jid, cid): i for i, (jid, cid, _) in enumerate(ordered, 1)}
            return analyses, renders

    def cancel(self, jid, cid=None):
        with self.condition:
            def keep(item):
                return item[0] != jid or (cid is not None and item[1] != cid)

            before = len(self.checks) + len(self.analysis) + len(self.priority) + sum(map(len, self.renders.values()))
            self.checks = deque(filter(keep, self.checks))
            self.analysis = deque(filter(keep, self.analysis))
            self.priority = deque(filter(keep, self.priority))
            if jid in self.renders:
                self.renders[jid] = deque(filter(keep, self.renders[jid]))
                if not self.renders[jid]:
                    self.renders.pop(jid)
            after = len(self.checks) + len(self.analysis) + len(self.priority) + sum(map(len, self.renders.values()))
            self.outstanding -= before - after
            self.condition.notify_all()

    def start(self):
        for kind, count in (("metadata", self.metadata_workers if self.preflight else 0),
                            ("analysis", self.analysis_workers), ("render", self.render_workers)):
            for index in range(count):
                thread = threading.Thread(target=self._worker, args=(kind,), daemon=True,
                                          name=f"{kind}-{index + 1}")
                self.threads.append(thread)
                thread.start()

    def join(self):
        with self.condition:
            self.condition.wait_for(lambda: self.outstanding == 0)

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        for thread in self.threads:
            thread.join(timeout=1)
