import threading

from clipper_studio.scheduler import Scheduler


def test_priority_round_robin_and_positions():
    rendered = []
    scheduler = Scheduler(lambda *a: None, lambda *item: rendered.append(item[:2]),
                          analysis_workers=1, render_workers=1)
    for cid in range(8):
        scheduler.put(("a", str(cid), 0))
    for cid in range(2):
        scheduler.put(("b", str(cid), 0))
    scheduler.put(("review", "edit", 0), priority=True)
    order = [("review", "edit"), ("a", "0"), ("b", "0"), ("a", "1"), ("b", "1")]
    order += [("a", str(cid)) for cid in range(2, 8)]
    assert scheduler.positions() == ({}, {item: i for i, item in enumerate(order, 1)})
    scheduler.start()
    try:
        scheduler.join()
        assert rendered == order
        assert scheduler.positions() == ({}, {})
    finally:
        scheduler.close()


def test_independent_pool_limits():
    analyses, renders = threading.Barrier(4), threading.Barrier(3)
    release = threading.Event()

    def analyze(*args):
        analyses.wait(timeout=5)
        assert release.wait(5)

    def render(*args):
        if not release.is_set():
            renders.wait(timeout=5)
        assert release.wait(5)

    scheduler = Scheduler(analyze, render, analysis_workers=3, render_workers=2)
    for jid in ("a", "b", "c"):
        scheduler.put((jid, None, 0))
        scheduler.put((jid, "clip", 0))
    scheduler.start()
    try:
        analyses.wait(timeout=5)
        renders.wait(timeout=5)
        assert len(scheduler.positions()[1]) == 1
    finally:
        release.set()
        scheduler.join()
        scheduler.close()


def test_cancel_all_queues_and_worker_survives_error(caplog):
    rendered = []

    def render(jid, cid, generation):
        if jid == "broken":
            raise RuntimeError("broken render")
        rendered.append(jid)

    scheduler = Scheduler(lambda *a: None, render, render_workers=1, preflight=lambda *a: None)
    scheduler.put(("cancelled", None, 0), preflight=True)
    scheduler.put(("cancelled", None, 0))
    scheduler.put(("cancelled", "priority", 0), priority=True)
    scheduler.put(("cancelled", "normal", 0))
    scheduler.put(("broken", "clip", 0))
    scheduler.put(("ready", "clip", 0))
    scheduler.cancel("cancelled")
    scheduler.start()
    try:
        scheduler.join()
        assert rendered == ["ready"]
        assert "broken render" in caplog.text
        assert scheduler.positions() == ({}, {})
    finally:
        scheduler.close()
