import json

from bpsd_aligner.job_dispatcher import (
    acquire_dispatcher_lock,
    queued_requests,
    release_dispatcher_lock,
    run_dispatcher,
)


def _queued_job(root, name: str, created_at: str):
    job = root / name
    job.mkdir()
    request = job / "job_request.json"
    request.write_text("{}", encoding="utf-8")
    (job / "job_status.json").write_text(
        json.dumps({"state": "queued", "created_at": created_at}),
        encoding="utf-8",
    )
    return request


def test_dispatcher_orders_queue_and_continues_after_failed_job(tmp_path, monkeypatch):
    later = _queued_job(tmp_path, "b" * 64, "2026-01-02T00:00:00Z")
    earlier = _queued_job(tmp_path, "a" * 64, "2026-01-01T00:00:00Z")
    calls = []

    def fake_worker(path):
        calls.append(path)
        (path.parent / "job_status.json").write_text(
            json.dumps({"state": "failed" if path == earlier else "completed"}),
            encoding="utf-8",
        )
        if path == earlier:
            raise RuntimeError("expected test failure")

    monkeypatch.setattr("bpsd_aligner.job_dispatcher.run_background_job", fake_worker)

    assert queued_requests(tmp_path) == [earlier, later]
    assert run_dispatcher(tmp_path, idle_grace_seconds=0, max_idle_checks=1) == 2
    assert calls == [earlier, later]


def test_dispatcher_lock_allows_only_one_local_process(tmp_path):
    lock = acquire_dispatcher_lock(tmp_path)
    assert lock is not None
    assert acquire_dispatcher_lock(tmp_path) == lock
    release_dispatcher_lock(lock)
    assert not lock.exists()
