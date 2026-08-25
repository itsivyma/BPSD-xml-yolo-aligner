import json

from bpsd_aligner.job_admin import collect_job_statuses
from bpsd_aligner.job_store import job_directory, write_job_manifest, write_job_status


def test_job_admin_reports_states_without_full_owner_or_job_ids(tmp_path):
    job = job_directory("a" * 64, root=tmp_path)
    write_job_manifest(
        job,
        fingerprint=job.name,
        pipeline_version="v1",
        owner_id="b" * 64,
        inputs=[],
    )
    write_job_status(
        job,
        state="running",
        stage="page_alignment",
        completed_pages=1,
        total_pages=3,
    )

    report = collect_job_statuses(tmp_path)

    assert report["state_counts"] == {"running": 1}
    assert report["jobs"][0]["job_id"] == "a" * 12
    assert report["jobs"][0]["owner_id"] == "b" * 12
    assert report["jobs"][0]["completed_pages"] == 1
    assert report["storage_bytes"] > 0
    json.dumps(report)
