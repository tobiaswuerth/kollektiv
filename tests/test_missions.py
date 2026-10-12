from kollektiv import Settings
from kollektiv.missions import mission_samples


def test_samples_cover_difficulty_tiers_and_preserve_custom_run_configuration():
    samples = mission_samples()
    assert len({sample["id"] for sample in samples}) == len(samples)
    for difficulty in ("easy", "medium", "elaborate"):
        assert len([sample for sample in samples if sample["difficulty"] == difficulty]) >= 2
    custom = Settings(model="my-local-model", max_concurrency=3)
    for sample in samples:
        merged = Settings.model_validate({**custom.model_dump(), **sample["settings"]})
        assert merged.team == custom.team
        assert merged.model == custom.model
        assert merged.max_concurrency == custom.max_concurrency
        assert merged.max_tasks < custom.max_tasks
        assert sample["goal"] and sample["description"]


def test_sample_copies_are_isolated_from_bootstrap_consumers():
    edited = mission_samples()
    edited[0]["settings"]["max_tasks"] = 200
    edited[0]["goal"] = "Changed by a consumer"
    original = mission_samples()[0]
    assert original["settings"]["max_tasks"] == 3
    assert original["goal"] != edited[0]["goal"]
