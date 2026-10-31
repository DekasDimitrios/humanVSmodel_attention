"""Parser tests against the schema of the first live Locate job (results v4.1.0, SDK 3.27),
with labeller details redacted."""

import pytest

from hvma.data import Sample
from hvma.rapidata_job import extract_taps, match_results_to_manifest


def _sample(i: int) -> Sample:
    return Sample(id=f"s{i:04d}", image_path=f"data/images/s{i:04d}.png", prompt="", width=200,
                  height=100)


def _response(locations, score):
    return {"locations": [{"x": x, "y": y} for x, y in locations],
            "userDetails": {"country": "XX", "userScores": {"global": score}}}


REAL_ENTRY = {
    "identifier": "dp_redacted",
    "originalFileName": "s0003.png",
    "privateMetadata": {"id": "s0003"},
    "notApplicable": 0,
    # the union of all responses' locations, repeated: must not be counted again
    "aggregatedResults": [{"x": 25.0, "y": 50.0}, {"x": 80.0, "y": 10.0}, {"x": 81.0, "y": 11.0}],
    "detailedResults": [
        _response([(25.0, 50.0)], 0.9),
        _response([(80.0, 10.0), (81.0, 11.0)], 0.5),  # one labeller, two taps
    ],
}


def test_extract_taps_reads_percent_and_ignores_aggregated():
    pts, weights = extract_taps(REAL_ENTRY)
    assert pts == [(0.25, 0.5), (0.8, 0.1), (0.81, 0.11)]  # percent -> 0-1, no duplicates
    assert weights == [0.9, 0.5, 0.5]  # each tap carries its labeller's reliability
    with pytest.raises(ValueError):
        extract_taps({"aggregatedResults": REAL_ENTRY["aggregatedResults"]})


def test_match_by_filename_then_fallback_to_order():
    samples = [_sample(0), _sample(1)]
    entries = [
        {"assetUrls": {"s0001.png": "https://assets.rapidata.ai/abc.png"}},
        {"assetUrls": {"s0000.png": "https://assets.rapidata.ai/def.png"}},
    ]
    m = match_results_to_manifest(entries, samples)
    assert m["s0000"] is entries[1] and m["s0001"] is entries[0]

    # what we submit: private_metadata={"id": ...}, returned on each entry (SDK 3.27)
    tagged = [{"privateMetadata": {"id": "s0001"}}, {"privateMetadata": {"id": "s0000"}}]
    m1 = match_results_to_manifest(tagged, samples)
    assert m1["s0000"] is tagged[1] and m1["s0001"] is tagged[0]

    anonymous = [{"foo": 1}, {"foo": 2}]
    m2 = match_results_to_manifest(anonymous, samples)
    assert m2["s0000"] is anonymous[0] and m2["s0001"] is anonymous[1]
