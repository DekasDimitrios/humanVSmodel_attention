"""Parser tests against a *synthetic* Locate result — the real schema must be confirmed with
`hvma parse-results --dump-schema` after the first live job."""

from hvma.data import Sample
from hvma.rapidata_job import extract_points, extract_weights, match_results_to_manifest


def _sample(i: int) -> Sample:
    return Sample(id=f"s{i:04d}", image_path=f"data/images/s{i:04d}.png", prompt="", width=200,
                  height=100)


def test_extract_points_normalised_and_pixel():
    entry = {"detailedResults": [
        {"points": [{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.4}],
         "userDetails": {"userScores": {"global": 0.9}}},
        {"points": [{"x": 0.5, "y": 0.6}], "userDetails": {"userScores": {"global": 0.7}}},
    ]}
    pts = extract_points(entry)
    assert pts == [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
    # weights only returned when there is one score per point; here 2 scores vs 3 points -> None
    assert extract_weights(entry, len(pts)) is None

    pixel_entry = {"responses": [{"x": 100, "y": 50}]}
    assert extract_points(pixel_entry, size=(100, 200)) == [(0.5, 0.5)]
    assert extract_points(pixel_entry) == []  # pixels but no size -> dropped


def test_match_by_filename_then_fallback_to_order():
    samples = [_sample(0), _sample(1)]
    entries = [
        {"assetUrls": {"s0001.png": "https://assets.rapidata.ai/abc.png"}},
        {"assetUrls": {"s0000.png": "https://assets.rapidata.ai/def.png"}},
    ]
    m = match_results_to_manifest(entries, samples)
    assert m["s0000"] is entries[1] and m["s0001"] is entries[0]

    anonymous = [{"foo": 1}, {"foo": 2}]
    m2 = match_results_to_manifest(anonymous, samples)
    assert m2["s0000"] is anonymous[0] and m2["s0001"] is anonymous[1]
