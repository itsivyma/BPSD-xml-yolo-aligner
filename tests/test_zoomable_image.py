from bpsd_aligner import zoomable_image


def test_zoomable_image_uses_component_key_for_persistent_view_state(monkeypatch):
    captured = {}

    def fake_component(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(zoomable_image, "_component", fake_component)

    result = zoomable_image.zoomable_image_coordinates(
        b"png-content",
        key="review-image-42",
        max_height=420,
    )

    assert result is None
    assert captured["key"] == "review-image-42"
    assert captured["state_key"] == "review-image-42"
    assert captured["max_height"] == 420
