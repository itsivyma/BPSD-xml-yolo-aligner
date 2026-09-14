"""Small Streamlit image viewer with click coordinates and local zoom."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import streamlit.components.v1 as components


_FRONTEND = Path(__file__).with_name("zoomable_image_frontend")
_component = components.declare_component(
    "bpsd_zoomable_image_coordinates", path=str(_FRONTEND)
)


def _image_data_url(source: object) -> str:
    if isinstance(source, (str, Path)):
        path = Path(source)
        content = path.read_bytes()
    elif isinstance(source, bytes):
        content = source
    elif hasattr(source, "save"):
        buffer = io.BytesIO()
        source.save(buffer, format="PNG", compress_level=1)  # type: ignore[attr-defined]
        content = buffer.getvalue()
    else:
        raise ValueError("image must be a path, PNG bytes, or PIL-compatible object")
    return "data:image/png;base64," + base64.b64encode(content).decode("ascii")


def zoomable_image_coordinates(
    source: object,
    *,
    key: str,
    cursor: str = "crosshair",
    max_height: int = 520,
) -> dict | None:
    """Show one independently zoomable image and return original-pixel clicks.

    The viewer supports its own controls, Ctrl/Cmd + wheel, and the pinch
    gesture emitted by macOS trackpads.  Returned width and height always
    describe the original image, so downstream note snapping is zoom-invariant.
    """

    return _component(
        src=_image_data_url(source),
        cursor=cursor,
        max_height=max(180, int(max_height)),
        min_zoom=0.5,
        max_zoom=4.0,
        default=None,
        key=key,
    )
