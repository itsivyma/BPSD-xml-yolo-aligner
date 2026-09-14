const image = document.getElementById("image");
const viewport = document.getElementById("viewport");
const range = document.getElementById("zoom-range");
const label = document.getElementById("zoom-label");
const zoomIn = document.getElementById("zoom-in");
const zoomOut = document.getElementById("zoom-out");
const zoomReset = document.getElementById("zoom-reset");

let zoom = 1;
let minZoom = 0.5;
let maxZoom = 4;
let maxHeight = 520;
let cursor = "crosshair";
let baseScale = 1;
let stateKey = "";
let touchStartDistance = null;
let touchStartZoom = 1;
let suppressClickUntil = 0;
let saveFrame = null;

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function availableWidth() {
  return Math.max(1, viewport.clientWidth || document.documentElement.clientWidth || 1);
}

function updateBaseScale() {
  if (!image.naturalWidth) return;
  baseScale = Math.min(1, availableWidth() / image.naturalWidth);
}

function layout() {
  if (!image.naturalWidth || !image.naturalHeight) return;
  updateBaseScale();
  const renderedWidth = Math.max(1, image.naturalWidth * baseScale * zoom);
  const renderedHeight = Math.max(1, image.naturalHeight * baseScale * zoom);
  image.style.width = `${renderedWidth}px`;
  image.style.height = `${renderedHeight}px`;
  image.style.cursor = cursor;
  const viewportHeight = Math.max(80, Math.min(renderedHeight, maxHeight));
  viewport.style.height = `${viewportHeight}px`;
  range.value = String(Math.round(zoom * 100));
  label.textContent = `${Math.round(zoom * 100)}%`;
  Streamlit.setFrameHeight(
    Math.ceil(document.getElementById("toolbar").getBoundingClientRect().height + viewportHeight + 6)
  );
}

function storageKey() {
  return stateKey ? `bpsd-zoomable-image:${stateKey}` : "";
}

function loadViewState() {
  const key = storageKey();
  if (!key) return null;
  try {
    const saved = JSON.parse(window.sessionStorage.getItem(key));
    if (!saved || !Number.isFinite(saved.zoom)) return null;
    return {
      zoom: saved.zoom,
      scrollLeft: Number.isFinite(saved.scrollLeft) ? saved.scrollLeft : 0,
      scrollTop: Number.isFinite(saved.scrollTop) ? saved.scrollTop : 0
    };
  } catch (_error) {
    return null;
  }
}

function saveViewState() {
  const key = storageKey();
  if (!key) return;
  try {
    window.sessionStorage.setItem(key, JSON.stringify({
      zoom: zoom,
      scrollLeft: viewport.scrollLeft,
      scrollTop: viewport.scrollTop
    }));
  } catch (_error) {
    // Storage can be disabled without affecting the image viewer itself.
  }
}

function scheduleViewStateSave() {
  if (saveFrame !== null) return;
  saveFrame = window.requestAnimationFrame(() => {
    saveFrame = null;
    saveViewState();
  });
}

function restoreViewState(saved) {
  layout();
  window.requestAnimationFrame(() => {
    viewport.scrollLeft = saved ? saved.scrollLeft : 0;
    viewport.scrollTop = saved ? saved.scrollTop : 0;
    saveViewState();
  });
}

function setZoom(nextZoom, clientX = null, clientY = null) {
  if (!image.naturalWidth) return;
  const oldRect = image.getBoundingClientRect();
  const viewportRect = viewport.getBoundingClientRect();
  const focusX = clientX === null ? viewportRect.left + viewport.clientWidth / 2 : clientX;
  const focusY = clientY === null ? viewportRect.top + viewport.clientHeight / 2 : clientY;
  const imageFractionX = oldRect.width ? (focusX - oldRect.left) / oldRect.width : 0.5;
  const imageFractionY = oldRect.height ? (focusY - oldRect.top) / oldRect.height : 0.5;
  const localX = focusX - viewportRect.left;
  const localY = focusY - viewportRect.top;

  zoom = clamp(nextZoom, minZoom, maxZoom);
  layout();

  const newRect = image.getBoundingClientRect();
  viewport.scrollLeft += imageFractionX * newRect.width - localX - viewport.scrollLeft;
  viewport.scrollTop += imageFractionY * newRect.height - localY - viewport.scrollTop;
  saveViewState();
}

function pointerDistance(first, second) {
  return Math.hypot(first.clientX - second.clientX, first.clientY - second.clientY);
}

image.addEventListener("click", (event) => {
  if (Date.now() < suppressClickUntil) return;
  const rect = image.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const x = (event.clientX - rect.left) * image.naturalWidth / rect.width;
  const y = (event.clientY - rect.top) * image.naturalHeight / rect.height;
  Streamlit.setComponentValue({
    x: x,
    y: y,
    width: image.naturalWidth,
    height: image.naturalHeight,
    unix_time: Date.now()
  });
});

viewport.addEventListener("wheel", (event) => {
  if (!(event.ctrlKey || event.metaKey)) return;
  event.preventDefault();
  const factor = Math.exp(-event.deltaY * 0.01);
  setZoom(zoom * factor, event.clientX, event.clientY);
}, {passive: false});

viewport.addEventListener("touchstart", (event) => {
  if (event.touches.length !== 2) return;
  touchStartDistance = pointerDistance(event.touches[0], event.touches[1]);
  touchStartZoom = zoom;
  suppressClickUntil = Date.now() + 500;
}, {passive: true});

viewport.addEventListener("touchmove", (event) => {
  if (event.touches.length !== 2 || !touchStartDistance) return;
  event.preventDefault();
  const distance = pointerDistance(event.touches[0], event.touches[1]);
  const centerX = (event.touches[0].clientX + event.touches[1].clientX) / 2;
  const centerY = (event.touches[0].clientY + event.touches[1].clientY) / 2;
  setZoom(touchStartZoom * distance / touchStartDistance, centerX, centerY);
  suppressClickUntil = Date.now() + 500;
}, {passive: false});

viewport.addEventListener("touchend", () => {
  touchStartDistance = null;
}, {passive: true});

viewport.addEventListener("scroll", scheduleViewStateSave, {passive: true});

range.addEventListener("input", () => setZoom(Number(range.value) / 100));
zoomIn.addEventListener("click", () => setZoom(zoom + 0.25));
zoomOut.addEventListener("click", () => setZoom(zoom - 0.25));
zoomReset.addEventListener("click", () => setZoom(1));
window.addEventListener("resize", layout);

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, (event) => {
  const args = event.detail.args;
  minZoom = Number(args.min_zoom || 0.5);
  maxZoom = Number(args.max_zoom || 4);
  maxHeight = Number(args.max_height || 520);
  cursor = args.cursor || "crosshair";
  stateKey = String(args.state_key || "");
  range.min = String(Math.round(minZoom * 100));
  range.max = String(Math.round(maxZoom * 100));
  if (image.src !== args.src) {
    const saved = loadViewState();
    zoom = clamp(saved ? saved.zoom : 1, minZoom, maxZoom);
    image.onload = () => restoreViewState(saved);
    image.src = args.src;
  } else {
    layout();
  }
});

Streamlit.setComponentReady();
