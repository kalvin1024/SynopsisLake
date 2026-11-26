"""Byte-accounted raw-count grids for the held-out-region comparison.

The payload reuses notebook 22's ``SLCNT001`` count codec: a 64-byte header
(``<8sII4d16x``) followed by an x-major square array of little-endian uint32
counts. Resolution is selected from the available *grid* bytes alone, before
query labels are used. The caller must subtract other deployed artifacts from
the total method budget before passing ``budget_bytes`` here.

``query_interval`` returns a fine-grid answer, conservative lower/upper count
bounds, and the maximum relative error over that interval. Bounds use inclusive
query semantics and conservatively include bins touching a query edge. No
additional reference histogram, model, query labels or stored answers are used
by this direct-answer score. It is different from notebook 22's reference mode,
which scores an unchanged ARM answer against this grid's estimate.
"""
from __future__ import annotations

import hashlib
import gzip
import math
from pathlib import Path
import time

import numpy as np


HEADER_BYTES = 64
COUNT_BYTES = 4
UINT32_MAX = int(np.iinfo(np.uint32).max)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _root(value):
    result = np.asarray(value, dtype=np.float64)
    if (result.shape != (4,) or not np.isfinite(result).all()
            or not result[0] < result[1] or not result[2] < result[3]):
        raise ValueError("root must be finite ordered (min_x, max_x, min_y, max_y)")
    return result


def payload_bytes(resolution):
    """Serialized bytes, including every required header/count byte."""
    n = _integer(resolution, "resolution", 1)
    if n > UINT32_MAX:
        raise ValueError("resolution exceeds the existing count-grid codec")
    return HEADER_BYTES + COUNT_BYTES * n * n


def resolution_for_budget(budget_bytes):
    """Largest square-grid resolution that fits, without padding or labels."""
    cap = _integer(budget_bytes, "budget_bytes", HEADER_BYTES + COUNT_BYTES)
    return min(UINT32_MAX, math.isqrt((cap - HEADER_BYTES) // COUNT_BYTES))


def write_count_histogram(path, counts, root, *, budget_bytes=None):
    """Validate and write using the existing notebook-22 uint32 codec.

    ``counts`` must already be exact integer bin masses. This helper never
    normalizes, clips, rounds, deduplicates or otherwise changes them.
    """
    values = np.asarray(counts)
    bounds = _root(root)
    if (values.ndim != 2 or values.shape[0] == 0
            or values.shape[0] != values.shape[1]
            or values.dtype.kind not in "uif"
            or not np.isfinite(values).all() or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError("counts must be a nonempty square matrix of exact nonnegative integers")
    if values.max(initial=0) > UINT32_MAX:
        raise OverflowError("Fine raw-count cell exceeds the existing uint32 codec")
    expected = payload_bytes(values.shape[0])
    if budget_bytes is not None and expected > _integer(budget_bytes, "budget_bytes", 1):
        raise ValueError("Count-grid payload exceeds its available grid budget")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Keep the existing format and all its header accounting unchanged.
    from .sidecar_workflow import _save_fine
    actual = _save_fine(target, values, bounds)
    if actual != expected:
        raise AssertionError("Existing count-grid codec changed its serialized size")
    return actual


def load_count_histogram(path):
    """Return the standard (rate, x_edges, y_edges, counts) histogram tuple."""
    from .sidecar_workflow import _load_fine
    return _load_fine(Path(path))


def build_fine_histogram(raw_paths, root, budget_bytes, output_path, *, chunk_size=250000):
    """Build a byte-matched count grid by streaming the actual raw inputs.

    Raw files use the repository's existing whitespace-separated reader and may
    be plain or gzip compressed. The first two columns are unweighted x/y; later
    columns are ignored. Rows from overlapping input files are summed exactly as
    in the shared Java raw counter, not deduplicated. The grid root must contain
    every input point. A point on the outer maximum belongs to the last bin.
    The returned manifest describes offline work; only the count payload is an
    online artifact. No query file or labels are accepted by this API.
    """
    from .sidecar_workflow import iter_raw
    from pandas.errors import EmptyDataError

    started = time.perf_counter()
    bounds = _root(root)
    cap = _integer(budget_bytes, "budget_bytes", HEADER_BYTES + COUNT_BYTES)
    chunk_size = _integer(chunk_size, "chunk_size", 1)
    n = resolution_for_budget(cap)
    paths = [Path(p).expanduser().resolve() for p in raw_paths]
    if not paths:
        raise ValueError("At least one raw input file is required")
    if len(set(paths)) != len(paths):
        raise ValueError("Raw input paths must be unique; duplicate files would double-count rows")
    counts = np.zeros((n, n), dtype=np.uint64)
    sources = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        rows = 0
        try:
            opener = gzip.open if path.suffix.lower() == ".gz" else open
            # Explicit ownership closes the source even when an invalid point
            # causes an early exit from the repository's chunk generator.
            with opener(path, "rt") as stream:
                for points in iter_raw(stream, chunk_size):
                    if ((points[:, 0] < bounds[0]).any() or (points[:, 0] > bounds[1]).any()
                            or (points[:, 1] < bounds[2]).any() or (points[:, 1] > bounds[3]).any()):
                        raise ValueError(f"Raw points fall outside the declared histogram root: {path}")
                    ix = np.minimum(((points[:, 0] - bounds[0]) / (bounds[1] - bounds[0]) * n)
                                    .astype(np.int64), n - 1)
                    iy = np.minimum(((points[:, 1] - bounds[2]) / (bounds[3] - bounds[2]) * n)
                                    .astype(np.int64), n - 1)
                    counts += np.bincount(ix * n + iy, minlength=n * n).reshape(n, n).astype(np.uint64)
                    rows += len(points)
        except EmptyDataError:
            # An empty input tile has no points. A malformed nonempty line is
            # rejected by the existing reader rather than silently skipped.
            if rows:
                raise
        if before != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError(f"Raw input changed during count-grid construction: {path}")
        sources.append({"path": str(path), "file_bytes": before[0], "rows": rows})
    total_rows = sum(source["rows"] for source in sources)
    if int(counts.sum(dtype=np.uint64)) != total_rows:
        raise AssertionError("Fine-grid construction lost raw point mass")
    actual = write_count_histogram(output_path, counts, bounds, budget_bytes=cap)
    return dict(
        status="complete", format="SLCNT001", version=1,
        resolution=n, grid_shape=[n, n], root=bounds.tolist(),
        grid_budget_bytes=cap, grid_bytes=actual, unused_grid_budget_bytes=cap - actual,
        raw_rows=total_rows, sources=sources,
        output_path=str(Path(output_path).resolve()),
        sha256=hashlib.sha256(Path(output_path).read_bytes()).hexdigest(),
        construction_seconds=time.perf_counter() - started,
        selection_uses_query_labels=False,
        resolution_rule="largest square grid whose existing64-byte-header plus uint32 counts fit the available grid cap",
        raw_semantics="unweighted first two whitespace-separated columns; sum source rows; last outer grid edge included",
    )


def _histogram(hist):
    if len(hist) != 4:
        raise ValueError("Expected the standard four-item histogram tuple")
    _, bx, by, counts = hist
    bx, by, counts = np.asarray(bx, float), np.asarray(by, float), np.asarray(counts)
    if (bx.ndim != 1 or by.ndim != 1 or len(bx) < 2 or len(by) < 2
            or counts.shape != (len(bx) - 1, len(by) - 1)
            or not np.isfinite(bx).all() or not np.isfinite(by).all()
            or (np.diff(bx) <= 0).any() or (np.diff(by) <= 0).any()
            or not np.isfinite(counts).all() or (counts < 0).any()
            or not np.equal(counts, np.floor(counts)).all()):
        raise ValueError("Invalid exact raw-count histogram")
    return bx, by, counts


def query_interval(hist, rectangle):
    """Return ``(F, L, U, score)`` for an inclusive rectangular count query.

    F is the uniform within-bin count estimate. Every fully contained bin adds
    its entire mass to L. U includes all bins with a possibly nonempty closed
    intersection, including zero-area boundary contacts. This is conservative
    for raw rows lying on internal or outer grid edges and for zero-area query
    rectangles. For true count T in [L,U], score upper-bounds |F-T|/T under the
    shared zero policy (both zero:0, positive answer at truth0:infinity).

    Positive-infinite scores must be hard-rejected by the calibrated gate.
    """
    bx, by, counts = _histogram(hist)
    q = np.asarray(rectangle, dtype=float)
    if (q.shape != (4,) or not np.isfinite(q).all()
            or q[0] > q[1] or q[2] > q[3]):
        raise ValueError("Expected finite ordered query bounds")
    if q[1] < bx[0] or q[0] > bx[-1] or q[3] < by[0] or q[2] > by[-1]:
        return 0., 0, 0, 0.
    i0 = max(0, int(np.searchsorted(bx, q[0], side="left")) - 1)
    i1 = min(len(bx) - 1, int(np.searchsorted(bx, q[1], side="right")))
    j0 = max(0, int(np.searchsorted(by, q[2], side="left")) - 1)
    j1 = min(len(by) - 1, int(np.searchsorted(by, q[3], side="right")))
    if i0 >= i1 or j0 >= j1:
        return 0., 0, 0, 0.
    sub = counts[i0:i1, j0:j1]
    left, right = bx[i0:i1], bx[i0 + 1:i1 + 1]
    bottom, top = by[j0:j1], by[j0 + 1:j1 + 1]
    fx = np.maximum(0., np.minimum(right, q[1]) - np.maximum(left, q[0])) / (right - left)
    fy = np.maximum(0., np.minimum(top, q[3]) - np.maximum(bottom, q[2])) / (top - bottom)
    full_x = (q[0] <= left) & (right <= q[1])
    full_y = (q[2] <= bottom) & (top <= q[3])
    lower = int(sub[np.ix_(full_x, full_y)].sum(dtype=np.uint64))
    upper = int(sub.sum(dtype=np.uint64))
    answer = float((sub * np.outer(fx, fy)).sum(dtype=np.float64))
    # Avoid an impossible ordering solely from a final floating-point reduction.
    answer = max(float(lower), min(float(upper), answer))
    if lower > 0:
        score = max((answer - lower) / lower, (upper - answer) / upper)
    elif answer > 0:
        score = math.inf
    else:
        score = 1. if upper > 0 else 0.
    return answer, lower, upper, float(score)


def query_count(hist, rectangle):
    """Uniform fine-grid answer; does not return a stored query lookup."""
    return query_interval(hist, rectangle)[0]


def direct_score(hist, rectangle):
    """Worst-case relative error of the fine answer over its count interval."""
    return query_interval(hist, rectangle)[3]


def reference_score(original_answer, reference_answer):
    """Notebook22-style control: assess unchanged H using fine-grid F."""
    original_answer, reference_answer = float(original_answer), float(reference_answer)
    if (not math.isfinite(original_answer) or not math.isfinite(reference_answer)
            or original_answer < 0 or reference_answer < 0):
        raise ValueError("Expected finite nonnegative count estimates")
    return abs(original_answer - reference_answer) / max(reference_answer, 1.)
