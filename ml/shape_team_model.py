"""Full team-v1 query features and learned controls for the R10 comparison.

The existing feature implementation is reused at query time. This adapter does
not accept a query ID or truth when constructing inputs. The short source
aggregation below mirrors ``features_v1.source_features``; the original helper
has notebook-style absolute imports, so importing it as a package would require
changing global module search state. The exact source files are fingerprinted.

The learner is the team's ``gate_eval.fit`` with its original log-error label,
63-leaf/4,000-tree cap, training-only early stopping, and two explicit threads.
Both reference-augmented controls add the same two feature names. No target,
capacity or stopping rule is changed between those controls.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from . import features as team_features
from . import sidecar_workflow as prior


SOURCE_FEATURES = ("src_log_est", "src_bound_rel", "src_edge_share", "src_edge_contrast", "src_in_out",
                   "src_empty_partial", "src_n_partial")
SHIFT_FEATURES = ("merge_shift", "abs_merge_shift")
TEAM_FEATURES = tuple(team_features.FEATURES) + SOURCE_FEATURES + SHIFT_FEATURES
AUX_FEATURES = ("critic_log_reference", "critic_proxy_score")
SCHEMA_VERSION = 1
MODEL_FILES = ("model.txt", "model.properties", "model.schema.json")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _rectangle(value):
    rectangle = np.asarray(value, dtype=float)
    if rectangle.shape != (4,) or not np.isfinite(rectangle).all():
        raise ValueError("Expected exactly four finite query coordinates, without IDs or truth")
    if rectangle[0] >= rectangle[1] or rectangle[2] >= rectangle[3]:
        raise ValueError("Query rectangles must have positive width and height")
    return rectangle


def _histogram(value):
    rate, bx, by, counts = value
    bx, by, counts = np.asarray(bx, float), np.asarray(by, float), np.asarray(counts, float)
    if (bx.ndim != 1 or by.ndim != 1 or len(bx) < 2 or len(by) < 2 or
        counts.shape != (len(bx)-1, len(by)-1) or not np.isfinite(float(rate)) or
        not np.isfinite(bx).all() or not np.isfinite(by).all() or
        (np.diff(bx) <= 0).any() or (np.diff(by) <= 0).any() or
        not np.isfinite(counts).all() or (counts < 0).any()):
        raise ValueError("Invalid histogram geometry, dimensions or nonnegative cell counts")
    return float(rate), bx, by, counts


def _context(hist):
    """Same array preparation as team features.py and features_v1.py."""
    _, bx, by, counts = _histogram(hist)
    area = np.outer(np.diff(bx), np.diff(by))
    density = counts/area
    contrast = np.abs(np.log((density+team_features.E)/(
        uniform_filter(density, size=3, mode="nearest")+team_features.E)))
    nonempty = density[counts > 0]
    return dict(bx=bx, by=by, C=counts, area=area, LC=contrast,
                med=float(np.median(nonempty)) if nonempty.size else 0.,
                total=float(counts.sum()), extent=(bx[0], bx[-1], by[0], by[-1]))


def _query(rectangle, context, sources=()):
    return team_features._query_features(rectangle, context["bx"], context["by"], context["C"],
        context["area"], context["LC"], context["med"], context["total"], sources)


def _load_merged(path):
    path = Path(path)
    return prior.load_histogram(path) if path.suffix == ".hist" else team_features.load_hist(path)


class TeamV1Runtime:
    """Loaded synopsis-only contexts for all 28 team-v1 inputs.

    Native R10 binary geometry and original Java text exports can differ after
    float32 quantization. Use the same explicit ``merged_path`` in every arm;
    original source text exports are retained here without a new quantization.
    ``compare_logged_features`` exposes Java/Python differences instead of
    replacing live values with logged inputs. Only the last geometry's feature
    calculation is available to internal helpers; ``describe`` always clears
    it so repeated live requests still compute all features. There is no
    query/answer lookup or raw-point state.
    """
    def __init__(self, histogram_dir, merged_histogram=None, *, merged_path=None, base_resolution=130):
        started = time.perf_counter()
        folder = Path(histogram_dir).expanduser().resolve()
        sources = sorted(folder.glob("src_*.txt"))
        if not sources:
            raise FileNotFoundError("Team-v1 source exports are missing: " + str(folder/"src_*.txt"))
        if merged_histogram is None:
            merged_path = Path(merged_path) if merged_path else folder/f"arm_{int(base_resolution)}.txt"
            merged_histogram = _load_merged(merged_path)
        elif merged_path is not None:
            serialized = _load_merged(merged_path)
            if not all(np.array_equal(a, b) for a, b in zip(merged_histogram, serialized)):
                raise ValueError("Explicit merged histogram differs from its serialized provenance")
        self.histogram = _histogram(merged_histogram)
        self.merge_rate = self.histogram[0]
        self.merged = _context(self.histogram)
        source_histograms = [_histogram(team_features.load_hist(path)) for path in sources]
        self.sources = [_context(hist) for hist in source_histograms]
        self.source_extents = [(*context["extent"], len(context["bx"])-1) for context in self.sources]
        self.alpha = float(prior._alpha(self.histogram))
        self.input_files = tuple([*(str(path.resolve()) for path in sources),
                                  *([] if merged_path is None else [str(Path(merged_path).resolve())])])
        self.input_manifest = [dict(path=path, bytes=Path(path).stat().st_size, sha256=_sha(path))
                               for path in self.input_files]
        self.serialized_input_bytes = sum(item["bytes"] for item in self.input_manifest)
        self._last_rectangle = None
        self._last_features = None
        self._last_answer = None
        self._last_source_estimate = None
        self.initialization_seconds = time.perf_counter()-started

    def _base(self, value):
        rectangle = _rectangle(value)
        if self._last_rectangle is not None and np.array_equal(rectangle, self._last_rectangle):
            return dict(self._last_features)
        result = _query(rectangle, self.merged, self.source_extents)
        answer = result.pop("est_py")
        a, b, c, d = rectangle
        estimate = lower = width = contrast = in_out = empty = partial = 0.
        for context in self.sources:
            x0, x1, y0, y1 = context["extent"]
            if not (x0 < b and a < x1 and y0 < d and c < y1):
                continue
            source = _query(rectangle, context)
            e = source["est_py"]
            estimate += e
            lower += np.expm1(source["log_full"])
            width += source["bound_rel"]*max(e, 1.)
            contrast += e*source["edge_contrast"]
            in_out += e*source["in_out"]
            empty += e*source["empty_partial"]
            partial += source["n_partial"]
        weight = max(estimate, team_features.E)
        shift = (answer-estimate)/max(estimate, 1.)
        result.update(merge_rate=self.merge_rate, skew_pred=prior._sa_score(rectangle, self.histogram, self.alpha),
            log_alpha=np.log10(max(self.alpha, 1.)), src_log_est=np.log1p(estimate),
            src_bound_rel=width/max(estimate, 1.), src_edge_share=(estimate-lower)/weight,
            src_edge_contrast=contrast/weight, src_in_out=in_out/weight,
            src_empty_partial=empty/weight, src_n_partial=partial,
            merge_shift=shift, abs_merge_shift=abs(shift))
        ordered = {name: float(result[name]) for name in TEAM_FEATURES}
        if not np.isfinite(list(ordered.values())).all():
            raise ValueError("Live team-v1 features must be finite")
        self._last_rectangle = rectangle.copy()
        self._last_features = ordered
        self._last_answer = float(answer)
        self._last_source_estimate = float(estimate)
        return dict(ordered)

    def describe(self, rectangle, reference=None):
        self.clear_query_cache()
        result = self._base(rectangle)
        if reference is not None:
            if np.asarray(reference).shape != () or not np.isfinite(reference) or float(reference) < 0:
                raise ValueError("The auxiliary reference must be one finite nonnegative count")
            result[AUX_FEATURES[0]] = float(np.log1p(reference))
            result[AUX_FEATURES[1]] = abs(self._last_answer-float(reference))/max(float(reference), 1.)
        return result

    def answer(self, rectangle):
        return float(prior.histogram_count(_rectangle(rectangle), self.histogram))

    def source_estimate(self, rectangle):
        self._base(rectangle)
        return self._last_source_estimate

    def sa_score(self, rectangle):
        return float(prior._sa_score(_rectangle(rectangle), self.histogram, self.alpha))

    def clear_query_cache(self):
        """Remove derived per-query state before any separate timed request."""
        self._last_rectangle = self._last_features = None
        self._last_answer = self._last_source_estimate = None

    def batch_features(self, rectangles, references=None):
        rectangles = np.asarray(rectangles, float)
        if rectangles.ndim != 2 or rectangles.shape[1] != 4:
            raise ValueError("Expected an Nx4 geometry-only array")
        if references is None:
            return pd.DataFrame([self.describe(q) for q in rectangles], columns=TEAM_FEATURES)
        references = np.asarray(references, float)
        if references.shape != (len(rectangles),):
            raise ValueError("Expected one auxiliary reference per rectangle")
        return pd.DataFrame([self.describe(q, r) for q, r in zip(rectangles, references)],
                            columns=TEAM_FEATURES+AUX_FEATURES)


def compare_logged_features(runtime, logged, query_rectangles, *, atol=1e-6, rtol=1e-6):
    """Report parity with build_table_v1 rows; do not substitute logged values.

    Original qId indexes ``query_rectangles``. The supplied log must describe
    this runtime's single merged histogram. This check is offline only.
    """
    required = {"qId", *TEAM_FEATURES}
    if not required <= set(logged):
        raise ValueError("Logged parity table lacks qId or team-v1 feature columns")
    qids = logged.qId.to_numpy()
    rectangles = np.asarray(query_rectangles, float)
    if rectangles.ndim != 2 or rectangles.shape[1] != 4 or qids.dtype.kind not in "iu":
        raise ValueError("Parity requires original integer query IDs and Nx4 coordinates")
    if (qids < 0).any() or (qids >= len(rectangles)).any():
        raise ValueError("Logged query IDs are outside the original geometry table")
    if not np.allclose(logged.merge_rate.to_numpy(float), runtime.merge_rate, rtol=0, atol=1e-12):
        raise ValueError("Logged rows describe a different merge rate")
    actual = runtime.batch_features(rectangles[qids])
    rows = []
    for name in TEAM_FEATURES:
        expected, observed = logged[name].to_numpy(float), actual[name].to_numpy(float)
        delta = abs(observed-expected)
        rows.append(dict(feature=name, queries=len(actual), max_abs_delta=float(delta.max(initial=0.)),
            max_relative_delta=float((delta/np.maximum(abs(expected), 1e-12)).max(initial=0.)),
            within_tolerance=bool(np.isclose(observed, expected, atol=atol, rtol=rtol).all()),
            absolute_tolerance=atol, relative_tolerance=rtol))
    return pd.DataFrame(rows)


def _feature_names(augmented):
    if not isinstance(augmented, (bool, np.bool_)):
        raise ValueError("augmented must be an explicit boolean")
    return TEAM_FEATURES + (AUX_FEATURES if augmented else ())


def _matrix(frame, names):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Expected a feature DataFrame with unique column names")
    missing = set(names)-set(frame)
    if missing:
        raise ValueError("Missing query-time features: " + ", ".join(sorted(missing)))
    result = frame.loc[:, list(names)].astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Model inputs must be finite; truth and IDs are never model features")
    return result


def _implementation_hashes():
    return {name: _sha(Path(__file__).with_name(name)) for name in
            ("shape_team_model.py", "features.py", "features_v1.py", "gate_eval.py")}


@dataclass
class TeamModel:
    booster: object
    schema: dict
    directory: Path
    load_seconds: float = 0.
    training_metadata: dict | None = None

    @property
    def feature_names(self):
        return tuple(self.schema["feature_names"])

    @property
    def model_bytes(self):
        return sum((self.directory/name).stat().st_size for name in MODEL_FILES)

    def log_scores(self, frame):
        scores = np.asarray(self.booster.predict(_matrix(frame, self.feature_names), num_threads=2), float)
        if scores.shape != (len(frame),) or not np.isfinite(scores).all():
            raise ValueError("Model returned nonfinite or invalid log-error scores")
        return scores

    def predict_errors(self, frame):
        with np.errstate(over="ignore"):
            values = np.exp(self.log_scores(frame))
        if not np.isfinite(values).all():
            raise ValueError("Exponentiated model error estimates overflowed")
        return values

    def predict_one(self, mapping):
        frame = pd.DataFrame([mapping], columns=self.feature_names)
        score = float(self.log_scores(frame)[0])
        estimate = float(np.exp(score))
        if not np.isfinite(estimate):
            raise ValueError("Exponentiated model error estimate overflowed")
        return score, estimate


def fit_team_model(training_table, seed, out_dir, *, augmented=False, target_column="actual_error",
                   model_name="full_v1"):
    """Fit exactly the team's declared learner using outer-TRAIN rows only.

    The caller owns outer memberships. This function accepts only their TRAIN
    table, holds back every tenth sorted TRAIN qId as gate_eval.fit does, and
    never sees calibration/test. It does not refit the held-back rows afterward,
    matching the existing team implementation. ``target_column`` is used only
    for the log label; it cannot enter the fixed feature whitelist.
    """
    from . import gate_eval
    names = _feature_names(augmented)
    matrix = _matrix(training_table, names)
    if "qId" not in training_table or target_column not in training_table:
        raise ValueError("TRAIN table requires qId and the explicit offline actual-error target")
    for role_column in ("split", "role", "outer_split"):
        if role_column in training_table:
            roles = training_table[role_column]
            if roles.isna().any() or not roles.astype(str).str.lower().eq("train").all():
                raise ValueError("Only outer-TRAIN rows are allowed; non-train " + role_column + " found")
    qids = training_table.qId.to_numpy()
    target = training_table[target_column].to_numpy(float)
    if (qids.dtype.kind not in "iu" or (qids < 0).any() or not np.isfinite(target).all()
            or (target < 0).any() or len(np.unique(qids)) < 12):
        raise ValueError("TRAIN needs at least 12 valid query IDs and finite nonnegative actual errors")
    if target_column in names:
        raise ValueError("Offline actual error cannot be a model feature")
    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    table = matrix.copy()
    table["qId"] = qids
    table["label"] = np.log(np.maximum(target, team_features.LABEL_FLOOR))
    parameters = {**gate_eval.PARAMS, "n_jobs": 2}
    held_back = set(sorted(np.unique(qids))[::10])
    started = time.perf_counter()
    model = gate_eval.fit(table, list(names), int(seed), params=parameters)
    training_seconds = time.perf_counter()-started
    booster = model.booster_
    booster.save_model(str(destination/"model.txt"))
    (destination/"model.properties").write_text("label.transform=log\n")
    schema = dict(schema_version=SCHEMA_VERSION, family="team_v1_LightGBM", model_name=str(model_name),
        feature_names=list(names), augmented=bool(augmented), label_transform="log",
        label_floor=team_features.LABEL_FLOOR, model_sha256=_sha(destination/"model.txt"),
        model_properties_sha256=_sha(destination/"model.properties"),
        implementation_sha256=_implementation_hashes(), seed=int(seed),
        best_iteration=int(model.best_iteration_), parameters=parameters,
        inner_validation="every tenth sorted outer-TRAIN qId, unchanged gate_eval.fit",
        refit_held_back=False)
    _json(destination/"model.schema.json", schema)
    fitted = TeamModel(booster, schema, destination)
    metadata = dict(model_name=str(model_name), seed=int(seed), train_queries=len(np.unique(qids)),
        train_rows=len(table), inner_fit_rows=int((~table.qId.isin(held_back)).sum()),
        inner_validation_rows=int(table.qId.isin(held_back).sum()), training_seconds=training_seconds,
        training_includes_inner_validation=True,
        best_iteration=int(model.best_iteration_), runtime_model_bytes=fitted.model_bytes,
        feature_count=len(names), target_column=target_column,
        outer_train_qIds=sorted(map(int, np.unique(qids))),
        inner_validation_qIds=sorted(map(int, held_back)),
        target_sha256=hashlib.sha256(np.ascontiguousarray(target, np.float64).tobytes()).hexdigest(),
        feature_matrix_sha256=hashlib.sha256(np.ascontiguousarray(matrix, np.float64).tobytes()).hexdigest())
    _json(destination/"training.json", metadata)
    fitted.training_metadata = metadata
    return fitted


def load_team_model(directory):
    started = time.perf_counter()
    path = Path(directory).expanduser().resolve()
    if path.is_file():
        path = path.parent
    schema = json.loads((path/"model.schema.json").read_text())
    if schema.get("schema_version") != SCHEMA_VERSION or schema.get("label_transform") != "log":
        raise ValueError("Unsupported team-v1 model schema/label transform")
    names = _feature_names(schema.get("augmented"))
    if tuple(schema.get("feature_names", ())) != names:
        raise ValueError("Model schema contains unsupported, reordered or leaking features")
    if schema.get("implementation_sha256") != _implementation_hashes():
        raise ValueError("Frozen team-v1 model implementation differs from the current source")
    for file, key in (("model.txt", "model_sha256"), ("model.properties", "model_properties_sha256")):
        if _sha(path/file) != schema[key]:
            raise ValueError("Model runtime artifact identity changed: " + file)
    lightgbm = importlib.import_module("lightgbm")
    booster = lightgbm.Booster(model_file=str(path/"model.txt"))
    if tuple(booster.feature_name()) != names:
        raise ValueError("Booster feature order differs from the fixed team-v1 schema")
    # training.json is offline provenance, not a runtime dependency. In
    # particular no training IDs, target hashes or validation labels are read.
    return TeamModel(booster, schema, path, time.perf_counter()-started)
