"""Probing tab: run linear-probe sweeps over persona vectors.

UX mirrors the Analysis tab (source -> mask -> variant -> personas), but
the action is a probe sweep and the output is a metric-over-layer curve,
the best-layer summary, and optional controls (shuffled-label selectivity,
save artifact).

The probe primitives all live in ``persona_vectors.probes``; this file
is a thin Streamlit wrapper around them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from persona_data.environment import get_artifacts_dir
from persona_vectors.analysis import LayeredSamples
from persona_vectors.attributes import attribute_display_label
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import plot_metric_comparison, plot_metric_over_layers
from persona_vectors.probes import (
    AttributeLabels,
    attribute_probe_labels,
    filter_attribute_samples_min_count,
    infer_probe_task,
    layer_matrix,
    save_probe_artifact,
    shuffle_label_baseline,
    sweep_attribute,
)

from utils.analysis_metadata import (
    synth_persona_attribute_names,
    synth_persona_dataset_cached,
)
from utils.analysis_sources import (
    DEFAULT_COMPARE_MODEL,
    DEFAULT_HUB_REPO,
    SOURCE_HUB,
    SOURCE_LOCAL,
    SOURCES,
    Store,
    activation_store_cached,
    available_variants,
    hub_models_by_mask_strategy,
    load_persona_vectors_cached,
    local_model_options_cached,
    persona_names_cached,
    personas_cached,
    store_cache_parts,
    store_layers_cached,
)
from utils.controls import render_mask_strategy_select
from utils.helpers import widget_key

# ---------------------------------------------------------------------------
# Constants and config
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = "artifacts/probes"
_MIN_CLASS_COUNT = 5

# Per-task primary metric for "best layer" + first plot.
_PRIMARY_METRIC = {
    "binary": "balanced_accuracy",
    "categorical": "balanced_accuracy",
    "ordinal": "balanced_accuracy",
    "numeric": "r2",
}
_SECONDARY_METRIC = {
    "binary": None,
    "categorical": None,
    "ordinal": "mae",
    "numeric": "mae",
}


@dataclass(frozen=True)
class _SweepInputs:
    source: str
    location: str
    model_name: str
    mask_value: str
    variant: str
    persona_ids: tuple[str, ...]
    attribute: str
    task: str
    n_pca_components: int | None
    layers: tuple[int, ...]
    min_class_count: int
    seed: int


# ---------------------------------------------------------------------------
# Source / store selection (slim mirror of the analysis tab pattern)
# ---------------------------------------------------------------------------


def _select_source() -> str:
    key = widget_key("probe", "source")
    source = st.segmented_control(
        "Source",
        options=SOURCES,
        default=st.session_state.get(key, SOURCE_HUB),
        key=key,
        label_visibility="collapsed",
    )
    return source or SOURCE_HUB


def _select_store(source: str, mask_strategy: MaskStrategy) -> Store:
    if source == SOURCE_HUB:
        repo = st.text_input(
            "Hub repo",
            value=st.session_state.get("probe:hub_repo", DEFAULT_HUB_REPO),
            key="probe:hub_repo",
        )
        models = hub_models_by_mask_strategy(repo).get(mask_strategy, [])
        if not models:
            st.warning(
                f"No Hub vector configs for `{mask_strategy.value}` in `{repo}`."
            )
            model_name = st.text_input(
                "Model",
                value=st.session_state.get("probe:hub_model_fallback", DEFAULT_COMPARE_MODEL),
                key="probe:hub_model_fallback",
            )
        else:
            previous = st.session_state.get(
                widget_key("probe", "hub_model", repo, mask_strategy.value),
                models[0],
            )
            model_name = st.selectbox(
                "Model",
                options=models,
                index=models.index(previous) if previous in models else 0,
                key=widget_key("probe", "hub_model", repo, mask_strategy.value),
            )
        return activation_store_cached(SOURCE_HUB, repo, model_name, mask_strategy.value)

    root = st.text_input(
        "Artifacts root",
        value=str(get_artifacts_dir() / "activations"),
        key="probe:local_root",
    )
    root = str(Path(root).expanduser())
    models = local_model_options_cached(root, mask_strategy.value)
    if models:
        previous = st.session_state.get("probe:local_model", models[0])
        model_name = st.selectbox(
            "Model",
            options=models,
            index=models.index(previous) if previous in models else 0,
            key="probe:local_model",
        )
    else:
        model_name = st.text_input(
            "Model",
            value=st.session_state.get("probe:local_model_fallback", DEFAULT_COMPARE_MODEL),
            key="probe:local_model_fallback",
        )
    return activation_store_cached(SOURCE_LOCAL, root, model_name, mask_strategy.value)


def _select_variant(store: Store, mask_strategy: MaskStrategy) -> str | None:
    variants = available_variants(store, mask_strategy)
    if not variants:
        st.info("No variants with saved vectors for this selection.")
        return None
    previous = st.session_state.get("probe:variant", variants[0])
    return st.selectbox(
        "Variant",
        options=variants,
        index=variants.index(previous) if previous in variants else 0,
        key="probe:variant",
    )


def _select_personas(
    store: Store, variant: str, mask_strategy: MaskStrategy
) -> list[str]:
    source, location, model_name = store_cache_parts(store)
    all_ids = personas_cached(source, location, model_name, mask_strategy.value, (variant,))
    if not all_ids:
        st.info("No personas found for this variant.")
        return []
    regular = all_ids
    if len(regular) < 2:
        st.info("At least two non-assistant personas are needed for probing.")
        return []
    min_count = min(10, len(regular))
    if min_count == len(regular):
        count = len(regular)
        st.warning(
            f"Only {count} non-assistant personas are available; using all of them."
        )
        st.session_state["probe:persona_count"] = count
        persona_ids = regular
        persona_names_cached(
            source,
            location,
            model_name,
            mask_strategy.value,
            (variant,),
            tuple(persona_ids),
        )
        st.caption(f"Probing {len(persona_ids)} non-assistant personas.")
        return persona_ids

    default_count = min(
        len(regular),
        max(min_count, st.session_state.get("probe:persona_count", len(regular))),
    )
    count = st.slider(
        "Personas",
        min_value=min_count,
        max_value=len(regular),
        value=default_count,
        key="probe:persona_count_slider",
    )
    st.session_state["probe:persona_count"] = count
    persona_ids = regular[:count]
    persona_names_cached(
        source, location, model_name, mask_strategy.value, (variant,), tuple(persona_ids)
    )
    st.caption(f"Probing {len(persona_ids)} of {len(regular)} non-assistant personas.")
    return persona_ids


# ---------------------------------------------------------------------------
# Probe config UI
# ---------------------------------------------------------------------------


def _select_attribute() -> str:
    dataset = synth_persona_dataset_cached()
    options = list(synth_persona_attribute_names())
    if "sex" in options:
        default_index = options.index("sex")
    else:
        default_index = 0
    return st.selectbox(
        "Attribute to probe",
        options=options,
        index=default_index,
        format_func=lambda name: attribute_display_label(dataset, name),
        key="probe:attribute",
    )


def _select_pca_components() -> int | None:
    use_pca = st.toggle(
        "Add PCA-compressed comparison",
        value=False,
        key="probe:use_pca",
        help="Runs the normal full-activation sweep and a second sweep where "
        "PCA is fit on the train split only before probing.",
    )
    if not use_pca:
        return None
    return int(
        st.number_input(
            "PCA components",
            min_value=2,
            max_value=512,
            value=10,
            step=1,
            key="probe:pca_components",
        )
    )


def _select_layers(num_layers: int) -> list[int]:
    fast = st.toggle(
        "Fast layer set (5 evenly-spaced)",
        value=True,
        key="probe:fast",
        help="Off = sweep every layer. Slow on big models.",
    )
    if not fast:
        return list(range(num_layers))
    return sorted({
        0,
        num_layers // 4,
        num_layers // 2,
        (3 * num_layers) // 4,
        num_layers - 1,
    })


# ---------------------------------------------------------------------------
# Sweep + display
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _cached_sweep(
    inputs: _SweepInputs,
) -> tuple[dict[str, list[dict[str, object]]], AttributeLabels, LayeredSamples]:
    samples = load_persona_vectors_cached(
        inputs.source, inputs.location, inputs.model_name,
        inputs.mask_value, inputs.variant, inputs.persona_ids,
    )
    dataset = synth_persona_dataset_cached()
    labels = attribute_probe_labels(
        dataset, inputs.attribute, list(inputs.persona_ids), task=inputs.task,  # type: ignore[arg-type]
    )
    probe_samples, labels = filter_attribute_samples_min_count(
        samples, labels, min_count=inputs.min_class_count
    )

    def _sweep(n_pca: int | None) -> list[dict[str, object]]:
        return sweep_attribute(
            probe_samples, labels,
            layers=list(inputs.layers),
            n_pca_components=n_pca,
            seed=inputs.seed,
        )

    if inputs.n_pca_components is not None:
        # Always overlay the compressed sweep against full activations.
        rows_by_label = {
            f"pca{inputs.n_pca_components}": _sweep(inputs.n_pca_components),
            "full": _sweep(None),
        }
    else:
        rows_by_label = {"full": _sweep(None)}
    return rows_by_label, labels, probe_samples


def _show_sweep(
    rows_by_label: dict[str, list[dict[str, object]]],
    labels: AttributeLabels,
    samples: LayeredSamples,
    attribute: str,
    task: str,
    inputs: _SweepInputs,
) -> None:
    primary = _PRIMARY_METRIC[task]
    secondary = _SECONDARY_METRIC.get(task)

    # Tolerate stale session state from a previous code version (bare rows).
    if isinstance(rows_by_label, list):
        rows_by_label = {"full": rows_by_label}
    primary_label = (
        f"pca{inputs.n_pca_components}" if inputs.n_pca_components else "full"
    )
    rows = rows_by_label.get(primary_label) or next(iter(rows_by_label.values()))

    def _plot(metric: str):
        if len(rows_by_label) > 1:
            return plot_metric_comparison(rows_by_label, attribute, metric=metric)
        return plot_metric_over_layers(rows, attribute, metric=metric)

    st.plotly_chart(_plot(primary), width="stretch")
    if secondary is not None:
        st.plotly_chart(_plot(secondary), width="stretch")

    higher_better = primary != "mae"

    def _best_row(label_rows: list[dict[str, object]]) -> dict[str, object] | None:
        valid_rows = [row for row in label_rows if row.get(primary) is not None]
        if not valid_rows:
            return None
        return max(
            valid_rows,
            key=lambda row: row[primary] * (1 if higher_better else -1),
        )

    valid = [row for row in rows if row.get(primary) is not None]
    if not valid:
        st.warning(f"No rows reported {primary!r}; can't pick a best layer.")
        return
    best = _best_row(rows)
    if best is None:
        return

    if len(rows_by_label) > 1:
        summary_rows = []
        for label, label_rows in rows_by_label.items():
            label_best = _best_row(label_rows)
            if label_best is None:
                continue
            summary_rows.append({
                "features": label,
                "best_layer": label_best["layer"],
                "probe": label_best["probe_kind"],
                primary: round(float(label_best[primary]), 3),
                f"baseline_{primary}": round(
                    float(label_best.get(f"baseline_{primary}", float("nan"))), 3
                ),
            })
        if summary_rows:
            st.dataframe(summary_rows, width="stretch", hide_index=True)

    feature_desc = (
        f" · pca{inputs.n_pca_components}" if inputs.n_pca_components else ""
    )

    cols = st.columns([1, 1.2, 1.8])
    cols[0].metric("Best layer", best["layer"])
    cols[1].metric(
        f"Best {primary}",
        f"{best[primary]:.3f}",
        delta=f"baseline {best.get(f'baseline_{primary}', float('nan')):.3f}",
        delta_color="off",
    )
    cols[2].metric("Probe", f"{best['probe_kind']}{feature_desc}")

    _render_selectivity_control(best, labels, samples, task, inputs)
    _render_save_artifact(best, labels, samples, attribute, task, inputs)


def _render_selectivity_control(
    best: dict[str, object],
    labels: AttributeLabels,
    samples: LayeredSamples,
    task: str,
    inputs: _SweepInputs,
) -> None:
    if task == "numeric":
        return  # selectivity control is classification-only
    with st.expander("Selectivity control (shuffled labels)"):
        st.caption(
            "Trains the same probe on shuffled labels. The gap between the real-label "
            "score and this shuffled score is the probe's *selectivity* "
            "(Hewitt & Liang 2019). High shuffled scores mean the probe is reading "
            "dataset artifacts, not the property."
        )
        n_repeats = st.slider(
            "Shuffle repeats", min_value=3, max_value=15, value=5,
            key="probe:shuffle_repeats",
        )
        if st.button("Run selectivity control", key="probe:run_shuffle"):
            with st.spinner("Running shuffled-label control..."):
                X = layer_matrix(samples, int(best["layer"]))
                shuffled = shuffle_label_baseline(
                    X, labels.y,
                    task=task,  # type: ignore[arg-type]
                    layer=int(best["layer"]),
                    probe_kind=best["probe_kind"],  # type: ignore[arg-type]
                    n_pca_components=inputs.n_pca_components,
                    n_repeats=n_repeats,
                )
            cols = st.columns(2)
            cols[0].metric(
                "Real balanced acc.",
                f"{float(best['balanced_accuracy']):.3f}",
            )
            cols[1].metric(
                "Shuffled balanced acc.",
                f"{shuffled['balanced_accuracy_mean']:.3f}",
                delta=f"+/- {shuffled['balanced_accuracy_std']:.3f}",
                delta_color="off",
            )


def _render_save_artifact(
    best: dict[str, object],
    labels: AttributeLabels,
    samples: LayeredSamples,
    attribute: str,
    task: str,
    inputs: _SweepInputs,
) -> None:
    def synced_default(key: str, default: str) -> str:
        default_key = f"{key}:default"
        previous_default = st.session_state.get(default_key)
        current_value = st.session_state.get(key)
        if current_value is None or current_value == previous_default:
            st.session_state[key] = default
        st.session_state[default_key] = default
        return st.session_state[key]

    with st.expander("Save best probe (loadable by the Chat tab)"):
        output_dir = st.text_input(
            "Output directory",
            value=st.session_state.get("probe:output_dir", _DEFAULT_OUTPUT_DIR),
            key="probe:output_dir",
            help="Probe artifacts will be written under this root.",
        )
        synced_default("probe:save_model", inputs.model_name)
        model_name = st.text_input(
            "Model name (for the artifact path)",
            key="probe:save_model",
        )
        synced_default("probe:save_variant", inputs.variant)
        variant = st.text_input(
            "Variant",
            key="probe:save_variant",
        )
        synced_default("probe:save_mask", inputs.mask_value)
        mask_value = st.text_input(
            "Mask strategy",
            key="probe:save_mask",
        )
        if st.button("Save", key="probe:save_artifact"):
            X = layer_matrix(samples, int(best["layer"]))
            directory = save_probe_artifact(
                X=X, y=labels.y, labels=labels,
                task=task,  # type: ignore[arg-type]
                probe_kind=best["probe_kind"],  # type: ignore[arg-type]
                n_pca_components=inputs.n_pca_components,
                layer=int(best["layer"]),
                model_name=model_name,
                variant=variant,
                mask_strategy=mask_value,
                output_dir=output_dir,
                metrics=best,
            )
            st.success(f"Saved to `{directory}`")
            st.caption(
                f"Wrote `probe.json` + `weights.safetensors`. "
                "The Chat tab can load the saved `probe.json` artifact."
            )


# ---------------------------------------------------------------------------
# Tab entry point
# ---------------------------------------------------------------------------


def render_probing_tab() -> None:
    st.title("Probing")

    source = _select_source()
    with st.expander("Source", expanded=True):
        mask_strategy = render_mask_strategy_select(
            key=widget_key("probe", "mask_strategy"),
            last_key="probe:last_mask_strategy",
            help_text="Which extracted activation set to load.",
        )
        store = _select_store(source, mask_strategy)
        variant = _select_variant(store, mask_strategy)
        if variant is None:
            return
        persona_ids = _select_personas(store, variant, mask_strategy)
        if not persona_ids:
            return

    dataset = synth_persona_dataset_cached()
    with st.expander("Probe configuration", expanded=True):
        attribute = _select_attribute()
        task = infer_probe_task(dataset, attribute)
        st.caption(f"Inferred task: **{task}**")

        n_pca_components = _select_pca_components()

        source, location, model_name = store_cache_parts(store)
        available_layers = store_layers_cached(
            source,
            location,
            model_name,
            mask_strategy.value,
            (variant,),
            tuple(persona_ids),
        )
        if not available_layers:
            st.info("No layers found for the selected personas.")
            return
        num_layers = max(available_layers) + 1
        layers = _select_layers(num_layers)
        min_class_count = _MIN_CLASS_COUNT
        seed = st.number_input(
            "Seed", min_value=0, max_value=10_000, value=0, step=1,
            key="probe:seed",
            help="Seeds the probe/PCA fit. The 80/20 split itself is fixed "
            "(random_state=0).",
        )

    inputs = _SweepInputs(
        source=source, location=location, model_name=model_name,
        mask_value=mask_strategy.value, variant=variant,
        persona_ids=tuple(persona_ids), attribute=attribute, task=task,
        n_pca_components=n_pca_components,
        layers=tuple(layers), min_class_count=min_class_count,
        seed=int(seed),
    )

    run = st.button("Run sweep", type="primary", key="probe:run")
    state_key = "probe:last_result"
    if run:
        with st.spinner("Evaluating probes across layers..."):
            try:
                sweep, labels, probe_samples = _cached_sweep(inputs)
            except Exception as exc:
                st.error(f"Sweep failed: {exc}")
                st.session_state.pop(state_key, None)
                return
        st.session_state[state_key] = (
            sweep,
            labels,
            probe_samples,
            attribute,
            task,
            inputs,
        )

    if state_key in st.session_state:
        saved_result = st.session_state[state_key]
        if len(saved_result) == 5:
            sweep, labels, probe_samples, last_attr, last_task = saved_result
            result_inputs = inputs
        else:
            sweep, labels, probe_samples, last_attr, last_task, result_inputs = saved_result
        _show_sweep(sweep, labels, probe_samples, last_attr, last_task, result_inputs)
