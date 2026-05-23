"""Probing tab: run linear-probe sweeps over persona vectors.

UX mirrors the Analysis tab (source -> mask -> variant -> personas), but
the action is a probe sweep and the output is a metric-over-layer curve,
the best-layer summary, and optional controls (shuffled-label selectivity,
save artifact).

The probe primitives all live in ``persona_vectors.probes``; this file
is a thin Streamlit wrapper around them.
"""

from __future__ import annotations

import streamlit as st
from persona_vectors.analysis import LayeredSamples
from persona_vectors.attributes import attribute_display_label
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import plot_metric_comparison, plot_metric_over_layers
from persona_vectors.probes import (
    AttributeLabels,
    default_probe_kinds,
    infer_probe_task,
    layer_matrix,
    save_probe_artifact,
    shuffle_label_baseline,
)

from tabs.probe_sweep import SweepInputs, cached_sweep
from utils.analysis_metadata import (
    synth_persona_attribute_names,
    synth_persona_dataset_cached,
)
from utils.analysis_sources import (
    Store,
    available_variants,
    persona_names_cached,
    personas_cached,
    store_cache_parts,
    store_layers_cached,
)
from utils.controls import render_mask_strategy_select
from utils.helpers import widget_key
from utils.source_controls import render_source_select, render_store_select

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
    all_ids = personas_cached(
        source, location, model_name, mask_strategy.value, (variant,)
    )
    if not all_ids:
        st.info("No personas found for this variant.")
        return []
    if len(all_ids) < 2:
        st.info("At least two non-assistant personas are needed for probing.")
        return []

    min_count = min(10, len(all_ids))
    has_slider = min_count < len(all_ids)
    if has_slider:
        default_count = max(
            min_count,
            min(len(all_ids), st.session_state.get("probe:persona_count", len(all_ids))),
        )
        count = st.slider(
            "Personas",
            min_value=min_count,
            max_value=len(all_ids),
            value=default_count,
            key="probe:persona_count_slider",
        )
    else:
        count = len(all_ids)
        st.warning(
            f"Only {count} non-assistant personas are available; using all of them."
        )

    st.session_state["probe:persona_count"] = count
    persona_ids = all_ids[:count]
    persona_names_cached(
        source,
        location,
        model_name,
        mask_strategy.value,
        (variant,),
        tuple(persona_ids),
    )
    if has_slider:
        st.caption(
            f"Probing {len(persona_ids)} of {len(all_ids)} non-assistant personas."
        )
    else:
        st.caption(f"Probing {len(persona_ids)} non-assistant personas.")
    return persona_ids


# ---------------------------------------------------------------------------
# Probe config UI
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _attribute_tasks() -> dict[str, str]:
    dataset = synth_persona_dataset_cached()
    return {
        name: infer_probe_task(dataset, name)
        for name in synth_persona_attribute_names()
    }


def _select_attributes() -> list[str]:
    """Multi-select locked to one task type.

    Picking the first attribute fixes the task; only same-task attributes stay
    selectable. Clearing the selection reopens every attribute again.
    """
    dataset = synth_persona_dataset_cached()
    tasks = _attribute_tasks()
    all_names = list(synth_persona_attribute_names())

    key = "probe:attributes"
    if key not in st.session_state:
        st.session_state[key] = ["sex"] if "sex" in all_names else all_names[:1]

    selected = st.session_state[key]
    if selected:
        locked = tasks[selected[0]]
        options = [name for name in all_names if tasks[name] == locked]
    else:
        options = all_names

    return st.multiselect(
        "Attributes to probe",
        options=options,
        format_func=lambda name: attribute_display_label(dataset, name),
        key=key,
        help="Pick one or more attributes of the same task type. They are "
        "overlaid in one figure. Remove all to switch to a different task type.",
    )


def _select_probe_kinds(task: str) -> list[str]:
    """Pick which probe families to fit. Only shown when the task has >1."""
    available = list(default_probe_kinds(task))  # type: ignore[arg-type]
    if len(available) < 2:
        return available
    selected = st.multiselect(
        "Probe kinds to fit",
        options=available,
        default=available,
        key=f"probe:kinds:{task}",
        help="Which probe families to fit at each layer. Defaults to all "
        "available for this task.",
    )
    return selected or available


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
    return sorted(
        {
            0,
            num_layers // 4,
            num_layers // 2,
            (3 * num_layers) // 4,
            num_layers - 1,
        }
    )


# ---------------------------------------------------------------------------
# Sweep + display
# ---------------------------------------------------------------------------


def _show_sweep(
    rows_by_label: dict[str, list[dict[str, object]]],
    per_attr: dict[str, tuple[AttributeLabels, LayeredSamples]],
    attributes: tuple[str, ...],
    task: str,
    inputs: SweepInputs,
) -> None:
    primary = _PRIMARY_METRIC[task]
    secondary = _SECONDARY_METRIC.get(task)

    primary_label = (
        f"pca{inputs.n_pca_components}" if inputs.n_pca_components else "full"
    )
    rows = rows_by_label.get(primary_label) or next(iter(rows_by_label.values()))

    def _plot(metric: str):
        if len(rows_by_label) > 1 or len(attributes) > 1:
            return plot_metric_comparison(
                rows_by_label, list(attributes), metric=metric
            )
        return plot_metric_over_layers(rows, attributes[0], metric=metric)

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

    multi_attr = len(attributes) > 1
    if len(rows_by_label) > 1 or multi_attr:
        summary_rows = []
        for label, label_rows in rows_by_label.items():
            for attribute in attributes:
                attr_rows = [
                    row for row in label_rows if row.get("attribute") == attribute
                ]
                label_best = _best_row(attr_rows)
                if label_best is None:
                    continue
                summary_row: dict[str, object] = {}
                if multi_attr:
                    summary_row["attribute"] = attribute
                summary_row.update(
                    {
                        "features": label,
                        "best_layer": label_best["layer"],
                        "probe": label_best["probe_kind"],
                        primary: round(float(label_best[primary]), 3),
                        f"baseline_{primary}": round(
                            float(label_best.get(f"baseline_{primary}", float("nan"))),
                            3,
                        ),
                    }
                )
                summary_rows.append(summary_row)
        if summary_rows:
            st.dataframe(summary_rows, width="stretch", hide_index=True)

    feature_desc = f" · pca{inputs.n_pca_components}" if inputs.n_pca_components else ""

    best_attr = str(best["attribute"])
    labels, samples = per_attr[best_attr]
    if multi_attr:
        # The per-attribute summary table above already covers every result;
        # a single "best" card would only show one attribute, so skip it and
        # just say which one the controls below operate on.
        st.caption(f"Controls below use the best result: **{best_attr}**.")
    else:
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
    _render_save_artifact(best, labels, samples, task, inputs)


def _render_selectivity_control(
    best: dict[str, object],
    labels: AttributeLabels,
    samples: LayeredSamples,
    task: str,
    inputs: SweepInputs,
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
            "Shuffle repeats",
            min_value=3,
            max_value=15,
            value=5,
            key="probe:shuffle_repeats",
        )
        if st.button("Run selectivity control", key="probe:run_shuffle"):
            with st.spinner("Running shuffled-label control..."):
                X = layer_matrix(samples, int(best["layer"]))
                shuffled = shuffle_label_baseline(
                    X,
                    labels.y,
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
    task: str,
    inputs: SweepInputs,
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
                X=X,
                y=labels.y,
                labels=labels,
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

    source = render_source_select(widget_scope="probe")
    with st.expander("Source", expanded=True):
        mask_strategy = render_mask_strategy_select(
            key=widget_key("probe", "mask_strategy"),
            last_key="probe:last_mask_strategy",
            remember_key="source:last_mask_strategy",
            help_text="Which extracted activation set to load.",
        )
        store = render_store_select(
            source,
            mask_strategy,
            state_prefix="probe",
            widget_scope="probe",
            artifacts_root_key="probe:local_root",
        )
        variant = _select_variant(store, mask_strategy)
        if variant is None:
            return
        persona_ids = _select_personas(store, variant, mask_strategy)
        if not persona_ids:
            return

    with st.expander("Probe configuration", expanded=True):
        attributes = _select_attributes()
        if not attributes:
            st.info("Select at least one attribute to probe.")
            return
        task = _attribute_tasks()[attributes[0]]
        st.caption(f"Inferred task: **{task}**")

        probe_kinds = _select_probe_kinds(task)
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
        seed = 0

    inputs = SweepInputs(
        source=source,
        location=location,
        model_name=model_name,
        mask_value=mask_strategy.value,
        variant=variant,
        persona_ids=tuple(persona_ids),
        attributes=tuple(attributes),
        task=task,
        probe_kinds=tuple(probe_kinds),
        n_pca_components=n_pca_components,
        layers=tuple(layers),
        min_class_count=min_class_count,
        seed=int(seed),
    )

    run = st.button("Run sweep", type="primary", key="probe:run")
    state_key = "probe:last_result"
    if run:
        with st.spinner("Evaluating probes across layers..."):
            try:
                sweep, per_attr = cached_sweep(inputs)
            except Exception as exc:
                st.error(f"Sweep failed: {exc}")
                st.session_state.pop(state_key, None)
                return
        st.session_state[state_key] = (sweep, per_attr, inputs)

    if state_key in st.session_state:
        saved_result = st.session_state[state_key]
        if len(saved_result) != 3:
            # Stale shape from a previous code version — drop it.
            st.session_state.pop(state_key, None)
        else:
            sweep, per_attr, result_inputs = saved_result
            _show_sweep(
                sweep,
                per_attr,
                result_inputs.attributes,
                result_inputs.task,
                result_inputs,
            )
