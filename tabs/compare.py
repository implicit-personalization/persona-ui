from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st
import torch
from persona_data.environment import get_artifacts_dir
from persona_vectors.analysis import build_embedding_figure, project_pca, project_umap
from persona_vectors.artifacts import SUPPORTED_VARIANTS, ActivationStore
from persona_vectors.artifacts import list_layers as list_available_layers
from persona_vectors.artifacts import list_personas as list_available_personas
from persona_vectors.artifacts import load_mean_activations, load_persona_names
from persona_vectors.plots import plot_layer_similarity, save_plot_html, save_plot_png

from utils.helpers import (
    ANALYSIS_HELP_TEXT,
    ANALYSIS_MODES,
    persona_display_label,
    prompt_variant_label,
    slugify,
    widget_key,
)


def _filename(*parts: str) -> str:
    return "__".join(slugify(part) for part in parts if part)


@dataclass(frozen=True)
class ProjectionConfig:
    title_prefix: str
    x_label: str
    y_label: str
    project_fn: Callable[[torch.Tensor], torch.Tensor]


_PROJECTION_CONFIGS: dict[str, ProjectionConfig] = {
    "PCA": ProjectionConfig("PCA", "PC1", "PC2", project_pca),
    "UMAP": ProjectionConfig("UMAP", "UMAP 1", "UMAP 2", project_umap),
}

_VARIANT_MODES: dict[str, tuple[str, ...]] = {
    "Template": ("templated",),
    "Biography": ("biography",),
    "Template + Biography": ("templated", "biography"),
}

_list_layers_cached = st.cache_data(show_spinner=False)(list_available_layers)

# Keep compare-tab selection state separate so projection defaults do not
# overwrite cosine similarity defaults.
_LAST_COSINE_PERSONAS_KEY = "compare:last_personas:cosine"
_LAST_PROJECTION_PERSONAS_KEY = "compare:last_personas:projection"
_LAST_PROJECTION_LAYERS_KEY = "compare:last_layers:projection"


def _load_mean_samples(
    store: ActivationStore,
    persona_ids: list[str],
    variants: list[str],
    selected_layers: list[int],
    persona_names: dict[str, str],
    project_fn: Callable[[torch.Tensor], torch.Tensor],
    multi_variant: bool,
    progress_fn: Callable[[int, int, int], None] | None = None,
) -> tuple[list[tuple[int, torch.Tensor, list[str], list[str]]], list[str]]:
    """Compute per-persona mean activations and project to 2D per layer."""

    plots: list[tuple[int, torch.Tensor, list[str], list[str]]] = []
    errors: list[str] = []
    loaded: dict[tuple[str, str], torch.Tensor] = {}

    for variant in variants:
        for persona_id in persona_ids:
            try:
                vectors, _ = store.load(variant, persona_id)
            except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
                errors.append(f"{persona_id} / {variant}: {exc}")
                continue
            loaded[(variant, persona_id)] = vectors

    total_layers = len(selected_layers)
    for idx, layer_idx in enumerate(selected_layers, start=1):
        samples: list[torch.Tensor] = []
        labels: list[str] = []
        hover_text: list[str] = []

        for variant in variants:
            variant_lbl = prompt_variant_label(variant)
            for persona_id in persona_ids:
                vectors = loaded.get((variant, persona_id))
                if vectors is None:
                    continue
                if layer_idx >= vectors.shape[1]:
                    errors.append(
                        f"{persona_id} / {variant}: missing layer {layer_idx}"
                    )
                    continue
                mean_vec = vectors[:, layer_idx, :].float().mean(dim=0)
                samples.append(mean_vec)
                display_name = persona_names.get(persona_id) or persona_id
                labels.append(variant_lbl if multi_variant else "Personas")
                hover_text.append(f"<b>{display_name}</b><br>{variant_lbl}")

        if len(samples) < 2:
            errors.append(
                f"Layer {layer_idx}: need at least 2 persona means to project"
            )
        else:
            try:
                stacked = torch.stack(samples, dim=0)
                coords = project_fn(stacked)
                plots.append((layer_idx, coords, labels, hover_text))
            except Exception as exc:
                errors.append(f"Layer {layer_idx}: {exc}")

        if progress_fn is not None:
            progress_fn(idx, total_layers, len(plots))

    return plots, errors


def _render_embedding_results(
    model_name: str,
    analysis_mode: str,
    rendered_figures: list[tuple[int, object]],
    saved_variant: str,
    saved_persona_key: str,
    total_samples: int,
) -> None:
    cols = st.columns(2)
    for idx, (_, fig) in enumerate(rendered_figures):
        with cols[idx % 2]:
            st.plotly_chart(fig, width="stretch")

    st.success(f"Loaded {total_samples} samples across {len(rendered_figures)} layers.")
    filenames = [
        _filename(
            "compare",
            analysis_mode,
            model_name,
            saved_variant,
            saved_persona_key,
            str(layer_idx),
        )
        for layer_idx, _ in rendered_figures
    ]
    _render_save_buttons([fig for _, fig in rendered_figures], filenames, analysis_mode)


def _select_artifact_personas(
    store: ActivationStore,
    variants: list[str],
    *,
    widget_scope: str,
    remember_key: str,
    default_all: bool = False,
) -> tuple[list[str], dict[str, str]]:
    persona_options = list_available_personas(
        store.root_dir, store.model_name, variants
    )
    persona_names = load_persona_names(
        store.root_dir, store.model_name, variants, persona_options
    )
    if not persona_options:
        if len(variants) > 1:
            st.info(
                "No personas have saved activations for all selected variants. Run extraction for both variants first."
            )
        else:
            st.info("No personas found for this model yet. Run extraction first.")
        return [], persona_names

    last_personas: list[str] = st.session_state.get(remember_key, [])
    default_personas = [p for p in last_personas if p in persona_options]
    if not default_personas:
        default_personas = persona_options if default_all else persona_options[:1]

    persona_key = widget_key(
        "load", "personas", widget_scope, store.model_name, *variants
    )

    def _remember_personas() -> None:
        st.session_state[remember_key] = [
            persona_id
            for persona_id in st.session_state.get(persona_key, [])
            if persona_id in persona_options
        ]

    persona_ids = st.multiselect(
        "Personas",
        options=persona_options,
        default=default_personas,
        format_func=lambda persona_id: persona_display_label(
            persona_id, persona_names.get(persona_id)
        ),
        key=persona_key,
        on_change=_remember_personas,
    )
    return persona_ids, persona_names


def _render_save_buttons(
    figs: list[object],
    filenames: list[str],
    key_suffix: str,
) -> None:
    """Render Save HTML / Save PNG column buttons for one or more figures."""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save HTML", key=widget_key("load", "save_html", key_suffix)):
            paths = [save_plot_html(fig, fn) for fig, fn in zip(figs, filenames)]
            st.success(f"Saved {len(paths)} HTML file(s) to `artifacts/plots`.")
    with col2:
        if st.button("Save PNG", key=widget_key("load", "save_png", key_suffix)):
            try:
                paths = [save_plot_png(fig, fn) for fig, fn in zip(figs, filenames)]
                st.success(f"Saved {len(paths)} PNG file(s) to `artifacts/plots`.")
            except Exception as exc:
                st.error(f"Could not save PNG: {exc}")


def _render_cosine_similarity(store: ActivationStore) -> None:
    col1, col2 = st.columns(2)
    with col1:
        variant_a = st.selectbox(
            "Variant A",
            options=SUPPORTED_VARIANTS,
            index=0,
            format_func=prompt_variant_label,
            key=widget_key("load", "variant_a"),
        )
    with col2:
        variant_b = st.selectbox(
            "Variant B",
            options=SUPPORTED_VARIANTS,
            index=min(1, len(SUPPORTED_VARIANTS) - 1),
            format_func=prompt_variant_label,
            key=widget_key("load", "variant_b"),
        )

    if variant_a == variant_b:
        st.warning("Choose two different variants to compare.")
        return

    persona_ids, _ = _select_artifact_personas(
        store,
        [variant_a, variant_b],
        widget_scope="cosine",
        remember_key=_LAST_COSINE_PERSONAS_KEY,
    )
    if not persona_ids:
        return

    cosine_fig_key = widget_key(
        "load", "cosine_fig_state", store.model_name, variant_a, variant_b
    )
    filename = _filename("compare", "cosine", store.model_name, variant_a, variant_b)

    if st.button("Compare vectors", type="primary"):
        traces, loaded_names, errors = load_mean_activations(
            store.root_dir, store.model_name, persona_ids, variant_a, variant_b
        )

        if errors:
            for err in errors:
                st.error(f"Failed to load vectors: `{err}`")
        if not traces:
            st.error("No personas loaded successfully.")
            st.info(
                "Check that extraction has been run for both variants and selected personas."
            )
            st.session_state.pop(cosine_fig_key, None)
            return

        display_traces = [
            (
                persona_display_label(persona_id, loaded_names.get(persona_id)),
                short,
                long,
            )
            for persona_id, short, long in traces
        ]
        fig = plot_layer_similarity(
            display_traces,
            title=f"{prompt_variant_label(variant_a)} vs {prompt_variant_label(variant_b)}",
            show=False,
        )
        st.session_state[cosine_fig_key] = (fig, len(traces))

    if cosine_fig_key in st.session_state:
        fig, n_traces = st.session_state[cosine_fig_key]
        st.plotly_chart(fig, width="stretch")
        _render_save_buttons([fig], [filename], "cosine")
        st.success(f"Loaded {n_traces} personas for cosine comparison.")


def _render_projection_analysis(store: ActivationStore, analysis_mode: str) -> None:
    projection_config = _PROJECTION_CONFIGS.get(analysis_mode)
    if projection_config is None:
        st.error(f"Unsupported analysis mode: {analysis_mode}")
        return

    mode_label = st.selectbox(
        "Variant",
        options=list(_VARIANT_MODES),
        key=widget_key("load", "variant_mode", analysis_mode),
    )
    variants = list(_VARIANT_MODES[mode_label])
    multi_variant = len(variants) > 1

    persona_ids, persona_names = _select_artifact_personas(
        store,
        variants,
        widget_scope="projection",
        remember_key=_LAST_PROJECTION_PERSONAS_KEY,
        default_all=True,
    )
    if not persona_ids:
        return

    persona_key = "_".join(sorted(persona_ids))
    layer_options = _list_layers_cached(
        str(store.root_dir),
        store.model_name,
        variants,
        persona_ids,
    )
    if not layer_options:
        st.info(
            "No shared layers are available for the selected personas. Try fewer personas or a different variant."
        )
        return

    layer_key = widget_key(
        "load",
        "layers",
        "projection",
        store.model_name,
        analysis_mode,
        mode_label,
        persona_key,
    )
    last_layers: list[int] = st.session_state.get(_LAST_PROJECTION_LAYERS_KEY, [])
    default_layers = [layer for layer in last_layers if layer in layer_options]
    if not default_layers:
        default_layers = layer_options

    def _remember_layers() -> None:
        st.session_state[_LAST_PROJECTION_LAYERS_KEY] = [
            layer
            for layer in st.session_state.get(layer_key, [])
            if layer in layer_options
        ]

    selected_layers = st.multiselect(
        "Layers",
        options=layer_options,
        default=default_layers,
        key=layer_key,
        on_change=_remember_layers,
    )
    if not selected_layers:
        st.info("Select at least one layer.")
        return

    embedding_fig_key = widget_key(
        "load", "projection_fig_state", store.model_name, analysis_mode
    )

    if st.button(f"Generate {analysis_mode} projection", type="primary"):
        progress = st.progress(0, text="Preparing projections...")

        def update_progress(current: int, total: int, loaded: int) -> None:
            fraction = current / total if total else 1.0
            progress.progress(
                fraction,
                text=f"Processing layer {current}/{total} ({loaded} plot(s) ready)",
            )

        try:
            plots, errors = _load_mean_samples(
                store,
                persona_ids,
                variants,
                selected_layers,
                persona_names,
                project_fn=projection_config.project_fn,
                multi_variant=multi_variant,
                progress_fn=update_progress,
            )

            if errors:
                for err in errors:
                    if "missing layer" in err:
                        st.warning(f"Skipping unavailable data: `{err}`")
                    else:
                        st.error(f"Failed to load vectors: `{err}`")
            if not plots:
                st.warning(
                    "No projections could be built for the current persona/layer selection."
                )
                st.info("Try fewer personas, fewer layers, or a different variant.")
                st.session_state.pop(embedding_fig_key, None)
            else:
                rendered_figures: list[tuple[int, object]] = []
                for layer_idx, coords, labels, hover_text in plots:
                    fig = build_embedding_figure(
                        coords=coords,
                        labels=labels,
                        title=f"{projection_config.title_prefix} (persona means), layer {layer_idx}",
                        x_label=projection_config.x_label,
                        y_label=projection_config.y_label,
                        hover_text=hover_text,
                    )
                    if multi_variant:
                        fig.update_traces(showlegend=True)
                    rendered_figures.append((layer_idx, fig))

                total_samples = sum(coords.shape[0] for _, coords, _, _ in plots)
                variant_key = "+".join(variants)
                st.session_state[embedding_fig_key] = (
                    rendered_figures,
                    persona_key,
                    variant_key,
                    total_samples,
                )
        finally:
            progress.empty()

    if embedding_fig_key in st.session_state:
        rendered_figures, saved_persona_key, saved_variant, total_samples = (
            st.session_state[embedding_fig_key]
        )
        _render_embedding_results(
            store.model_name,
            analysis_mode,
            rendered_figures,
            saved_variant,
            saved_persona_key,
            total_samples,
        )


def render_compare_tab(model_name: str) -> None:
    """Render the compare tab."""

    st.title("Compare")
    st.caption("Compare saved activations by cosine similarity, PCA, or UMAP.")

    st.subheader("Analysis")

    with st.expander("Advanced", expanded=False):
        artifacts_root = st.text_input(
            "Artifacts root",
            value=str(get_artifacts_dir() / "activations"),
        )

    store = ActivationStore(model_name, artifacts_root)

    analysis_mode = st.segmented_control(
        "Analysis mode",
        options=ANALYSIS_MODES,
        default=ANALYSIS_MODES[0],
        key=widget_key("load", "analysis_mode"),
        label_visibility="collapsed",
    )
    if analysis_mode is None:
        analysis_mode = ANALYSIS_MODES[0]
    st.caption(ANALYSIS_HELP_TEXT[analysis_mode])

    if analysis_mode == "Cosine similarity":
        _render_cosine_similarity(store)
        return

    _render_projection_analysis(store, analysis_mode)
