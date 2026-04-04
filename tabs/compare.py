import streamlit as st
from persona_data.environment import get_artifacts_dir
from persona_vectors.analysis import build_embedding_figure, project_pca, project_umap
from persona_vectors.plots import (
    plot_multiple_layer_similarities,
    save_plot_html,
    save_plot_png,
)

from utils.artifacts import (
    artifact_persona_options,
    list_available_layers,
    load_cosine_traces,
    load_embedding_samples,
)
from utils.helpers import (
    ANALYSIS_HELP_TEXT,
    ANALYSIS_LABELS,
    ANALYSIS_MODES,
    PROMPT_VARIANTS,
    persona_display_label,
    prompt_variant_label,
    slugify,
    widget_key,
)


def _filename(*parts: str) -> str:
    return "__".join(slugify(part) for part in parts if part)


def _select_artifact_personas(
    artifacts_root: str,
    model_name: str,
    variants: list[str],
) -> tuple[list[str], dict[str, str]]:
    persona_options, persona_names = artifact_persona_options(
        artifacts_root,
        model_name,
        variants,
    )
    if not persona_options:
        if len(variants) > 1:
            st.info(
                "No personas have saved activations for all selected variants. Run extraction for both variants first."
            )
        else:
            st.info("No personas found for this model yet. Run extraction first.")
        return [], persona_names

    persona_ids = st.multiselect(
        "Personas",
        options=persona_options,
        default=persona_options[:1] if len(persona_options) > 1 else persona_options,
        format_func=lambda persona_id: persona_display_label(
            persona_id, persona_names.get(persona_id)
        ),
        key=widget_key("load", "personas", model_name, *variants),
    )
    return persona_ids, persona_names


def _render_cosine_similarity(
    artifacts_root: str,
    model_name: str,
) -> None:
    col1, col2 = st.columns(2)
    with col1:
        variant_a = st.selectbox(
            "Variant A",
            options=PROMPT_VARIANTS,
            index=0,
            format_func=prompt_variant_label,
            key=widget_key("load", "variant_a"),
        )
    with col2:
        variant_b = st.selectbox(
            "Variant B",
            options=PROMPT_VARIANTS,
            index=min(1, len(PROMPT_VARIANTS) - 1),
            format_func=prompt_variant_label,
            key=widget_key("load", "variant_b"),
        )

    if variant_a == variant_b:
        st.warning("Choose two different variants to compare.")
        return

    persona_ids, _ = _select_artifact_personas(
        artifacts_root,
        model_name,
        [variant_a, variant_b],
    )
    if not persona_ids:
        return

    cosine_fig_key = widget_key("load", "cosine_fig_state", model_name)
    filename = _filename("compare", "cosine", model_name, variant_a, variant_b)

    if st.button("Compare vectors", type="primary"):
        traces, loaded_names, errors = load_cosine_traces(
            artifacts_root,
            model_name,
            persona_ids,
            variant_a,
            variant_b,
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
        fig = plot_multiple_layer_similarities(
            display_traces,
            title=f"{prompt_variant_label(variant_a)} vs {prompt_variant_label(variant_b)}",
            show=False,
        )
        st.session_state[cosine_fig_key] = (fig, len(traces))

    if cosine_fig_key in st.session_state:
        fig, n_traces = st.session_state[cosine_fig_key]
        st.plotly_chart(fig, use_container_width=True)
        save_col1, save_col2 = st.columns(2)
        with save_col1:
            if st.button("Save HTML", key=widget_key("load", "save_cosine_html")):
                output_path = save_plot_html(fig, filename)
                st.success(f"Saved HTML to `{output_path}`")
        with save_col2:
            if st.button("Save PNG", key=widget_key("load", "save_cosine_png")):
                try:
                    output_path = save_plot_png(fig, filename)
                    st.success(f"Saved PNG to `{output_path}`")
                except Exception as exc:
                    st.error(f"Could not save PNG: {exc}")
        st.success(f"Loaded {n_traces} personas for cosine comparison.")


def _render_embedding_analysis(
    artifacts_root: str,
    model_name: str,
    analysis_mode: str,
) -> None:
    selected_variant = st.selectbox(
        "Variant",
        options=PROMPT_VARIANTS,
        format_func=prompt_variant_label,
        key=widget_key("load", "variant"),
    )

    persona_ids, persona_names = _select_artifact_personas(
        artifacts_root,
        model_name,
        [selected_variant],
    )
    if not persona_ids:
        return

    layer_options = list_available_layers(
        artifacts_root,
        model_name,
        [selected_variant],
        persona_ids,
    )
    if not layer_options:
        st.info(
            "No shared layers are available for the selected personas. Try fewer personas or a different variant."
        )
        return

    persona_key = "_".join(sorted(persona_ids))
    layer_key = widget_key("load", "layers", model_name, selected_variant, persona_key)
    default_layers = [
        layer
        for layer in st.session_state.get(layer_key, layer_options[:3])
        if layer in layer_options
    ] or layer_options[:3]
    selected_layers = st.multiselect(
        "Layers",
        options=layer_options,
        default=default_layers,
        key=layer_key,
    )
    if not selected_layers:
        st.info("Select at least one layer.")
        return

    button_label = (
        "Generate PCA projection"
        if analysis_mode == "PCA"
        else "Generate UMAP projection"
    )

    embedding_fig_key = widget_key(
        "load", "embedding_fig_state", model_name, analysis_mode
    )

    if st.button(button_label, type="primary"):
        progress = st.progress(0, text="Preparing projections...")

        def update_progress(current: int, total: int, loaded: int) -> None:
            fraction = current / total if total else 1.0
            progress.progress(
                fraction,
                text=f"Processing layer {current}/{total} ({loaded} plot(s) ready)",
            )

        project_fn = project_pca if analysis_mode == "PCA" else project_umap
        try:
            plots, errors = load_embedding_samples(
                artifacts_root,
                model_name,
                persona_ids,
                selected_variant,
                selected_layers,
                project_fn,
                persona_names,
                progress_fn=update_progress,
            )

            if errors:
                for err in errors:
                    if (
                        "missing layer" in err
                        or "no selected personas have this layer" in err
                    ):
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
                title_prefix, x_label, y_label = ANALYSIS_LABELS[analysis_mode]
                rendered_figures: list[tuple[int, object]] = []
                for layer_idx, coords, labels, hover_text in plots:
                    fig = build_embedding_figure(
                        coords=coords,
                        labels=labels,
                        title=f"{title_prefix}, layer {layer_idx}",
                        x_label=x_label,
                        y_label=y_label,
                        hover_text=hover_text,
                    )
                    rendered_figures.append((layer_idx, fig))
                total_samples = sum(coords.shape[0] for _, coords, _, _ in plots)
                st.session_state[embedding_fig_key] = (
                    rendered_figures,
                    persona_key,
                    selected_variant,
                    total_samples,
                )
        finally:
            progress.empty()

    if embedding_fig_key in st.session_state:
        rendered_figures, saved_persona_key, saved_variant, total_samples = (
            st.session_state[embedding_fig_key]
        )
        cols = st.columns(2)
        for idx, (layer_idx, fig) in enumerate(rendered_figures):
            with cols[idx % 2]:
                st.plotly_chart(fig, use_container_width=True)
        st.success(
            f"Loaded {total_samples} samples across {len(rendered_figures)} layers."
        )
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
        save_col1, save_col2 = st.columns(2)
        with save_col1:
            if st.button(
                "Save HTML",
                key=widget_key("load", "save_embedding_html", analysis_mode),
            ):
                saved_paths = [
                    save_plot_html(fig, fn)
                    for (_, fig), fn in zip(rendered_figures, filenames)
                ]
                st.success(
                    f"Saved {len(saved_paths)} HTML plot(s) to `artifacts/plots`."
                )
        with save_col2:
            if st.button(
                "Save PNG",
                key=widget_key("load", "save_embedding_png", analysis_mode),
            ):
                try:
                    saved_paths = [
                        save_plot_png(fig, fn)
                        for (_, fig), fn in zip(rendered_figures, filenames)
                    ]
                    st.success(
                        f"Saved {len(saved_paths)} PNG plot(s) to `artifacts/plots`."
                    )
                except Exception as exc:
                    st.error(f"Could not save PNGs: {exc}")


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
        _render_cosine_similarity(artifacts_root, model_name)
        return

    _render_embedding_analysis(artifacts_root, model_name, analysis_mode)
