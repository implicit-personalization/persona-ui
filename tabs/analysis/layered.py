import gc
from collections.abc import Callable

import plotly.graph_objects as go
import streamlit as st
from persona_vectors.attributes import attribute_color_kwargs, attribute_display_label
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import (
    build_layered_figure,
    build_pair_similarity_figure,
    build_similarity_figures,
)

from tabs.analysis._shared import (
    _gray_out_unselected_personas,
    _load_persona_vectors,
    _plotly_chart,
    _render_save_buttons,
    _select_single_variant_samples,
)
from tabs.analysis._state import (
    _CLUSTER_MODES,
    _DEFAULT_GRAPH_NEIGHBORS,
    _LAST_PROJECTION_ATTRIBUTE_KEY,
    _LAST_PROJECTION_CLUSTER_K_KEY,
    _LAST_PROJECTION_CLUSTER_MODE_KEY,
    _LAST_PROJECTION_COLOR_MODE_KEY,
    _LAST_PROJECTION_HIGHLIGHTS_KEY,
    _LAST_PROJECTION_PERSONAS_KEY,
    _LAST_PROJECTION_VARIANT_KEY,
    _LAST_SIMILARITY_VARIANT_KEY,
    _MAX_ATTRIBUTE_CATEGORIES,
    _MAX_PAIR_TRAJECTORY_TRACES,
    _MAX_SIMILARITY_CELLS,
    _PROJECTION_COLOR_MODES,
    _PROJECTION_KINDS,
    LayeredFigureStateKeys,
    ProjectionColorConfig,
    _clear_old_figure_states,
    _clear_old_prepared_states,
    _highlight_persona_groups,
    _persona_display_label,
    _persona_names_state_key,
    _remember_multiselect,
    _remembered_selectbox,
    _store_figure_state,
)
from utils.analysis_metadata import (
    synth_persona_attribute_names,
    synth_persona_dataset_cached,
)
from utils.analysis_sources import (
    Store,
    kmeans_groups_cached,
    projection_data_cached,
    store_cache_parts,
    store_id,
)
from utils.helpers import personas_fingerprint, prompt_variant_label, widget_key


def _render_pair_trajectory_control(
    *,
    enabled: bool,
    persona_count: int,
    scope: str,
    store: Store,
) -> bool:
    if not enabled:
        return False
    pair_count = persona_count * (persona_count - 1) // 2
    if pair_count > _MAX_PAIR_TRAJECTORY_TRACES:
        st.caption(
            "Pair trajectories hidden because this selection would create "
            f"{pair_count:,} Plotly traces."
        )
        return False
    return st.checkbox(
        "Pair trajectories",
        value=False,
        key=widget_key("load", "pair_trajectories", scope, store_id(store)),
        help="Adds one line per persona pair. Keep this off for larger selections.",
    )


def _validate_layered_figure_size(
    figure_kind: str,
    persona_count: int,
    selected_layers: list[int],
) -> bool:
    if figure_kind != "similarity":
        return True
    similarity_cells = persona_count * persona_count * len(selected_layers)
    if similarity_cells <= _MAX_SIMILARITY_CELLS:
        return True
    st.error(
        "Reduce personas or layer frames before generating the similarity "
        f"matrix ({similarity_cells:,} cells selected)."
    )
    return False


def _render_projection_color_config(
    store: Store,
    scope: str,
    persona_ids: list[str],
) -> ProjectionColorConfig | None:
    widget_scope = f"{scope}:{store_id(store)}"
    persona_key = personas_fingerprint(persona_ids)
    persona_names = st.session_state.get(
        _persona_names_state_key(widget_scope),
        {},
    )
    color_mode_key = widget_key("load", "color_mode", scope, store_id(store))
    color_mode = _remembered_selectbox(
        "Color by",
        key=color_mode_key,
        remember_key=_LAST_PROJECTION_COLOR_MODE_KEY,
        options=_PROJECTION_COLOR_MODES,
        default="Persona attribute",
    )
    if color_mode == "K-means clusters":
        max_clusters = min(10, len(persona_ids))
        if max_clusters < 2:
            st.info("Select at least two personas to use K-means coloring.")
            return None
        cluster_key = widget_key("load", "cluster_k", scope, store_id(store))
        default_clusters = min(3, len(persona_ids))
        if cluster_key not in st.session_state:
            st.session_state[cluster_key] = min(
                max(
                    int(
                        st.session_state.get(
                            _LAST_PROJECTION_CLUSTER_K_KEY,
                            default_clusters,
                        )
                    ),
                    2,
                ),
                max_clusters,
            )
        n_clusters = st.slider(
            "K (clusters)",
            min_value=2,
            max_value=max_clusters,
            key=cluster_key,
        )
        mode_key = widget_key("load", "cluster_mode", scope, store_id(store))
        mode_options = list(_CLUSTER_MODES)
        mode_label = _remembered_selectbox(
            "Cluster fit",
            key=mode_key,
            remember_key=_LAST_PROJECTION_CLUSTER_MODE_KEY,
            options=mode_options,
            default=mode_options[0],
            help=(
                "Mean across layers is the previous behavior. First selected "
                "layer keeps one fixed clustering from the first frame. Per layer "
                "recomputes clustering for each animation frame."
            ),
        )
        st.session_state[_LAST_PROJECTION_CLUSTER_K_KEY] = n_clusters
        return ProjectionColorConfig(
            color_mode=color_mode,
            n_clusters=n_clusters,
            cluster_mode=_CLUSTER_MODES[mode_label],
        )

    if color_mode == "Persona attribute":
        persona_dataset = synth_persona_dataset_cached()
        attribute_options = list(synth_persona_attribute_names())
        if not attribute_options:
            st.info("No persona attributes are available for this dataset.")
            return None
        default_attribute = (
            attribute_options.index("sex") if "sex" in attribute_options else 0
        )
        attribute_key = widget_key("load", "attribute", scope, store_id(store))
        attribute_name = _remembered_selectbox(
            "Attribute",
            key=attribute_key,
            remember_key=_LAST_PROJECTION_ATTRIBUTE_KEY,
            options=attribute_options,
            default=attribute_options[default_attribute],
            format_func=lambda name: attribute_display_label(persona_dataset, name),
        )
        info = persona_dataset.attribute_info(attribute_name)
        if info.get("high_cardinality"):
            st.caption(
                "High-cardinality categorical attributes are grouped to the "
                f"top {_MAX_ATTRIBUTE_CATEGORIES} values plus Other."
            )
        return ProjectionColorConfig(
            color_mode=color_mode,
            attribute_name=attribute_name,
        )

    highlight_persona_ids: tuple[str, ...] = ()
    if persona_ids:
        highlight_key = widget_key(
            "load", "persona_highlight", scope, store_id(store), persona_key
        )
        highlighted = st.multiselect(
            "Highlight personas",
            options=persona_ids,
            default=_remember_multiselect(
                key=highlight_key,
                remember_key=_LAST_PROJECTION_HIGHLIGHTS_KEY,
                options=persona_ids,
            ),
            format_func=lambda persona_id: _persona_display_label(
                persona_names, persona_id
            ),
            key=highlight_key,
            help=(
                "Select a few personas to keep their default colors while the rest "
                "are grayed out."
            ),
        )
        highlight_persona_ids = tuple(highlighted)
        st.session_state[_LAST_PROJECTION_HIGHLIGHTS_KEY] = list(highlighted)

    highlight_persona_key = (
        personas_fingerprint(highlight_persona_ids) if highlight_persona_ids else ""
    )

    return ProjectionColorConfig(
        color_mode=color_mode,
        highlight_persona_ids=highlight_persona_ids,
        highlight_persona_key=highlight_persona_key,
    )


def _layered_figure_state_keys(
    store: Store,
    mask_strategy: MaskStrategy,
    *,
    scope: str,
    figure_kind: str,
    n_components: int,
    color_config: ProjectionColorConfig,
    variant: str,
    persona_key: str,
    selected_layers: list[int],
    pair_trajectories: bool,
) -> LayeredFigureStateKeys:
    layer_key = "_".join(map(str, selected_layers))
    figure_key = widget_key(
        "load",
        f"{scope}_fig_state",
        store_id(store),
        store.model_name,
        mask_strategy.value,
        figure_kind,
        str(n_components),
        color_config.color_mode,
        str(color_config.attribute_name),
        str(color_config.n_clusters),
        str(color_config.cluster_mode),
        str(color_config.highlight_persona_key),
        variant,
        "persona_vector",
        persona_key,
        layer_key,
        str(pair_trajectories),
    )
    if figure_kind not in _PROJECTION_KINDS:
        return LayeredFigureStateKeys(figure=figure_key)
    prepared_key = widget_key(
        "load",
        f"{scope}_projection_ready",
        store_id(store),
        store.model_name,
        mask_strategy.value,
        figure_kind,
        str(n_components),
        str(figure_kind == "isomap"),
        str(_DEFAULT_GRAPH_NEIGHBORS),
        variant,
        persona_key,
        layer_key,
    )
    return LayeredFigureStateKeys(figure=figure_key, prepared=prepared_key)


def _projection_build_kwargs(
    *,
    store: Store,
    mask_strategy: MaskStrategy,
    variant: str,
    figure_kind: str,
    selected_layers: list[int],
    n_components: int,
    color_config: ProjectionColorConfig,
    persona_ids: list[str],
    persona_names: dict[str, str],
) -> dict:
    if figure_kind not in _PROJECTION_KINDS:
        return {}

    graph_overlay = figure_kind == "isomap"
    build_kwargs = {
        "n_components": n_components,
        "graph_overlay": graph_overlay,
        "graph_n_neighbors": _DEFAULT_GRAPH_NEIGHBORS,
    }
    source, location, model_name = store_cache_parts(store)
    cache_args = (
        source,
        location,
        model_name,
        mask_strategy.value,
        variant,
        tuple(persona_ids),
        tuple(selected_layers),
    )
    build_kwargs["projection_data"] = projection_data_cached(
        *cache_args,
        figure_kind,
        n_components,
        graph_overlay,
        _DEFAULT_GRAPH_NEIGHBORS,
    )
    if color_config.n_clusters is not None:
        build_kwargs["groups"] = kmeans_groups_cached(
            *cache_args,
            color_config.n_clusters,
            color_config.cluster_mode or "mean_across_layers",
        )
    if color_config.attribute_name is not None:
        build_kwargs.update(
            attribute_color_kwargs(
                synth_persona_dataset_cached(),
                color_config.attribute_name,
                persona_ids,
                max_categories=_MAX_ATTRIBUTE_CATEGORIES,
            )
        )
    if color_config.color_mode == "Persona" and color_config.highlight_persona_ids:
        groups = _highlight_persona_groups(
            persona_ids,
            persona_names,
            color_config.highlight_persona_ids,
        )
        if groups is not None:
            build_kwargs["groups"] = groups
    return build_kwargs


def _build_layered_analysis_figures(
    samples,
    *,
    figure_kind: str,
    selected_layers: list[int],
    variant: str,
    title_fn: Callable[[str], str],
    pair_trajectories: bool,
    build_kwargs: dict,
) -> tuple[go.Figure, go.Figure | None]:
    if figure_kind == "similarity" and pair_trajectories:
        return build_similarity_figures(
            samples,
            layers=selected_layers,
            title=title_fn(variant),
            pair_title=(
                "Pair similarity trajectories - "
                f"{prompt_variant_label(variant)} - persona vectors"
            ),
        )

    main_fig = build_layered_figure(
        samples,
        figure_kind,
        layers=selected_layers,
        title=title_fn(variant),
        **build_kwargs,
    )
    if figure_kind == "isomap":
        _add_isomap_connection_toggle(main_fig)
    if figure_kind in _PROJECTION_KINDS:
        main_fig.update_layout(height=700)
    extra_fig = (
        build_pair_similarity_figure(
            samples,
            layers=selected_layers,
            title=(
                "Pair similarity trajectories - "
                f"{prompt_variant_label(variant)} - persona vectors"
            ),
        )
        if pair_trajectories
        else None
    )
    return main_fig, extra_fig


def _add_isomap_connection_toggle(fig: go.Figure) -> None:
    """Add an in-plot control for the Isomap kNN graph trace."""
    if not fig.data or fig.data[0].name != "kNN graph":
        return

    existing_menus = tuple(fig.layout.updatemenus or ())
    fig.update_layout(
        updatemenus=existing_menus
        + (
            dict(
                type="buttons",
                direction="left",
                active=0,
                showactive=False,
                x=0,
                xanchor="left",
                y=1.16,
                yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(
                        label="Show connections",
                        method="restyle",
                        args=[{"visible": True}, [0]],
                    ),
                    dict(
                        label="Hide connections",
                        method="restyle",
                        args=[{"visible": False}, [0]],
                    ),
                ],
            ),
        ),
    )


def _render_layered_figure_analysis(
    store: Store,
    mask_strategy: MaskStrategy,
    *,
    scope: str,
    figure_kind: str,
    button_label: str,
    title_fn: Callable[[str], str],
    include_pair_trajectories: bool = False,
    n_components: int = 2,
    remember_key: str = _LAST_PROJECTION_PERSONAS_KEY,
    default_count_limit: int = 500,
    max_count_limit: int | None = None,
    allow_specific_personas: bool = False,
) -> None:
    """Render a single-variant layered analysis: select → button → figure(s).

    Used for similarity matrix, PCA, and UMAP. Set ``include_pair_trajectories``
    to add the pair-similarity-trajectory figure (similarity matrix only).
    """
    selected = _select_single_variant_samples(
        store,
        mask_strategy,
        scope,
        remember_key=remember_key,
        variant_remember_key=(
            _LAST_PROJECTION_VARIANT_KEY
            if figure_kind in _PROJECTION_KINDS
            else _LAST_SIMILARITY_VARIANT_KEY
        ),
        default_count_limit=default_count_limit,
        max_count_limit=max_count_limit,
        allow_specific_personas=allow_specific_personas,
    )
    if selected is None:
        return
    variant, persona_ids, persona_key, selected_layers = selected

    pair_trajectories = _render_pair_trajectory_control(
        enabled=include_pair_trajectories,
        persona_count=len(persona_ids),
        scope=scope,
        store=store,
    )
    if not _validate_layered_figure_size(
        figure_kind, len(persona_ids), selected_layers
    ):
        return

    color_config = ProjectionColorConfig()
    if figure_kind in _PROJECTION_KINDS:
        color_config = _render_projection_color_config(store, scope, persona_ids)
        if color_config is None:
            return

    state_keys = _layered_figure_state_keys(
        store,
        mask_strategy,
        scope=scope,
        figure_kind=figure_kind,
        n_components=n_components,
        color_config=color_config,
        variant=variant,
        persona_key=persona_key,
        selected_layers=selected_layers,
        pair_trajectories=pair_trajectories,
    )
    filename = scope
    _clear_old_figure_states(state_keys.figure)
    persona_names = st.session_state.get(
        _persona_names_state_key(f"{scope}:{store_id(store)}"),
        {},
    )

    build_clicked = st.button(button_label, type="primary")
    recolor_from_warm_projection = (
        state_keys.prepared is not None
        and bool(st.session_state.get(state_keys.prepared))
        and state_keys.figure not in st.session_state
    )
    if build_clicked or recolor_from_warm_projection:
        build_label = {
            "umap": "Computing UMAP projections…",
            "pca": "Computing PCA projections…",
            "isomap": "Computing Isomap projections…",
            "similarity": "Computing similarity matrices…",
        }.get(figure_kind, "Building figure…")
        progress = st.progress(0, text="Loading activation vectors…")
        try:
            progress.progress(15, text="Loading activation vectors…")
            samples = _load_persona_vectors(
                store,
                variant,
                mask_strategy,
                persona_ids,
            )
            progress.progress(55, text=build_label)
            build_kwargs = _projection_build_kwargs(
                store=store,
                mask_strategy=mask_strategy,
                variant=variant,
                figure_kind=figure_kind,
                selected_layers=selected_layers,
                n_components=n_components,
                color_config=color_config,
                persona_ids=persona_ids,
                persona_names=persona_names,
            )
            main_fig, extra_fig = _build_layered_analysis_figures(
                samples,
                figure_kind=figure_kind,
                selected_layers=selected_layers,
                variant=variant,
                title_fn=title_fn,
                pair_trajectories=pair_trajectories,
                build_kwargs=build_kwargs,
            )
            if (
                color_config.color_mode == "Persona"
                and color_config.highlight_persona_ids
            ):
                _gray_out_unselected_personas(main_fig)
            progress.progress(90, text="Storing figure state…")
            n_samples = samples.vectors.shape[0]
            del samples
            _store_figure_state(state_keys.figure, (main_fig, extra_fig, n_samples))
            if state_keys.prepared is not None:
                _clear_old_prepared_states(state_keys.prepared)
                st.session_state[state_keys.prepared] = True
            progress.progress(100, text="Done.")
        except Exception as exc:
            st.error(f"Could not build figure: {exc}")
            st.session_state.pop(state_keys.figure, None)
        finally:
            gc.collect()
            progress.empty()

    if state_keys.figure in st.session_state:
        main_fig, extra_fig, n_samples = st.session_state[state_keys.figure]
        _plotly_chart(main_fig)
        figs = [main_fig]
        filenames = [filename]
        if extra_fig is not None:
            st.subheader("Pair trajectories")
            _plotly_chart(extra_fig)
            figs.append(extra_fig)
            filenames.append(f"{filename}__pair_trajectories")
        _render_save_buttons(figs, filenames, scope)
        st.success(f"Loaded {n_samples} samples.")
