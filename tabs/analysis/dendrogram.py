import streamlit as st
from persona_vectors.extraction import MaskStrategy
from persona_vectors.plots import plot_persona_dendrogram

from utils.analysis_sources import (
    Store,
    available_variants,
    store_cache_parts,
    store_id,
    store_layers_cached,
)
from utils.helpers import personas_fingerprint, prompt_variant_label, widget_key

from tabs.analysis._shared import (
    _load_persona_options,
    _load_variant_vectors,
    _plotly_chart,
    _release_vector_memory,
    _render_layer_frame_controls,
    _render_save_buttons,
    _select_artifact_personas,
)
from tabs.analysis._state import (
    _DEFAULT_PERSONA_LIMITS,
    PersonaOptions,
    _clear_old_figure_states,
    _filename,
    _persona_names_state_key,
    _personas_empty_message,
    _store_figure_state,
)

_LAST_DENDRO_PERSONAS_KEY = "analysis:last_personas:dendro"
_DENDRO_LINKAGE_OPTIONS = ["ward", "complete", "average", "single"]


def _render_persona_select_controls(
    options: PersonaOptions,
    widget_scope: str,
) -> list[str]:
    select_key = widget_key("load", "persona_select", widget_scope)
    assistant_key = widget_key("load", "persona_select_assistant", widget_scope)

    label_map = {
        pid: f"{options.persona_names.get(pid, pid)} ({pid})"
        for pid in options.regular_ids
    }
    sorted_labels = sorted(label_map.values())
    selected_labels = st.multiselect(
        "Select personas",
        options=sorted_labels,
        key=select_key,
        placeholder="Search and select personas...",
    )
    label_to_id = {v: k for k, v in label_map.items()}
    selected_ids = [label_to_id[lbl] for lbl in selected_labels]

    if options.assistant_id is not None:
        include_assistant = st.checkbox(
            "Include Assistant persona",
            key=assistant_key,
        )
        if include_assistant:
            selected_ids.append(options.assistant_id)

    st.session_state[_persona_names_state_key(widget_scope)] = dict(
        options.persona_names
    )

    if not selected_ids:
        st.info("Select at least one persona.")

    return selected_ids


def _render_dendrogram_analysis(
    store: Store,
    mask_strategy: MaskStrategy,
) -> None:
    variants = available_variants(store, mask_strategy)
    if not variants:
        st.info("No variants with saved vectors for this model.")
        return

    with st.expander("Variant selection", expanded=True):
        col1, col2 = st.columns(2)
        default_a = "biography" if "biography" in variants else variants[0]
        default_b_idx = (
            variants.index("templated")
            if "templated" in variants
            else min(1, len(variants) - 1)
        )
        with col1:
            variant_a = st.selectbox(
                "Variant A",
                options=variants,
                index=variants.index(default_a),
                format_func=prompt_variant_label,
                key=widget_key("load", "dendro_variant_a", store_id(store)),
            )
        with col2:
            variant_b = st.selectbox(
                "Variant B",
                options=variants,
                index=default_b_idx,
                format_func=prompt_variant_label,
                key=widget_key("load", "dendro_variant_b", store_id(store)),
            )

    shared_variants = list(dict.fromkeys([variant_a, variant_b]))

    select_specific = st.toggle(
        "Select specific personas",
        value=False,
        key=widget_key("load", "dendro_select_mode", store_id(store)),
        help="Search and select specific personas instead of using the first N.",
    )

    if select_specific:
        empty_message = _personas_empty_message(shared_variants)
        options = _load_persona_options(
            store,
            shared_variants,
            mask_strategy,
            empty_message=empty_message,
        )
        if options is None:
            st.session_state.pop(
                _persona_names_state_key(f"dendro:{store_id(store)}"), None
            )
            return
        persona_ids = _render_persona_select_controls(
            options,
            widget_scope=f"dendro:{store_id(store)}",
        )
        if not persona_ids:
            return
    else:
        persona_ids = _select_artifact_personas(
            store,
            shared_variants,
            mask_strategy,
            widget_scope=f"dendro:{store_id(store)}",
            remember_key=_LAST_DENDRO_PERSONAS_KEY,
            default_count_limit=_DEFAULT_PERSONA_LIMITS["dendro"],
        )
        if not persona_ids:
            return

    col_opts1, col_opts2 = st.columns(2)
    with col_opts1:
        layered_mode = st.toggle(
            "Per-layer animated",
            value=False,
            key=widget_key("load", "dendro_layered", store_id(store)),
            help="Animated dendrogram with one frame per layer instead of averaging all layers.",
        )
    with col_opts2:
        linkage = st.selectbox(
            "Linkage",
            options=_DENDRO_LINKAGE_OPTIONS,
            index=0,
            key=widget_key("load", "dendro_linkage", store_id(store)),
        )

    selected_layers: list[int] | None = None
    if layered_mode:
        source, location, model_name = store_cache_parts(store)
        layer_options = store_layers_cached(
            source,
            location,
            model_name,
            mask_strategy.value,
            tuple(shared_variants),
            tuple(persona_ids),
        )
        if not layer_options:
            st.info("No shared layers are available for the selected personas.")
            return
        selected_layers = _render_layer_frame_controls(store, "dendro", layer_options)

    persona_key = personas_fingerprint(persona_ids)
    fig_key = widget_key(
        "load",
        "dendro_fig_state",
        store_id(store),
        store.model_name,
        mask_strategy.value,
        variant_a,
        variant_b,
        persona_key,
        str(layered_mode),
        linkage,
        "_".join(map(str, selected_layers or [])),
    )
    _clear_old_figure_states(fig_key)

    if st.button(
        "Generate dendrograms",
        type="primary",
        key=widget_key(
            "load", "dendro_btn", store_id(store), variant_a, variant_b, persona_key
        ),
    ):
        progress = st.progress(0, text="Loading first variant vectors…")
        try:
            progress.progress(15, text="Loading variant vectors…")
            by_variant = _load_variant_vectors(
                store,
                shared_variants,
                mask_strategy,
                persona_ids,
            )
            samples_a = by_variant[variant_a]
            progress.progress(40, text="Building first dendrogram…")
            fig_a = plot_persona_dendrogram(
                samples_a,
                layered=layered_mode,
                layers=selected_layers,
                linkage=linkage,
                title=f"Dendrogram — {prompt_variant_label(variant_a)}",
            )
            fig_a.update_layout(height=750)
            del samples_a
            fig_b = None
            if variant_a != variant_b:
                progress.progress(60, text="Building second dendrogram…")
                samples_b = by_variant[variant_b]
                progress.progress(75, text="Building second dendrogram…")
                fig_b = plot_persona_dendrogram(
                    samples_b,
                    layered=layered_mode,
                    layers=selected_layers,
                    linkage=linkage,
                    title=f"Dendrogram — {prompt_variant_label(variant_b)}",
                )
                fig_b.update_layout(height=750)
                del samples_b
            progress.progress(90, text="Storing figure state…")
            _store_figure_state(
                fig_key,
                (fig_a, fig_b, len(persona_ids), variant_a, variant_b),
            )
            progress.progress(100, text="Done.")
        except Exception as exc:
            st.error(f"Could not build dendrogram: {exc}")
            st.session_state.pop(fig_key, None)
        finally:
            _release_vector_memory()
            progress.empty()

    if fig_key in st.session_state:
        fig_a, fig_b, n_personas, va, vb = st.session_state[fig_key]
        if fig_b is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader(prompt_variant_label(va))
                _plotly_chart(fig_a)
            with col_b:
                st.subheader(prompt_variant_label(vb))
                _plotly_chart(fig_b)
        else:
            _plotly_chart(fig_a)

        figs = [fig_a] + ([fig_b] if fig_b else [])
        filenames = [
            _filename("dendro", store.model_name, mask_strategy.value, va),
            *(
                [_filename("dendro", store.model_name, mask_strategy.value, vb)]
                if fig_b
                else []
            ),
        ]
        _render_save_buttons(figs, filenames, "dendro")
        st.success(f"Generated dendrogram(s) for {n_personas} persona(s).")
