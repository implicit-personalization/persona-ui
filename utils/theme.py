"""Catppuccin Plotly template installer."""

import plotly.graph_objects as go
import plotly.io as pio
from catppuccin import PALETTE


def _flavor(base: str | None):
    return PALETTE.latte if base == "light" else PALETTE.mocha


def install_catppuccin_theme(base: str | None = None) -> None:
    """Register a Catppuccin template and alias it as ``plotly_white``.

    Call once at startup. Persona-vectors pins ``template="plotly_white"`` on
    every figure, so replacing that entry themes all plots without any
    per-figure code.
    """
    c = _flavor(base).colors
    bg, surface, line = c.base.hex, c.surface0.hex, c.surface1.hex
    text, subtext = c.text.hex, c.subtext1.hex

    axis = dict(
        gridcolor=line,
        zerolinecolor=line,
        linecolor=line,
        tickcolor=line,
        tickfont=dict(color=subtext),
        title=dict(font=dict(color=text)),
    )
    scene_axis = dict(
        backgroundcolor=bg,
        gridcolor=line,
        zerolinecolor=line,
        showbackground=True,
        color=text,
        tickfont=dict(color=subtext),
    )

    template = go.layout.Template(
        layout=dict(
            paper_bgcolor=bg,
            plot_bgcolor=bg,
            font=dict(color=text),
            colorway=[
                c.blue.hex,
                c.mauve.hex,
                c.green.hex,
                c.peach.hex,
                c.teal.hex,
                c.pink.hex,
                c.yellow.hex,
                c.sapphire.hex,
                c.lavender.hex,
                c.red.hex,
                c.sky.hex,
                c.maroon.hex,
            ],
            xaxis=axis,
            yaxis=axis,
            scene=dict(xaxis=scene_axis, yaxis=scene_axis, zaxis=scene_axis),
            legend=dict(bgcolor=surface, bordercolor=line, font=dict(color=text)),
            colorscale=dict(
                diverging=[[0.0, c.blue.hex], [0.5, surface], [1.0, c.red.hex]],
            ),
        )
    )
    pio.templates["catppuccin"] = template
    pio.templates["plotly_white"] = template
    pio.templates.default = "catppuccin"


def style_plotly_layer_controls(fig: go.Figure, base: str | None = None) -> go.Figure:
    """Theme persona-vectors layer sliders/buttons for this Streamlit app."""
    c = _flavor(base).colors
    surface = c.surface0.hex
    overlay = c.surface1.hex
    text = c.text.hex
    primary = c.blue.hex

    for slider in fig.layout.sliders:
        slider.bgcolor = surface
        slider.activebgcolor = primary
        slider.bordercolor = overlay
        slider.borderwidth = 1
        slider.font = dict(color=text, size=11)
        slider.tickcolor = primary
        slider.currentvalue = dict(
            slider.currentvalue.to_plotly_json(),
            font=dict(color=text, size=13),
        )

    for menu in fig.layout.updatemenus:
        if menu.type != "buttons":
            continue
        menu.bgcolor = surface
        menu.bordercolor = overlay
        menu.font = dict(color=text, size=13)

    return fig
