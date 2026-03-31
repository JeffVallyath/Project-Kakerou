"""Plotting helpers for the Kakerou Cockpit — refined minimal aesthetic."""

from __future__ import annotations

import altair as alt
import pandas as pd


def hypothesis_history_chart(history: list[dict]) -> alt.Chart:
    """Multi-line chart of hypothesis probabilities over turns."""
    if not history:
        return alt.Chart(pd.DataFrame()).mark_text()

    df = pd.DataFrame(history)

    lines = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=28), strokeWidth=1.5)
        .encode(
            x=alt.X("turn:Q", title=None,
                     axis=alt.Axis(tickMinStep=1, labelColor="#3a3a4a",
                                   gridColor="rgba(255,255,255,0.03)",
                                   domainColor="rgba(255,255,255,0.06)")),
            y=alt.Y("probability:Q", title=None,
                     scale=alt.Scale(domain=[0.0, 1.0]),
                     axis=alt.Axis(labelColor="#3a3a4a", format=".0%",
                                   gridColor="rgba(255,255,255,0.03)",
                                   domainColor="rgba(255,255,255,0.06)")),
            color=alt.Color(
                "hypothesis:N",
                title=None,
                scale=alt.Scale(
                    domain=sorted(df["hypothesis"].unique().tolist()),
                    range=["#f87171", "#60a5fa", "#34d399", "#fbbf24"],
                ),
                legend=alt.Legend(labelColor="#525264", labelFont="Söhne, -apple-system, sans-serif",
                                  labelFontSize=10),
            ),
            tooltip=["turn:Q", "hypothesis:N", alt.Tooltip("probability:Q", format=".1%")],
        )
    )

    baselines = df.drop_duplicates(subset=["hypothesis"])[["hypothesis", "baseline"]]
    baseline_rules = (
        alt.Chart(baselines)
        .mark_rule(strokeDash=[4, 4], opacity=0.15)
        .encode(
            y="baseline:Q",
            color=alt.Color("hypothesis:N", legend=None),
        )
    )

    chart = (lines + baseline_rules).properties(
        height=300,
    ).configure_axis(
        gridColor="rgba(255,255,255,0.03)",
        labelColor="#3a3a4a",
        titleColor="#525264",
    ).configure_title(
        color="#525264",
    ).configure_legend(
        labelColor="#525264",
        titleColor="#525264",
    ).configure_view(
        strokeWidth=0,
    )

    return chart


def signal_bar_chart(signals: dict[str, float], title: str = "Signal Values") -> alt.Chart:
    """Horizontal bar chart for signal values or reliabilities."""
    if not signals:
        return alt.Chart(pd.DataFrame()).mark_text()

    df = pd.DataFrame([
        {"signal": k.replace("_", " ").title(), "value": v}
        for k, v in signals.items()
    ])

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=2, height=8)
        .encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0.0, 1.0]), title=None,
                     axis=alt.Axis(labelColor="#3a3a4a", gridColor="rgba(255,255,255,0.03)",
                                   domainColor="rgba(255,255,255,0.06)")),
            y=alt.Y("signal:N", sort="-x", title=None,
                     axis=alt.Axis(labelColor="#525264", labelFont="Söhne, -apple-system, sans-serif",
                                   labelFontSize=10)),
            color=alt.condition(
                alt.datum.value > 0.7,
                alt.value("#f87171"),
                alt.value("#60a5fa"),
            ),
            tooltip=["signal:N", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=120, title=alt.TitleParams(title, color="#525264",
                                                       font="Söhne, -apple-system, sans-serif",
                                                       fontSize=11))
        .configure_axis(gridColor="rgba(255,255,255,0.03)", labelColor="#3a3a4a")
        .configure_view(strokeWidth=0)
    )

    return chart
