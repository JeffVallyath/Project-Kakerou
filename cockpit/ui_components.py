"""UI components for the Kakerou Cockpit — refined minimal aesthetic."""

from __future__ import annotations

import html as _html
import streamlit as st


def system_status_card(turn: int, status: str, session_id: str) -> None:
    """Render the system status block."""
    status_colors = {
        "active": "#34d399",
        "insufficient_evidence": "#fbbf24",
        "noisy_input": "#f97316",
        "backend_error": "#f87171",
    }
    color = status_colors.get(status, "#6b7280")

    st.markdown(
        f"""
        <div style="
            background: #131316;
            border-left: 3px solid {color};
            border-right: 1px solid rgba(255,255,255,0.05);
            border-top: 1px solid rgba(255,255,255,0.05);
            border-bottom: 1px solid rgba(255,255,255,0.05);
            padding: 18px 22px;
            border-radius: 2px 12px 12px 2px;
            margin-bottom: 18px;
        ">
            <div style="color: #a1a1aa; font-size: 0.75em; text-transform: uppercase;
                        letter-spacing: 2px; font-family: 'DM Mono', 'SF Mono', monospace;">
                System Status
            </div>
            <div style="color: {color}; font-size: 0.95em; font-weight: 500; margin: 8px 0 6px 0;
                        font-family: 'DM Sans', -apple-system, sans-serif;">
                {status.upper().replace('_', ' ')}
            </div>
            <div style="color: #71717a; font-size: 0.7em;
                        font-family: 'DM Mono', 'SF Mono', monospace;">
                Turn {turn} &nbsp;&middot;&nbsp; {session_id[:12]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hypothesis_card(
    name: str,
    probability: float,
    baseline: float,
    momentum: float,
) -> None:
    """Render a single hypothesis card — clean, data-dense."""
    pct = probability * 100

    if probability > 0.7:
        accent = "#f87171"
    elif probability > 0.4:
        accent = "#fbbf24"
    else:
        accent = "#34d399"

    if momentum > 0.005:
        trend_icon = "&#8593;"
        m_color = "#f87171"
    elif momentum < -0.005:
        trend_icon = "&#8595;"
        m_color = "#34d399"
    else:
        trend_icon = "&#8212;"
        m_color = "#525264"

    display_name = name.replace("target_is_", "").replace("_", " ").title()

    delta_abs = abs(momentum)
    delta_display = f"{momentum:+.3f}" if delta_abs > 0.001 else "stable"

    st.markdown(
        f"""
        <div style="
            background: #131316;
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            padding: 22px 24px 18px;
            margin-bottom: 12px;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="color: #e4e4e7; font-weight: 500; font-size: 0.85em;
                                font-family: 'DM Sans', -apple-system, sans-serif;">{display_name}</div>
                    <div style="color: #71717a; font-size: 0.7em; margin-top: 2px;
                                font-family: 'DM Mono', 'SF Mono', monospace;">
                        baseline {baseline*100:.0f}%
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="color: {accent}; font-size: 1.8em; font-weight: 300;
                                font-family: 'DM Mono', 'SF Mono', monospace;
                                letter-spacing: -1px; line-height: 1;">{pct:.1f}%</div>
                    <div style="color: {m_color}; font-size: 0.7em; margin-top: 4px;
                                font-family: 'DM Mono', 'SF Mono', monospace;">
                        {trend_icon} {delta_display}
                    </div>
                </div>
            </div>
            <div style="
                background: rgba(255,255,255,0.05);
                border-radius: 4px;
                height: 8px;
                margin: 16px 0 0;
                overflow: hidden;
            ">
                <div style="
                    background: {accent};
                    height: 100%;
                    width: {pct:.1f}%;
                    border-radius: 4px;
                    transition: width 0.4s ease;
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def warning_banner(message: str, level: str = "warning") -> None:
    """Render a warning or error banner."""
    if level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.info(message)


def chat_message(speaker: str, text: str, turn_num: int | None = None) -> None:
    """Render a single chat message — minimal, type-led."""
    speaker_colors = {
        "Target": "#f87171",
        "User": "#60a5fa",
        "System": "#6b7280",
    }
    # Handle compound names like "Target (Alice)"
    base_speaker = speaker.split("(")[0].strip() if "(" in speaker else speaker
    color = speaker_colors.get(base_speaker, "#6b7280")
    turn_label = (
        f"<span style='color:#71717a; font-size:0.7em; "
        f"font-family: DM Mono, SF Mono, monospace;'>{turn_num}</span>&nbsp;&nbsp;"
        if turn_num else ""
    )

    st.markdown(
        f"""
        <div style="
            padding: 8px 0;
            margin-bottom: 1px;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            font-size: 0.85em;
            line-height: 1.5;
        ">
            {turn_label}<span style="color: {color}; font-weight: 500;
                                     font-family: 'DM Sans', -apple-system, sans-serif;">{speaker}</span>
            &nbsp;&nbsp;<span style="color: #a1a1aa;">{_html.escape(text)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
