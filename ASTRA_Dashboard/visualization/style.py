import plotly.express as px
import plotly.graph_objects as go

# 2. Paleta de Cores de Alto Contraste & Design Tokens ASTRA
ASTRA_COLORS = {
    "bg_dark": "#0B1120",
    "card_dark": "#1E293B",
    "border_subtle": "#334155",
    "text_primary": "#F8FAFC",
    "text_muted": "#94A3B8",
    "accent_cyan": "#38BDF8",
    "accent_indigo": "#818CF8",
    "accent_emerald": "#34D399",
    "accent_amber": "#FBBF24",
    "accent_rose": "#F43F5E",
    "accent_violet": "#A78BFA",
    "accent_teal": "#2DD4BF"
}

CATEGORICAL_PALETTE = [
    ASTRA_COLORS["accent_cyan"],
    ASTRA_COLORS["accent_indigo"],
    ASTRA_COLORS["accent_emerald"],
    ASTRA_COLORS["accent_amber"],
    ASTRA_COLORS["accent_rose"],
    ASTRA_COLORS["accent_violet"],
    ASTRA_COLORS["accent_teal"]
]

# Função auxiliar para padronizar o estilo dos gráficos Plotly
def estilizar_grafico(fig, titulo=""):
    fig.update_layout(
        template="plotly_dark",
        title={
            "text": f"<b>{titulo}</b>",
            "font": {"size": 15, "color": "#F8FAFC", "family": "Plus Jakarta Sans"}
        },
        paper_bgcolor=ASTRA_COLORS["card_dark"],
        plot_bgcolor=ASTRA_COLORS["card_dark"],
        font=dict(color="#E2E8F0", family="Plus Jakarta Sans"),
        xaxis=dict(
            gridcolor=ASTRA_COLORS["border_subtle"],
            zerolinecolor=ASTRA_COLORS["border_subtle"],
            tickfont=dict(color="#94A3B8", size=11),
            title_font=dict(color="#CBD5E1", size=12)
        ),
        yaxis=dict(
            gridcolor=ASTRA_COLORS["border_subtle"],
            zerolinecolor=ASTRA_COLORS["border_subtle"],
            tickfont=dict(color="#94A3B8", size=11),
            title_font=dict(color="#CBD5E1", size=12)
        ),
        legend=dict(
            font=dict(color="#CBD5E1", size=11),
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor=ASTRA_COLORS["border_subtle"],
            borderwidth=1
        ),
        margin=dict(l=45, r=35, t=55, b=45)
    )
    return fig