import json
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import geopandas as gpd

print("Lade Daten und initialisiere Dashboard...")

# Use a light Plotly template
pio.templates.default = "plotly_white"

# --------- THEME COLORS ---------
HEADER_BG = "#0F1A2A"
HEADER_BORDER = "#1F2A3A"
HEADER_TEXT_MAIN = "#FFFFFF"
HEADER_TEXT_SUB = "#D0D6E2"

SIDEBAR_BG = "#eef2ff"
SIDEBAR_BORDER = "#c7d2fe"

# --------- LAYOUT STYLES ---------
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": "130px",     # below header
    "left": 0,
    "bottom": 0,
    "width": "300px",
    "padding": "20px 15px",
    "backgroundColor": SIDEBAR_BG,
    "color": "#1e1b4b",
    "borderRight": f"1px solid {SIDEBAR_BORDER}",
    "overflowY": "auto",
    "overflowX": "hidden",
    "zIndex": 1,
    "boxShadow": "0px 4px 10px rgba(0,0,0,0.25)",
}

CONTENT_STYLE = {
    "marginLeft": "290px",
    "marginTop": "130px",
    "padding": "20px 20px 40px 20px",
    "backgroundColor": "#ffffff",
    "minHeight": "100vh",
}

CARD_STYLE = {
    "backgroundColor": "#e8f0fe",
    "borderRadius": "10px",
    "padding": "16px",
    "margin": "6px",
    "boxShadow": "0 1px 4px rgba(0,0,0,0.1)",
    "color": "#111827",
}

KPI_VALUE_STYLE = {"fontSize": "26px", "fontWeight": "bold"}
KPI_LABEL_STYLE = {"fontSize": "13px", "opacity": 0.8}
STANDARD_HEIGHT = 500
# --------- DATA META ---------
STATE_MAP = {
    1: "Schleswig-Holstein",
    2: "Hamburg",
    3: "Niedersachsen",
    4: "Bremen",
    5: "Nordrhein-Westfalen",
    6: "Hessen",
    7: "Rheinland-Pfalz",
    8: "Baden-Württemberg",
    9: "Bayern",
    10: "Saarland",
    11: "Berlin",
    12: "Brandenburg",
    13: "Mecklenburg-Vorpommern",
    14: "Sachsen",
    15: "Sachsen-Anhalt",
    16: "Thüringen",
}

AGE_COLS = {
    "Kinder <14": "Opfer Kinder bis 14 Jahre- insgesamt",
    "Jugendliche 14–<18": "Opfer Jugendliche 14 bis unter 18 Jahre - insgesamt",
    "Heranwachsende 18–<21": "Opfer - Heranwachsende 18 bis unter 21 Jahre - insgesamt",
    "Erwachsene 21–<60": "Opfer Erwachsene 21 bis unter 60 Jahre - insgesamt",
    "Senior:innen 60+": "Opfer - Erwachsene 60 Jahre und aelter - insgesamt",
}

# --------- LOAD DATA ---------
def load_data():
    dfs = []
    for year in range(2019, 2025):
        df = pd.read_csv(f"{year} Opfer.csv", sep=";", encoding="latin1")
        df.columns = [c.strip() for c in df.columns]  # removes trailing spaces
        df["Jahr"] = year
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["Bundesland_Code"] = (df_all["Gemeindeschluessel"] // 1000).astype(int)
    df_all["Bundesland"] = df_all["Bundesland_Code"].map(STATE_MAP)

    df_insg = df_all[df_all["Fallstatus"] == "insg."].copy()

    def short(s: str) -> str:
        s = s.strip()
        if "Gefährliche und schwere Körperverletzung" in s:
            return "Gefährliche/schwere KV"
        if "Vorsätzliche einfache Körperverletzung" in s:
            return "Einfache KV"
        if s.startswith("Mord Totschlag"):
            return "Mord/Totschlag"
        if "Vergewaltigung sexuelle Nötigung" in s:
            return "Sexualdelikte"
        if s.startswith("Straftaten insgesamt"):
            return "Straftaten insgesamt"
        return s[:45] + ("…" if len(s) > 45 else "")

    df_insg["Straftat_kurz"] = df_insg["Straftat"].apply(short)
    return df_insg


df = load_data()
YEARS = sorted(df["Jahr"].unique())
CRIME_SHORT = sorted(df["Straftat_kurz"].unique())
STATES = sorted(df["Bundesland"].dropna().unique())

# --------- HELPERS ---------
def filter_data(years, crimes, states):
    d = df
    if years:
        d = d[d["Jahr"].isin(years)]
    if crimes:
        d = d[d["Straftat_kurz"].isin(crimes)]
    if states:
        d = d[d["Bundesland"].isin(states)]
    return d


def empty_fig(msg="Keine Daten verfügbar"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def format_int(x):
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return "0"


# --------- KPI CALC ---------
def build_kpis(d):
    """
    KPIs:
    - Gesamtzahl der Opfer
    - Opfer pro Jahr (Ø)
    - männlich vs. weiblich (%)
    - Unter 18 vs. Erwachsene (%)
    - Anzahl Deliktsgruppen
    """
    if d.empty:
        return ("0", "0", "0 % / 0 %", "0 % / 0 %", "0")

    # 1) Gesamtzahl der Opfer
    total_victims = d["Oper insgesamt"].sum()

    # 2) Ø Opfer pro Jahr
    n_years = d["Jahr"].nunique()
    victims_per_year = int(round(total_victims / n_years)) if n_years > 0 else 0

    # 3) männlich vs. weiblich (%)
    male = d["Opfer maennlich"].sum() if "Opfer maennlich" in d.columns else 0
    female = d["Opfer weiblich"].sum() if "Opfer weiblich" in d.columns else 0
    sex_total = male + female

    def pct(part, whole):
        if whole <= 0:
            return "0,0 %"
        return f"{100 * part / whole:.1f} %".replace(".", ",")

    male_female_str = f"{pct(male, sex_total)} / {pct(female, sex_total)}"

    # 4) Unter 18 vs Erwachsene (%)
    col_children = "Opfer Kinder bis 14 Jahre- insgesamt"
    col_youth_14_18 = "Opfer Jugendliche 14 bis unter 18 Jahre - insgesamt"

    under18 = 0
    if col_children in d.columns:
        under18 += d[col_children].sum()
    if col_youth_14_18 in d.columns:
        under18 += d[col_youth_14_18].sum()

    adults = max(total_victims - under18, 0)
    under18_adults_str = f"{pct(under18, total_victims)} / {pct(adults, total_victims)}"

    # 5) Anzahl Deliktsgruppen (ohne 'Straftaten insgesamt')
    if "Straftat_kurz" in d.columns:
        crime_types = (
            d.loc[d["Straftat_kurz"] != "Straftaten insgesamt", "Straftat_kurz"]
            .nunique()
        )
    else:
        crime_types = 0

    return (
        format_int(total_victims),
        format_int(victims_per_year),
        male_female_str,
        under18_adults_str,
        str(crime_types),
    )


# --------- OVERVIEW FIGURES ---------
def fig_trend(d):
    if d.empty:
        return empty_fig()

    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"]
    g = d2.groupby("Jahr")["Oper insgesamt"].sum().reset_index()

    fig = px.line(
        g,
        x="Jahr",
        y="Oper insgesamt",
        markers=True,
        color_discrete_sequence=["#1f77b4"],
        title="Zeitliche Entwicklung der Opferzahlen",
        labels={"Oper insgesamt": "Opferzahl"},
    )
    return fig


def fig_top5(d):
    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"]
    if d2.empty:
        return empty_fig()
    g = (
        d2.groupby("Straftat_kurz")["Oper insgesamt"]
        .sum()
        .nlargest(5)
        .reset_index()
        .sort_values("Oper insgesamt")
    )
    fig = px.bar(
        g,
        x="Oper insgesamt",
        y="Straftat_kurz",
        orientation="h",
        color="Oper insgesamt",
        color_continuous_scale="YlOrRd",
        title="Top 5 Deliktsgruppen nach Opferzahl",
        labels={"Oper insgesamt": "Opferzahl", "Straftat_kurz": "Deliktsgruppe"},
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def fig_donut(d):
    """
    Statt Donut: Treemap zur Darstellung der Deliktsstruktur.
    Besser lesbar bei vielen Kategorien.
    """
    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"]
    if d2.empty:
        return empty_fig()

    g = d2.groupby("Straftat_kurz")["Oper insgesamt"].sum().reset_index()

    fig = px.treemap(
        g,
        path=["Straftat_kurz"],
        values="Oper insgesamt",
        color="Oper insgesamt",
        color_continuous_scale="Turbo",  # oder 'Viridis' für neutraler Look
        title="Struktur der Deliktsgruppen (Treemap)",
    )

    fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
    return fig


# --------- GEOGRAPHIC FIGURES ---------
# Modern interactive SHP-based Mapbox-style map
def fig_geo_map(d):
    if d.empty:
        return empty_fig()

    try:
        if not hasattr(fig_geo_map, "gdf"):
            print("Lade SHP für Bundesländer ...")
            # Adjust path if needed
            gdf = gpd.read_file("data/gadm41_DEU_1.shp")

            # Handle multi-polygons
            gdf = gdf.explode(index_parts=True).reset_index(drop=True)

            # Reproject to WGS84
            gdf = gdf.to_crs("EPSG:4326")

            # Bundesland name in this SHP:
            gdf["Bundesland"] = gdf["NAME_1"]

            fig_geo_map.gdf = gdf
        else:
            gdf = fig_geo_map.gdf
    except Exception as e:
        return empty_fig(f"Shapefile konnte nicht geladen werden: {e}")

    # Only victims (exclude Straftaten insgesamt)
    victims_df = d[d["Straftat_kurz"] != "Straftaten insgesamt"]
    victims = (
        victims_df.groupby("Bundesland")["Oper insgesamt"]
        .sum()
        .reset_index()
        .rename(columns={"Oper insgesamt": "Opfer_insgesamt"})
    )

    gdf_merged = gdf.merge(victims, on="Bundesland", how="left")
    gdf_merged = gdf_merged.dropna(subset=["Opfer_insgesamt"])

    geojson_data = json.loads(gdf_merged.to_json())

    fig = px.choropleth_mapbox(
        gdf_merged,
        geojson=geojson_data,
        locations=gdf_merged.index,
        color="Opfer_insgesamt",
        hover_name="Bundesland",
        hover_data={"Opfer_insgesamt": True},
        opacity=0.55,
        mapbox_style="open-street-map",  # no token required
        color_continuous_scale="Reds",
        zoom=4.5,
        center={"lat": 51.0, "lon": 10.2},
        title="Opfer nach Bundesland (Interaktive SHP-basierte Karte)",
    )

    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar_title="Opfer gesamt",
    )
    return fig


def fig_geo_state_bar(d):
    if d.empty:
        return empty_fig()
    g = d.groupby("Bundesland")["Oper insgesamt"].sum().reset_index()
    fig = px.bar(
        g,
        x="Bundesland",
        y="Oper insgesamt",
        color="Oper insgesamt",
        color_continuous_scale="Blues",
        title="Opfer nach Bundesland",
        labels={"Oper insgesamt": "Opferzahl"},
    )
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(coloraxis_showscale=False)
    return fig


def fig_geo_top(d):
    if d.empty:
        return empty_fig()
    g = (
        d.groupby(["Stadt/Landkreis", "Bundesland"])["Oper insgesamt"]
        .sum()
        .reset_index()
        .nlargest(10, "Oper insgesamt")
        .sort_values("Oper insgesamt")
    )
    g["Label"] = g["Stadt/Landkreis"] + " (" + g["Bundesland"] + ")"
    fig = px.bar(
        g,
        x="Oper insgesamt",
        y="Label",
        orientation="h",
        color="Oper insgesamt",
        color_continuous_scale="Reds",
        title="Top 10 Regionen nach Opferzahl",
        labels={"Oper insgesamt": "Opferzahl"},
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


# --------- CRIME TYPE FIGURES ---------
def fig_heatmap(d):
    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"]
    if d2.empty:
        return empty_fig()

    g = d2.groupby(["Straftat_kurz", "Jahr"])["Oper insgesamt"].sum().reset_index()

    fig = px.density_heatmap(
        g,
        x="Jahr",
        y="Straftat_kurz",
        z="Oper insgesamt",
        color_continuous_scale="Reds",
        title="Heatmap – Opferzahlen nach Deliktsgruppe und Jahr",
        labels={"Oper insgesamt": "Opferzahl", "Straftat_kurz": "Deliktsgruppe"},
    )

    fig.update_yaxes(autorange="reversed")

    # ✅ FORCE FULL SIZE
    fig.update_layout(
        height=750,
    )

    return fig


def fig_stacked(d):
    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"]
    if d2.empty:
        return empty_fig()
    top = d2.groupby("Straftat_kurz")["Oper insgesamt"].sum().nlargest(6).index
    d_top = d2[d2["Straftat_kurz"].isin(top)]
    g = d_top.groupby(["Jahr", "Straftat_kurz"])["Oper insgesamt"].sum().reset_index()
    fig = px.bar(
        g,
        x="Jahr",
        y="Oper insgesamt",
        color="Straftat_kurz",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Top-Deliktsgruppen im Zeitverlauf",
        labels={"Oper insgesamt": "Opferzahl", "Straftat_kurz": "Deliktsgruppe"},
    )
    return fig


def fig_age(d, crime):
    if d.empty:
        return empty_fig()
    d_sel = d[d["Straftat_kurz"] == crime]
    if d_sel.empty:
        return empty_fig("Keine Daten für diese Deliktsgruppe")
    vals = {lbl: d_sel[col].sum() for lbl, col in AGE_COLS.items() if col in d_sel}
    if not vals:
        return empty_fig("Keine Altersdaten verfügbar")
    fig = px.bar(
        x=list(vals.keys()),
        y=list(vals.values()),
        color=list(vals.values()),
        color_continuous_scale="Viridis",
        labels={"x": "Altersgruppe", "y": "Opferzahl"},
        title=f"Altersstruktur der Opfer – {crime}",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


# --------- TEMPORAL FIGURES ---------
def fig_state_trend(d):
    if d.empty:
        return empty_fig()
    top = d.groupby("Bundesland")["Oper insgesamt"].sum().nlargest(6).index
    d_top = d[d["Bundesland"].isin(top)]
    g = d_top.groupby(["Jahr", "Bundesland"])["Oper insgesamt"].sum().reset_index()
    fig = px.line(
        g,
        x="Jahr",
        y="Oper insgesamt",
        color="Bundesland",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set1,
        title="Ländervergleich im Zeitverlauf",
        labels={"Oper insgesamt": "Opferzahl"},
    )
    return fig


def fig_diverg(d):
    if d.empty:
        return empty_fig()
    years = sorted(d["Jahr"].unique())
    if len(years) < 2:
        return empty_fig("Mindestens zwei Jahre notwendig.")
    first, last = years[0], years[-1]
    g = d.groupby(["Bundesland", "Jahr"])["Oper insgesamt"].sum().reset_index()
    start = g[g["Jahr"] == first].set_index("Bundesland")["Oper insgesamt"]
    end = g[g["Jahr"] == last].set_index("Bundesland")["Oper insgesamt"]
    diff = (end - start).dropna().reset_index()
    diff.columns = ["Bundesland", "Delta"]
    diff = diff.sort_values("Delta")
    colors = ["#10b981" if x < 0 else "#ef4444" for x in diff["Delta"]]
    fig = go.Figure(
        go.Bar(
            x=diff["Delta"],
            y=diff["Bundesland"],
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(title=f"Veränderung der Opferzahlen {first} → {last}")
    return fig


def fig_gender(d):
    if d.empty:
        return empty_fig()
    g = d.groupby(["Stadt/Landkreis", "Bundesland"])[
        ["Opfer maennlich", "Opfer weiblich"]
    ].sum().reset_index()
    fig = px.scatter(
        g,
        x="Opfer maennlich",
        y="Opfer weiblich",
        color="Bundesland",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hover_name="Stadt/Landkreis",
        title="Geschlechtervergleich (m/w)",
    )
    return fig


# --------- DASH APP ---------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True,
)
app.title = "Crime Analysis Dashboard"


# --------- SIDEBAR ---------
def sidebar_layout(path):
    def nav_link(label, href):
        return dbc.NavLink(
            label,
            href=href,
            active=(path == href or (href == "/overview" and path in ("/", None))),
            className="w-100 text-start mb-1",
        )

    return html.Div(
        style=SIDEBAR_STYLE,
        children=[
            dbc.Card(
                body=True,
                style={"marginBottom": "12px"},
                children=[
                    html.H5("Navigation", className="card-title"),
                    dbc.Nav(
                        [
                            nav_link("Übersicht", "/overview"),
                            nav_link("Geografisch", "/geo"),
                            nav_link("Deliktskategorien", "/crime"),
                            nav_link("Zeitliche Einblicke", "/temporal"),
                        ],
                        vertical=True,
                        pills=True,
                    ),
                ],
            ),
            dbc.Card(
                body=True,
                children=[
                    html.H5("Analysefilter", className="card-title"),
                    html.Label("Jahr(e)", className="mt-2"),
                    dcc.Dropdown(
                        id="filter-year",
                        options=[{"label": str(y), "value": y} for y in YEARS],
                        value=YEARS,
                        multi=True,
                    ),
                    html.Label("Deliktsgruppen", className="mt-3"),
                    dcc.Dropdown(
                        id="filter-crime",
                        options=[{"label": c, "value": c} for c in CRIME_SHORT],
                        multi=True,
                        value=[],
                    ),
                    html.Label("Bundesland", className="mt-3"),
                    dcc.Dropdown(
                        id="filter-state",
                        options=[{"label": s, "value": s} for s in STATES],
                        multi=True,
                        value=[],
                    ),
                ],
            ),
        ],
    )


# --------- PAGE LAYOUTS ---------
def layout_overview():
    return html.Div(
        children=[
            html.H2("Übersicht", className="mb-3"),
            html.P(
                "Überblick über zentrale Kennzahlen, Trends und die Verteilung nach Deliktsgruppen.",
                className="text-muted",
            ),
            html.Div(
                style={"display": "flex", "flexWrap": "wrap"},
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Gesamtzahl der Opfer", style=KPI_LABEL_STYLE),
                            html.Div(id="kpi-total-victims", style=KPI_VALUE_STYLE),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Opfer pro Jahr (Ø)", style=KPI_LABEL_STYLE),
                            html.Div(
                                id="kpi-victims-per-year", style=KPI_VALUE_STYLE
                            ),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div(
                                "Verhältnis männlich / weiblich", style=KPI_LABEL_STYLE
                            ),
                            html.Div(id="kpi-male-female", style=KPI_VALUE_STYLE),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Unter 18 / Erwachsene", style=KPI_LABEL_STYLE),
                            html.Div(id="kpi-under18-adults", style=KPI_VALUE_STYLE),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Anzahl Deliktsgruppen", style=KPI_LABEL_STYLE),
                            html.Div(id="kpi-crime-types", style=KPI_VALUE_STYLE),
                        ],
                    ),
                ],
            ),
            html.Br(),
            dcc.Graph(id="trend"),
            html.Br(),
            dcc.Graph(id="top5"),
            html.Br(),
            dcc.Graph(id="donut"),
        ]
    )


def layout_geo():
    return html.Div(
        children=[
            html.H2("Geografische Analyse", className="mb-3"),
            html.P(
                "Vergleich der Opferzahlen nach Bundesland und Region.",
                className="text-muted",
            ),
            dcc.Graph(id="map"),
            html.Br(),
            dcc.Graph(id="statebar"),
            html.Br(),
            dcc.Graph(id="topregions"),
        ]
    )


def layout_crime():
    return html.Div(
        children=[
            html.H2("Crime Types (Deliktsstruktur)", className="mb-3"),
            html.P(
                "Analyse der Opferzahlen nach Deliktsgruppen sowie der Altersstruktur der Opfer.",
                className="text-muted",
            ),

            dcc.Graph(id="top5-crime", style={"width": "100%", "height": f"{STANDARD_HEIGHT}px"}),
            html.Br(),

            dcc.Graph(id="donut-crime", style={"width": "100%", "height": f"{STANDARD_HEIGHT}px"}),
            html.Br(),

            dcc.Graph(id="heat", style={"width": "100%", "height": f"{STANDARD_HEIGHT}px"}),
            html.Br(),

            dcc.Graph(id="stacked", style={"width": "100%", "height": f"{STANDARD_HEIGHT}px"}),
            html.Br(),

            html.Div(
                style={"maxWidth": "500px"},
                children=[
                    html.Label("Deliktsgruppe für Altersanalyse"),
                    dcc.Dropdown(
                        id="age-crime",
                        options=[{"label": c, "value": c} for c in CRIME_SHORT],
                        value="Straftaten insgesamt",
                        clearable=False,
                    ),
                ],
            ),
            html.Br(),

            dcc.Graph(id="agechart", style={"width": "100%", "height": f"{STANDARD_HEIGHT}px"}),
        ]
    )


def layout_temporal():
    return html.Div(
        children=[
            html.H2("Zeitliche Einblicke", className="mb-3"),
            html.P(
                "Dynamik der Opferzahlen im Ländervergleich sowie geschlechtsspezifische Muster.",
                className="text-muted",
            ),
            dcc.Graph(id="trendstates"),
            html.Br(),
            dcc.Graph(id="diverg"),
            html.Br(),
            dcc.Graph(id="gender"),
        ]
    )


# --------- ROOT LAYOUT (HEADER + SIDEBAR + CONTENT) ---------
app.layout = html.Div(
    children=[
        html.Div(
            style={
                "backgroundColor": HEADER_BG,
                "padding": "22px 30px",
                "paddingBottom": "30px",
                "borderBottom": f"1px solid {HEADER_BORDER}",
                "boxShadow": "0px 4px 10px rgba(0,0,0,0.25)",
                "position": "fixed",
                "top": 0,
                "left": 0,
                "right": 0,
                "zIndex": 1000,
                "textAlign": "left",
                "fontFamily": "Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            },
            children=[
                html.H1(
                    "Crime Analysis Dashboard",
                    style={
                        "fontSize": "28px",
                        "fontWeight": "700",
                        "color": HEADER_TEXT_MAIN,
                        "marginBottom": "4px",
                    },
                ),
                html.H4(
                    "Polizeiliche Kriminalstatistik Deutschland (2019–2024)",
                    style={
                        "fontSize": "18px",
                        "fontWeight": "450",
                        "color": HEADER_TEXT_SUB,
                        "marginTop": "0px",
                    },
                ),
            ],
        ),
        dcc.Location(id="url"),
        html.Div(id="sidebar"),
        html.Div(id="page-content", style=CONTENT_STYLE),
    ]
)


# --------- NAVIGATION CALLBACKS ---------
@app.callback(Output("sidebar", "children"), Input("url", "pathname"))
def update_sidebar(path):
    return sidebar_layout(path)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(path):
    if path in ("/", "/overview", None):
        return layout_overview()
    if path == "/geo":
        return layout_geo()
    if path == "/crime":
        return layout_crime()
    if path == "/temporal":
        return layout_temporal()
    return html.Div([html.H2("404 – Seite nicht gefunden")])


# --------- OVERVIEW CALLBACK ---------
@app.callback(
    Output("kpi-total-victims", "children"),
    Output("kpi-victims-per-year", "children"),
    Output("kpi-male-female", "children"),
    Output("kpi-under18-adults", "children"),
    Output("kpi-crime-types", "children"),
    Output("trend", "figure"),
    Output("top5", "figure"),
    Output("donut", "figure"),
    Input("filter-year", "value"),
    Input("filter-state", "value"),
)
def update_overview(years, states):
    d = filter_data(years or YEARS, [], states or [])
    (
        total_victims,
        victims_per_year,
        male_female,
        under18_adults,
        crime_types,
    ) = build_kpis(d)
    return (
        total_victims,
        victims_per_year,
        male_female,
        under18_adults,
        crime_types,
        fig_trend(d),
        fig_top5(d),
        fig_donut(d),
    )


# --------- GEOGRAPHIC CALLBACK ---------
@app.callback(
    Output("map", "figure"),
    Output("statebar", "figure"),
    Output("topregions", "figure"),
    Input("filter-year", "value"),
    Input("filter-crime", "value"),
    Input("filter-state", "value"),
)
def update_geo(years, crimes, states):
    d = filter_data(years or YEARS, crimes or [], states or [])
    return fig_geo_map(d), fig_geo_state_bar(d), fig_geo_top(d)


# --------- CRIME TYPES CALLBACK ---------
@app.callback(
    Output("heat", "figure"),
    Output("stacked", "figure"),
    Output("agechart", "figure"),
    Output("top5-crime", "figure"),   # NEW
    Output("donut-crime", "figure"),  # NEW
    Input("filter-year", "value"),
    Input("filter-crime", "value"),
    Input("filter-state", "value"),
    Input("age-crime", "value"),
)
def update_crime(years, crimes, states, age_crime_sel):
    d = filter_data(years or YEARS, crimes or [], states or [])

    heat_fig = fig_heatmap(d)
    stacked_fig = fig_stacked(d)
    age_fig = fig_age(d, age_crime_sel)

    top5_fig = fig_top5(d)      # reuse overview logic
    donut_fig = fig_donut(d)    # reuse overview logic

    return heat_fig, stacked_fig, age_fig, top5_fig, donut_fig


# --------- TEMPORAL CALLBACK ---------
@app.callback(
    Output("trendstates", "figure"),
    Output("diverg", "figure"),
    Output("gender", "figure"),
    Input("filter-year", "value"),
    Input("filter-crime", "value"),
    Input("filter-state", "value"),
)
def update_temporal(years, crimes, states):
    d = filter_data(years or YEARS, crimes or [], states or [])
    return fig_state_trend(d), fig_diverg(d), fig_gender(d)


if __name__ == "__main__":
    app.run(debug=True)
