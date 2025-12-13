import json
import re
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import geopandas as gpd
import numpy as np

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
CRIME_SYNONYMS = {
    # ===== HOMICIDE =====
    "Mord Totschlag und Tötung auf Verlangen": "Mord & Totschlag",
    "Mord": "Mord",
    "Totschlag": "Totschlag",

    # ===== SEXUAL CRIME =====
    "Vergewaltigung sexuelle Nötigung und sexueller Übergriff": "Sexualstraftaten",
    "Vergewaltigung sexuelle Nötigung und sexueller Übergriff im besonders schweren Fall": "Sexualstraftaten",
    "Sexueller Missbrauch von Kindern": "Missbrauch Kinder",

    # ===== ROBBERY =====
    "Raub räuberische Erpressung und räuberischer Angriff auf Kraftfahrer": "Raub & Erpressung",
    "Raub räuberische Erpressung auf/gegen Geldinstitute": "Raub Banken/Post",
    "Raub räuberische Erpressung auf/gegen sonstige Kassenräume und Geschäfte": "Raub Geschäfte",
    "Raub räuberische Erpressung auf/gegen sonstige Zahlstellen und Geschäfte": "Raub Geschäfte",
    "Handtaschenraub": "Handtaschenraub",
    "Sonstige Raubüberfälle auf Straßen": "Raub auf Straßen",
    "Raubüberfälle in Wohnungen": "Raub in Wohnungen",

    # ===== ASSAULT =====
    "Gefährliche und schwere Körperverletzung": "Schwere KV",
    "Vorsätzliche einfache Körperverletzung": "Einfache KV",

    # ===== POLICE OFFENCES =====
    "Widerstand gegen und tätlicher Angriff auf Vollstreckungsbeamte": "Widerstand/Angriff Beamte",
    "Widerstand gegen Vollstreckungsbeamte": "Widerstand gegen Beamte",
    "Tätlicher Angriff auf Vollstreckungsbeamte": "Angriff auf Beamte",

    # ===== OTHER =====
    "Gewaltkriminalität": "Gewaltkriminalität",
    "Diebstahl insgesamt": "Diebstahl",
    "Betrug insgesamt": "Betrug",
    "Computerbetrug": "Cyberbetrug",
    "Rauschgiftdelikte": "Drogendelikte",
    "Sachbeschädigung": "Sachbeschädigung",
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
    
    # Create a proper Region column (Stadt/Landkreis)
    if "Stadt/Landkreis" in df_all.columns:
        df_all["Region"] = df_all["Stadt/Landkreis"]
    else:
        # Fallback if column name is different
        df_all["Region"] = "Unbekannt"

    df_insg = df_all[df_all["Fallstatus"] == "insg."].copy()

    def short(s: str) -> str:
            s = s.strip()

            for long_name, short_name in CRIME_SYNONYMS.items():
                if long_name in s:
                    return short_name

    # fallback: clean & shorten safely if unknown
            return s.replace("  ", " ").strip()

    df_insg["Straftat_kurz"] = df_insg["Straftat"].apply(short)
    return df_insg


df = load_data()
YEARS = sorted(df["Jahr"].unique())
CRIME_SHORT = sorted(df["Straftat_kurz"].unique())
STATES = sorted(df["Bundesland"].dropna().unique())


# Show the longest crime names that are still used

# --------- LOAD GEO DATA ---------
print("Lade Geodaten...")
try:
    # Load state boundaries
    gdf_states = gpd.read_file("data/gadm41_DEU_1.shp")
    gdf_states = gdf_states.explode(index_parts=True).reset_index(drop=True)
    gdf_states = gdf_states.to_crs("EPSG:4326")
    gdf_states["Bundesland"] = gdf_states["NAME_1"]
    
    # Load city boundaries (level 2) - still needed for city view
    gdf_cities = gpd.read_file("data/gadm41_DEU_2.shp")
    gdf_cities = gdf_cities.explode(index_parts=True).reset_index(drop=True)
    gdf_cities = gdf_cities.to_crs("EPSG:4326")
    gdf_cities["Bundesland"] = gdf_cities["NAME_1"]
    gdf_cities["City"] = gdf_cities["NAME_2"]
    
    print(f"Geladen: {len(gdf_states)} Bundesländer, {len(gdf_cities)} Städte/Landkreise")
except Exception as e:
    print(f"Fehler beim Laden der Geodaten: {e}")
    gdf_states = None
    gdf_cities = None

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
        color_continuous_scale="Turbo",
        title="Struktur der Deliktsgruppen (Treemap)",
    )

    fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
    return fig

# --------- PIE CHART FOR OVERVIEW ---------
PKS_PIE_COLORS = [
    "#1e40af", # blue
    "#f59e42", # orange
    "#84cc16", # lime green
    "#f43f5e", # rose
    "#a21caf", # purple
    "#0ea5e9", # sky blue
    "#facc15", # yellow
    "#64748b", # slate
]

def fig_crime_pie(d):
    """
    Two-level pie chart like the reference figure:
    - Main chart (left): Top 10 crime categories + one slice "Andere".
    - Sub chart (right): Breakdown of the remaining categories (those inside "Andere").

    If there are <= 10 categories, we show a single pie.
    """
    if d.empty:
        return empty_fig()

    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"]
    if d2.empty:
        return empty_fig()

    g = (
        d2.groupby("Straftat_kurz")["Oper insgesamt"]
        .sum()
        .reset_index()
        .sort_values("Oper insgesamt", ascending=False)
    )

    # Keep top 10 for the main chart
    top_n = 10
    top = g.head(top_n).copy()
    rest = g.iloc[top_n:].copy()

    # If there is no remainder, show a single donut pie
    if rest.empty:
        fig = px.pie(
            top,
            names="Straftat_kurz",
            values="Oper insgesamt",
            hole=0.4,
            title="Anteile der Deliktsgruppen (Überblick)",
            color_discrete_sequence=PKS_PIE_COLORS,
        )
        fig.update_traces(
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Opfer: %{value:,}<br>%{percent}<extra></extra>",
        )
        fig.update_layout(height=STANDARD_HEIGHT, legend_title_text="Deliktsgruppe")
        return fig

    # Add "Andere" slice to the main chart
    other_sum = rest["Oper insgesamt"].sum()
    main_df = pd.concat(
        [
            top,
            pd.DataFrame({"Straftat_kurz": ["Andere"], "Oper insgesamt": [other_sum]}),
        ],
        ignore_index=True,
    )

    # Build a two-pie layout (main + sub)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        column_widths=[0.60, 0.40],
        horizontal_spacing=0.02,
        subplot_titles=(
            "Top 10 + Andere",
            "Aufschlüsselung von \"Andere\"",
        ),
    )

    # --- Colors ---
    # Main pie: use the PKS colors for the top 10; a neutral grey for "Andere"
    main_colors = (PKS_PIE_COLORS + px.colors.qualitative.Dark24)[: len(main_df)]
    if len(main_colors) >= 1:
        main_colors[-1] = "#cbd5e1"  # grey for "Andere"

    # Sub pie: use a bigger qualitative palette
    sub_colors = (px.colors.qualitative.Set3 + px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet)
    sub_colors = sub_colors[: len(rest)]

    # --- Main pie (left) ---
    fig.add_trace(
        go.Pie(
            labels=main_df["Straftat_kurz"],
            values=main_df["Oper insgesamt"],
            hole=0.4,
            textinfo="percent+label",
            marker=dict(colors=main_colors),
            sort=False,
            hovertemplate="<b>%{label}</b><br>Opfer: %{value:,}<br>%{percent}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # --- Sub pie (right): breakdown of remainder ---
    fig.add_trace(
        go.Pie(
            labels=rest["Straftat_kurz"],
            values=rest["Oper insgesamt"],
            hole=0.0,
            textinfo="percent+label",
            marker=dict(colors=sub_colors),
            sort=False,
            hovertemplate="<b>%{label}</b><br>Opfer: %{value:,}<br>%{percent}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # ---- Visual connector (wedge) between the two pies (paper coordinates) ----
    # Left pie domain will be roughly x in [0.0, ~0.60], right pie domain in [~0.62, 1.0]
    # We draw a light-grey wedge from the right edge of the left pie to the left edge of the right pie.
    x_left_edge = 0.60
    x_right_edge = 0.62
    y_top = 0.64
    y_bottom = 0.36
    x_apex = 0.52
    y_apex = 0.50

    # Filled wedge
    fig.add_shape(
        type="path",
        xref="paper",
        yref="paper",
        path=(
            f"M {x_apex},{y_apex} "
            f"L {x_left_edge},{y_top} "
            f"L {x_right_edge},{y_top} "
            f"L {x_right_edge},{y_bottom} "
            f"L {x_left_edge},{y_bottom} "
            f"Z"
        ),
        fillcolor="#cbd5e1",
        opacity=0.55,
        line=dict(color="#94a3b8", width=2),
        layer="above",
    )

    # Connector lines (to emulate the reference figure)
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=x_left_edge,
        y0=y_top,
        x1=x_right_edge,
        y1=y_top,
        line=dict(color="#94a3b8", width=2),
        layer="above",
    )
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=x_left_edge,
        y0=y_bottom,
        x1=x_right_edge,
        y1=y_bottom,
        line=dict(color="#94a3b8", width=2),
        layer="above",
    )

    # Optional: label in the wedge area (subtle)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.61,
        y=0.50,
        text="Andere",
        showarrow=False,
        font=dict(size=12, color="#475569"),
        bgcolor="rgba(255,255,255,0.0)",
    )

    fig.update_layout(
        title_text="Anteile der Deliktsgruppen (Überblick) – Top 10 + Aufschlüsselung",
        height=STANDARD_HEIGHT,
        legend_title_text="Deliktsgruppe",
        margin=dict(t=80, l=10, r=10, b=40),
    )

    return fig


# --------- GEOGRAPHIC FIGURES ---------
def prepare_state_geo_data(d, value_col="Oper insgesamt", age_group_col=None):
    """Prepare state-level geographic data for the given metric column."""
    if d.empty or gdf_states is None:
        return None, None

    victims_df = d[d["Straftat_kurz"] != "Straftaten insgesamt"]

    if value_col not in victims_df.columns:
        value_col = "Oper insgesamt"

    # Calculate total victims for each state
    victims = (
        victims_df.groupby("Bundesland")[value_col]
        .sum()
        .reset_index()
        .rename(columns={value_col: "Opfer_insgesamt"})
    )
    
    # Calculate age group victims if specified
    age_group_victims = None
    if age_group_col and age_group_col in victims_df.columns:
        age_group_victims = (
            victims_df.groupby("Bundesland")[age_group_col]
            .sum()
            .reset_index()
            .rename(columns={age_group_col: "Opfer_altersgruppe"})
        )
    
    # Merge with geo data
    gdf_merged = gdf_states.merge(victims, on="Bundesland", how="left")
    
    # Add age group data if available
    if age_group_victims is not None:
        gdf_merged = gdf_merged.merge(age_group_victims, on="Bundesland", how="left")
    else:
        gdf_merged["Opfer_altersgruppe"] = 0
    
    # Fill NaN values
    gdf_merged["Opfer_insgesamt"] = gdf_merged["Opfer_insgesamt"].fillna(0)
    gdf_merged["Opfer_altersgruppe"] = gdf_merged["Opfer_altersgruppe"].fillna(0)

    geojson_data = json.loads(gdf_merged.to_json())
    return gdf_merged, geojson_data

def _norm_admin_name(x: str) -> str:
    """Normalize German admin/city strings so Region names match shapefile city names better."""
    x = str(x).lower().strip()

    # Replace umlauts/eszett (common mismatch cause)
    x = (
        x.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )

    # Remove common administrative words that appear in CSV but not in shapes
    x = re.sub(
        r"\b(landkreis|kreisfreie\s+stadt|kreis|stadt|region|lk|sk|reg\.|bezirk)\b",
        " ",
        x,
    )

    # Remove punctuation and collapse whitespace
    x = re.sub(r"[^a-z0-9\s-]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def prepare_city_geo_data(d, selected_state=None, value_col="Oper insgesamt", age_group_col=None):
    """
    Prepare city-level geographic data.
    - selected_state = None  -> all cities in Germany
    - selected_state = Name  -> only cities of this state

    FIX:
    - match Region->City first
    - keep only matched cities
    - THEN apply Top-N later in fig_geo_map

    This prevents Top10 showing 9, Top20 showing 19, etc.
    """
    if d.empty or gdf_cities is None:
        return None, None, None

    if selected_state:
        state_data = d[d["Bundesland"] == selected_state].copy()
        gdf_subset = gdf_cities[gdf_cities["Bundesland"] == selected_state].copy()
    else:
        state_data = d.copy()
        gdf_subset = gdf_cities.copy()

    if state_data.empty or gdf_subset.empty:
        return None, None, None

    if value_col not in state_data.columns:
        value_col = "Oper insgesamt"

    # --- Aggregate by Region + Bundesland (prevents ambiguity like "Neustadt" in multiple states) ---
    city_victims = (
        state_data.groupby(["Region", "Bundesland"])[value_col]
        .sum()
        .reset_index()
        .rename(columns={value_col: "Opfer_insgesamt"})
    )

    # --- Age group victims (same grouping) ---
    if age_group_col and age_group_col in state_data.columns:
        age_group_city_victims = (
            state_data.groupby(["Region", "Bundesland"])[age_group_col]
            .sum()
            .reset_index()
            .rename(columns={age_group_col: "Opfer_altersgruppe"})
        )
        city_victims = city_victims.merge(
            age_group_city_victims, on=["Region", "Bundesland"], how="left"
        )
    else:
        city_victims["Opfer_altersgruppe"] = 0

    city_victims["Opfer_altersgruppe"] = city_victims["Opfer_altersgruppe"].fillna(0)

    # --- Build per-state normalized lookup for shapefile city names ---
    gdf_subset = gdf_subset.copy()
    gdf_subset["City_norm"] = gdf_subset["City"].apply(_norm_admin_name)

    # Dict: {Bundesland -> {City_norm -> City}}
    lookup_by_state = {}
    for bl, sub in gdf_subset.groupby("Bundesland"):
        lookup_by_state[bl] = dict(zip(sub["City_norm"], sub["City"]))

    def match_city(region_name: str, bundesland_name: str):
        r = _norm_admin_name(region_name)
        if not r:
            return None

        city_norm_to_city = lookup_by_state.get(bundesland_name, {})
        if not city_norm_to_city:
            return None

        # 1) exact normalized match
        if r in city_norm_to_city:
            return city_norm_to_city[r]

        # 2) substring fallback inside the same Bundesland
        for cn, real_city in city_norm_to_city.items():
            if r in cn or cn in r:
                return real_city

        return None

    # Match Region -> City within the same Bundesland (much higher accuracy)
    city_victims["City_match"] = city_victims.apply(
        lambda row: match_city(row["Region"], row["Bundesland"]), axis=1
    )

    # Debug: show unmatched Regions (enable if needed)
    # unmatched = city_victims[city_victims["City_match"].isna()][["Region"]].drop_duplicates()
    # print("Unmatched Regions (sample):", unmatched.head(30).to_string(index=False))

    # Keep only matched entries (these are drawable on the map)
    matched = city_victims.dropna(subset=["City_match"]).copy()
    if matched.empty:
        return None, None, None

    matched_city = (
        matched.groupby(["Bundesland", "City_match"])[["Opfer_insgesamt", "Opfer_altersgruppe"]]
        .sum()
        .reset_index()
        .rename(columns={"City_match": "City"})
    )

    # Merge into GeoDataFrame using BOTH keys (Bundesland + City)
    gdf_merged = gdf_subset.merge(matched_city, on=["Bundesland", "City"], how="left")
    gdf_merged["Opfer_insgesamt"] = gdf_merged["Opfer_insgesamt"].fillna(0)
    gdf_merged["Opfer_altersgruppe"] = gdf_merged["Opfer_altersgruppe"].fillna(0)

    # Calculate map center
    try:
        gdf_projected = gdf_merged.to_crs("EPSG:32632")
        centroid = gdf_projected.geometry.centroid
        centroid_wgs84 = centroid.to_crs("EPSG:4326")
        center_lat = centroid_wgs84.y.mean()
        center_lon = centroid_wgs84.x.mean()
    except Exception as e:
        print(
            f"Warning: Could not calculate proper centroid for {selected_state or 'Deutschland'}, using simple mean: {e}"
        )
        center_lat = gdf_merged.geometry.centroid.y.mean()
        center_lon = gdf_merged.geometry.centroid.x.mean()

    geojson_data = json.loads(gdf_merged.to_json())
    return gdf_merged, geojson_data, (center_lat, center_lon)

# ----- COLOR SCALES FOR SAFETY MODE -----
COLOR_SCALE_UNSAFE = "Reds"
COLOR_SCALE_SAFE = [
    [0.0, "#2ecc71"],   # green (safest)
    [0.5, "#f39c12"],   # orange (medium)
    [1.0, "#e74c3c"],   # red (least safe)
]
COLOR_SCALE_ALL = "OrRd"



def fig_geo_map(d, selected_state=None, city_mode="bundesland", age_group="all", safety_mode="all"):
    """
    Handles BOTH Bundesländer & City view with safety-mode coloring.
    safety_mode:
        - "safe"   → green scale (low = good)
        - "unsafe" → red scale (high = dangerous)
        - "all"    → neutral scale
    """

    if d.empty or gdf_states is None:
        return empty_fig("Keine Geodaten verfügbar")

    # ----- Select metric column (age-aware) -----
    value_col = "Oper insgesamt"
    age_group_col = None
    age_label_for_title = "alle Altersgruppen"
    if age_group != "all" and age_group in AGE_COLS:
        candidate = AGE_COLS[age_group]
        if candidate in d.columns:
            age_group_col = candidate
            age_label_for_title = age_group

    # Metric used for coloring + Top-N ranking
    metric_col = "Opfer_altersgruppe" if age_group_col is not None else "Opfer_insgesamt"
    metric_label = f"Opfer {age_label_for_title}" if age_group_col is not None else "Opfer gesamt"

    # ----- Choose color scale -----
    if safety_mode == "safe":
        color_scale = COLOR_SCALE_SAFE   # greens
        ascending = True                # safest first
    elif safety_mode == "unsafe":
        color_scale = COLOR_SCALE_UNSAFE  # reds
        ascending = False
    else:
        color_scale = COLOR_SCALE_ALL   # orange neutral
        ascending = False

    # ----------------------------------------------------
    # ✅ BUNDESLÄNDER VIEW ----------------------------------------------------
    # ----------------------------------------------------
    if city_mode == "bundesland" and selected_state is None:
        gdf_states_data, geojson_data = prepare_state_geo_data(d, value_col, age_group_col)
        if gdf_states_data is None:
            return empty_fig("Keine Bundeslanddaten verfügbar")

        # Sort by safe/unsafe using metric_col
        gdf_states_data = gdf_states_data.sort_values(metric_col, ascending=ascending)

        fig = px.choropleth_map(
            gdf_states_data,
            geojson=geojson_data,
            locations=gdf_states_data.index,
            color=metric_col,
            hover_name="Bundesland",
            hover_data={
                metric_col: True,
                "Opfer_insgesamt": True,
                "Opfer_altersgruppe": True,
                "Bundesland": False
            },
            custom_data=["Bundesland", metric_col, "Opfer_insgesamt", "Opfer_altersgruppe"],
            opacity=0.7,
            map_style="carto-positron",
            color_continuous_scale=color_scale,
            zoom=4.5,
            center={"lat": 51.0, "lon": 10.2},
            title=f"Opfer nach Bundesland – {age_label_for_title}",
        )

        # Hovertemplate: always show metric_label, total, and age group if selected
        if age_group != "all":
            fig.update_traces(
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    + f"{metric_label}: %{{customdata[1]:,.0f}}<br>"
                    + "Opfer gesamt: %{customdata[2]:,.0f}<br>"
                    + f"Opfer {age_label_for_title}: %{{customdata[3]:,.0f}}<br>"
                    + "<extra></extra>"
                )
            )
        else:
            fig.update_traces(
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    + f"{metric_label}: %{{customdata[1]:,.0f}}<br>"
                    + "<extra></extra>"
                )
            )

        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            height=500,
            clickmode="event+select"
        )

        return fig

    # ----------------------------------------------------
    # ✅ CITY VIEW (all Germany OR inside Bundesland)
    # ----------------------------------------------------
    gdf_cities_data, geojson_data, center = prepare_city_geo_data(d, selected_state, value_col, age_group_col)
    if gdf_cities_data is None:
        return empty_fig("Keine Städtedaten verfügbar")

    center_lat, center_lon = center

    gdf_plot = gdf_cities_data.copy()

    # For Top-N views, rank ONLY cities with data (avoid irrelevant 0-value polygons)
    if city_mode != "all" and isinstance(city_mode, int):
        gdf_rank = gdf_plot[gdf_plot[metric_col] > 0].copy()
        if gdf_rank.empty:
            return empty_fig("Keine Städtedaten verfügbar (nach Filter).")
        gdf_plot = gdf_rank.sort_values(metric_col, ascending=ascending).head(city_mode)

    fig = px.choropleth_map(
        gdf_plot,
        geojson=geojson_data,
        locations=gdf_plot.index,
        color=metric_col,
        hover_name="City",
        hover_data={
            metric_col: True,
            "Opfer_insgesamt": True,
            "Opfer_altersgruppe": True,
            "Bundesland": True,
            "City": False
        },
        custom_data=["City", "Bundesland", metric_col, "Opfer_insgesamt", "Opfer_altersgruppe"],
        opacity=0.8,
        map_style="carto-positron",
        color_continuous_scale=color_scale,
        zoom=6 if selected_state else 5,
        center={"lat": center_lat, "lon": center_lon},
        title=f"Opfer – Städteansicht – {age_label_for_title}",
    )

    # Hovertemplate: always show metric_label, total, and age group if selected
    if age_group != "all":
        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + "Bundesland: %{customdata[1]}<br>"
                + f"{metric_label}: %{{customdata[2]:,.0f}}<br>"
                + "Opfer gesamt: %{customdata[3]:,.0f}<br>"
                + f"Opfer {age_label_for_title}: %{{customdata[4]:,.0f}}<br>"
                + "<extra></extra>"
            )
        )
    else:
        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + "Bundesland: %{customdata[1]}<br>"
                + f"{metric_label}: %{{customdata[2]:,.0f}}<br>"
                + "<extra></extra>"
            )
        )

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=550,
        clickmode="none"
    )

    return fig

def fig_geo_state_bar(d):
    if d.empty:
        return empty_fig()

    # Remove total-crime category
    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"]

    # Aggregate only real crime categories
    g = (
        d2.groupby("Bundesland")["Oper insgesamt"]
        .sum()
        .reset_index()
        .sort_values("Oper insgesamt", ascending=True)
    )

    # Normalize values for smooth color scaling
    min_val = g["Oper insgesamt"].min()
    max_val = g["Oper insgesamt"].max()
    norm = (g["Oper insgesamt"] - min_val) / (max_val - min_val + 1e-9)

    # Create figure
    fig = go.Figure()

    # --- Lollipop stem (neutral color) ---
    fig.add_trace(
        go.Scatter(
            x=g["Oper insgesamt"],
            y=g["Bundesland"],
            mode="lines",
            line=dict(color="#bfbfbf", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # --- Lollipop dot (red intensity color) ---
    fig.add_trace(
        go.Scatter(
            x=g["Oper insgesamt"],
            y=g["Bundesland"],
            mode="markers",
            marker=dict(
                size=14,
                color=norm,                         # mapped intensity
                colorscale="Reds",                  # red scale
                showscale=False,                    # hide colorbar
                line=dict(color="black", width=0.6),
            ),
            hovertemplate="<b>%{y}</b><br>Opfer: %{x}<extra></extra>",
            showlegend=False,
        )
    )

    # Layout styling
    fig.update_layout(
        title="Opfer nach Bundesland",
        xaxis_title="Opferzahl (ohne 'Straftaten insgesamt')",
        yaxis_title="Bundesland",
        height=500,
        margin=dict(l=80, r=20, t=60, b=40),
        plot_bgcolor="white",
    )

    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=False)

    return fig




def fig_geo_top(d):
    if d.empty:
        return empty_fig()

    # Aggregate by city/region only
    g = (
        d.groupby("Region")["Oper insgesamt"]
        .sum()
        .reset_index()
        .nlargest(10, "Oper insgesamt")
        .sort_values("Oper insgesamt")
    )

    fig = px.bar(
        g,
        x="Oper insgesamt",
        y="Region",               # <-- Only city names
        orientation="h",
        color="Oper insgesamt",
        color_continuous_scale="Reds",
        title="Top 10 Städte / Regionen nach Opferzahl",
        labels={"Oper insgesamt": "Opferzahl", "Region": "Stadt / Region"},
    )

    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Opferzahl",
        yaxis_title="Stadt / Region",
        height=550,
        margin=dict(l=80, r=20, t=50, b=40),
    )

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
    g = d.groupby(["Region", "Bundesland"])[
        ["Opfer maennlich", "Opfer weiblich"]
    ].sum().reset_index()
    fig = px.scatter(
        g,
        x="Opfer maennlich",
        y="Opfer weiblich",
        color="Bundesland",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hover_name="Region",
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
                            nav_link("Trends", "/trends"),
                        ],
                        vertical=True,
                        pills=True,
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
            html.Br(),
            dcc.Graph(id="crime-pie"),
        ]
    )

def layout_geo():
    return html.Div(
        children=[
            html.H2("Geografische Analyse", className="mb-3"),
            html.P(
                "Vergleich der Opferzahlen nach Bundesland und Städten/Landkreisen. "
                "Klicken Sie auf ein Bundesland, um die Städteansicht aufzurufen.",
                className="text-muted",
            ),
            html.Div(
                id="state-info",
                style={
                    "backgroundColor": "#f0f9ff",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "marginBottom": "20px",
                    "borderLeft": "4px solid #3b82f6"
                },
                children=[
                    html.Div(
                        id="current-state-display",
                        children="Aktuelle Ansicht: Deutschland – Ebene: Bundesländer"
                    ),
                    html.Div(
                        id="state-back-button",
                        style={"display": "none"},
                        children=[
                            html.Button(
                                "← Zurück zur Deutschland-Ansicht",
                                id="back-to-germany",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#3b82f6",
                                    "color": "white",
                                    "border": "none",
                                    "padding": "5px 10px",
                                    "borderRadius": "3px",
                                    "cursor": "pointer",
                                    "marginTop": "10px"
                                }
                            )
                        ]
                    )
                ]
            ),

            # Store bleibt, weil wir weiterhin per Klick Bundesland auswählen
            dcc.Store(id="selected-state-store", data=None),

            # ===== FILTERLEISTE ÜBER DER KARTE =====
            html.Div(
                style={
                    "display": "flex",
                    "gap": "16px",
                    "marginBottom": "20px",
                    "padding": "12px",
                    "backgroundColor": "#eef2ff",
                    "borderRadius": "8px",
                    "border": "1px solid #c7d2fe",
                },
                children=[
                    # 1) City-Modus / Top N
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Auswahl"),
                            dcc.Dropdown(
                                id="geo-city-mode",
                                options=[
                                    {"label": "Bundesländer ", "value": "bundesland"},
                                    {"label": "Alle Städte", "value": "all"},
                                    {"label": "Top 10 Städte", "value": 10},
                                    {"label": "Top 20 Städte", "value": 20},
                                    {"label": "Top 50 Städte", "value": 50},
                                    {"label": "Top 100 Städte", "value": 100},
                                ],
                                value="bundesland",  # Standard: Bundesländer-Ansicht
                                clearable=False,
                            ),
                        ],
                    ),
                    # 2) Altersgruppe
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Altersgruppe"),
                            dcc.Dropdown(
                                id="geo-age-group",
                                options=(
                                    [{"label": "Alle Altersgruppen", "value": "all"}]
                                    + [
                                        {"label": label, "value": label}
                                        for label in AGE_COLS.keys()
                                    ]
                                ),
                                value="all",  # Standard: alle Altersgruppen
                                clearable=False,
                            ),
                        ],
                    ),
                    # 3) Safe / Unsafe
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Modus"),
                            dcc.Dropdown(
                            id="geo-safety-mode",
                             options=[
                            {"label": "Alle", "value": "all"},
                            {"label": "Gefährlich", "value": "unsafe"},
                            {"label": "Sicher", "value": "safe"},
                                      ],
                            value="all",
                            clearable=False,
                            ),

                        ],
                    ),
                ],
            ),
            # ===== ENDE FILTERLEISTE =====

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

def layout_trends():
    return html.Div(
        children=[
            html.H2("Trends", className="mb-3"),

            # ---------- TREND 1: Städte mit stärkstem Anstieg ----------
            html.H3("1. Which Cities Are Becoming More Dangerous? (2019–2024)"),
            html.P(
                "Städte mit dem stärksten Anstieg der Opferzahlen (2019–2024).",
                className="text-muted",
            ),

            html.Div(
                style={
                    "display": "flex",
                    "gap": "16px",
                    "maxWidth": "600px",
                    "marginBottom": "20px",
                },
                children=[
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Anzahl Städte"),
                            dcc.Dropdown(
                                id="city-count",
                                options=[
                                    {"label": "Top 5", "value": 5},
                                    {"label": "Top 10", "value": 10},
                                    {"label": "Top 15", "value": 15},
                                    {"label": "Top 20", "value": 20},
                                ],
                                value=10,
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Farbschema"),
                            dcc.Dropdown(
                                id="city-color-scale",
                                options=[
                                    {"label": "Orange-Rot (OrRd)", "value": "OrRd"},
                                    {"label": "Rot (Reds)", "value": "Reds"},
                                    {"label": "Grün-Blau (YlGnBu)", "value": "YlGnBu"},
                                    {"label": "Blau (Blues)", "value": "Blues"},
                                    {"label": "Lila (Purples)", "value": "Purples"},
                                    {"label": "Viridis", "value": "Viridis"},
                                ],
                                value="OrRd",
                                clearable=False,
                            ),
                        ],
                    ),
                ],
            ),

            dcc.Graph(
                id="city-danger",
                style={"width": "100%", "height": f"{STANDARD_HEIGHT}px"},
            ),

            html.Hr(style={"margin": "32px 0"}),

            # ---------- TREND 3: Kinder 0–14 ----------
            html.H3("2. Which Cities Are Safest / Most Dangerous for Children (0–14)?"),
            html.P(
                "Ranking der Städte nach Anzahl der Opfer im Alter von 0–14 Jahren.",
                className="text-muted",
            ),

            html.Div(
                style={
                    "display": "flex",
                    "gap": "16px",
                    "maxWidth": "600px",
                    "marginBottom": "20px",
                },
                children=[
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Anzahl Städte (Top N)"),
                            dcc.Dropdown(
                                id="trend-children-topn",
                                options=[
                                    {"label": "Top 5", "value": 5},
                                    {"label": "Top 10", "value": 10},
                                    {"label": "Top 15", "value": 15},
                                    {"label": "Top 20", "value": 20},
                                    {"label": "Top 30", "value": 30},
                                    {"label": "Top 50", "value": 50},
                                    {"label": "Top 100", "value": 100},
                                    {"label": "Alle Städte", "value": -1},
                                ],
                                value=-1,
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Altersgruppe"),
                            dcc.Dropdown(
                                id="trend-age-group",
                                options=(
                                    [{"label": "Kinder <14", "value": "Kinder <14"}]
                                    + [
                                        {"label": label, "value": label}
                                        for label in AGE_COLS.keys()
                                        if label != "Kinder <14"
                                    ]
                                ),
                                value="Kinder <14",
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Ansicht"),
                            dcc.Dropdown(
                                id="trend-children-mode",
                                options=[
                                    {"label": "Gefährlich", "value": "dangerous"},
                                    {"label": "Sicher", "value": "safe"},
                                ],
                                value="dangerous",
                                clearable=False,
                            ),
                        ],
                    ),
                ],
            ),

            dcc.Graph(
                id="trend-children-cities",
                style={
                     "width": "100%",
                     "maxWidth": "100%",              # ensure no internal limit
                     "height": f"{STANDARD_HEIGHT}px"
                 },
                config={"responsive": True},         # let Plotly stretch with container
            ),
            html.Br(),
            dcc.Graph(
                id="trend-children-bar",
                style={"width": "100%", "height": f"{STANDARD_HEIGHT}px"},
            ),

            html.H3("3. Steigt die Gewalt gegen Frauen an?"),
            dcc.Graph(id="trend-women-violence", style={"height": "500px"}),
        ]
    )

def fig_city_danger(d, top_n=10, color_scale="OrRd"):
    if d.empty:
        return empty_fig("Keine Daten verfügbar")

    g = d.groupby(["Region", "Jahr"])["Oper insgesamt"].sum().reset_index()
    years = sorted(g["Jahr"].unique())
    if len(years) < 2:
        return empty_fig("Mindestens zwei Jahre notwendig (z.B. 2019 und 2024).")

    first, last = years[0], years[-1]

    start = g[g["Jahr"] == first].set_index("Region")["Oper insgesamt"]
    end = g[g["Jahr"] == last].set_index("Region")["Oper insgesamt"]

    diff = (end - start).dropna().reset_index()
    diff.columns = ["Region", "Delta"]

    diff = diff[diff["Delta"] > 0]
    diff = diff.sort_values("Delta", ascending=False)

    if top_n is not None and top_n > 0:
        diff = diff.head(top_n)

    fig = px.bar(
        diff,
        x="Delta",
        y="Region",
        orientation="h",
        color="Delta",
        color_continuous_scale=color_scale,
        labels={
            "Delta": f"Zunahme Opfer {first}–{last}",
            "Region": "Region / Stadt",
        },
        title=(
            f"Städte mit größtem Opferanstieg ({first}–{last}) – Alle Städte"
            if top_n == -1
            else f"Städte mit größtem Opferanstieg ({first}–{last}) – Top {top_n}"
        ),
    )

    # 🔑 This line makes the highest value appear at the TOP
    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        coloraxis_showscale=False,
        height=STANDARD_HEIGHT,
    )

    return fig


# Which city is safer or dangerous for children function (now as risk scatter)
def fig_children_ranking(d, top_n=10, mode="dangerous", age_group="Kinder <14"):
    """
    Karte für Kinderopfer (0–14) auf Stadt-/Landkreisebene
    mit gleichem Stil wie die Geografie-Ansicht (px.choropleth_map).

    mode = "dangerous" -> Städte mit den meisten Kinderopfern (Top N, rot)
    mode = "safe"      -> Städte mit den wenigsten Kinderopfern (Top N, grün)
    """
    if d.empty or gdf_cities is None:
        return empty_fig("Keine Geodaten für Kinder (0–14) verfügbar.")

    # Selected age group column (falls back to Kinder <14)
    if age_group not in AGE_COLS:
        age_group = "Kinder <14"
    col_children = AGE_COLS[age_group]
    if col_children not in d.columns:
        return empty_fig(f"Keine Daten für {age_group} verfügbar.")

    # --- Kinderopfer nach Region + Bundesland ---
    g_children = (
        d.groupby(["Region", "Bundesland"])[col_children]
        .sum()
        .reset_index()
        .rename(columns={col_children: "Kinder_0_14"})
    )

    if g_children.empty:
        return empty_fig("Zu wenige Daten für Kinder (0–14).")

    # --- Gesamtopfer nach Region + Bundesland (für Hover) ---
    g_total = (
        d.groupby(["Region", "Bundesland"])["Oper insgesamt"]
        .sum()
        .reset_index()
        .rename(columns={"Oper insgesamt": "Gesamtopfer"})
    )

    g = g_children.merge(g_total, on=["Region", "Bundesland"], how="left")
    g["Gesamtopfer"] = g["Gesamtopfer"].fillna(0)

    # Anteil Kinder an allen Opfern (in %), nur für Tooltip
    g["Anteil_Kinder"] = np.where(
        g["Gesamtopfer"] > 0,
        100 * g["Kinder_0_14"] / g["Gesamtopfer"],
        0.0,
    )

    # --- Sortierung nach Modus (gefährlich/sicher) ---
    ascending = True if mode == "safe" else False
    g = g.sort_values("Kinder_0_14", ascending=ascending)

    # Top N auswählen (oder alle, falls -1)
    if top_n is not None and top_n > 0:
        g = g.head(top_n)

    if g.empty:
        return empty_fig("Keine Städte für diese Auswahl gefunden.")

    # --- Zuordnung Region ↔ Stadt-Shapes (gdf_cities) ---
    gdf = gdf_cities.copy()
    gdf["Kinder_0_14"] = np.nan
    gdf["Gesamtopfer"] = np.nan
    gdf["Anteil_Kinder"] = np.nan

    def find_matching_city_idx(region_name, bundesland_name):
        """Versucht, eine Stadt im Shape-File zu finden, die zur Region passt."""
        region_lower = str(region_name).lower()
        subset = gdf[gdf["Bundesland"] == bundesland_name]
        for idx, city in subset["City"].items():
            city_lower = str(city).lower()
            if region_lower in city_lower or city_lower in region_lower:
                return idx
        return None

    # Werte in GeoDataFrame schreiben (nur Top-N-Städte)
    for _, row in g.iterrows():
        idx = find_matching_city_idx(row["Region"], row["Bundesland"])
        if idx is not None:
            gdf.at[idx, "Kinder_0_14"] = row["Kinder_0_14"]
            gdf.at[idx, "Gesamtopfer"] = row["Gesamtopfer"]
            gdf.at[idx, "Anteil_Kinder"] = row["Anteil_Kinder"]

    # Nur Städte mit zugeordneten Kinderwerten behalten
    gdf_plot = gdf.dropna(subset=["Kinder_0_14"]).copy()
    if gdf_plot.empty:
        return empty_fig("Keine Zuordnung Stadt ↔ Region möglich.")

    # ID-Spalte für Verbindung GeoJSON ↔ DataFrame
    gdf_plot = gdf_plot.reset_index().rename(columns={"index": "id"})
    geojson_data = json.loads(gdf_plot.to_json())

    # Farbskala nach Modus
    # Semantic danger scale: green → yellow → orange → red
    danger_scale = [
        [0.0, "#2ecc71"],  # green (lowest danger)
        [0.33, "#f1c40f"], # yellow
        [0.66, "#f39c12"], # orange
        [1.0, "#e74c3c"],  # red (highest danger)
    ]

    # Context-dependent title
    if mode == "safe":
        title_mode = "sichersten (wenigste Opfer)"
    else:
        title_mode = "gefährlichsten (meiste Opfer)"

    # 🔵 Neue Map im Stil des Dash-Beispiels: px.choropleth_map
    fig = px.choropleth_map(
        gdf_plot,
        geojson=geojson_data,
        locations="id",
        featureidkey="properties.id",
        color="Kinder_0_14",
        hover_name="City",
        hover_data={
            "Kinder_0_14": True,
            "Gesamtopfer": True,
            "Anteil_Kinder": ":.1f",
            "Bundesland": True,
        },
        color_continuous_scale=danger_scale,
        labels={"Kinder_0_14": f"Opfer ({age_group})"},
        title=f"Top {top_n} {title_mode} Städte – Opfer ({age_group})",
        center={"lat": 51.0, "lon": 10.2},
        zoom=4.5,
        map_style="carto-positron",  # gleiche Stil-Familie wie moderne Dash-Beispiele
    )

    fig.update_layout(
        height=STANDARD_HEIGHT,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar_title=f"Opfer ({age_group})",
    )

    return fig


# --------- Helper: Bar chart for children 0–14 Top-N ---------
def fig_children_bar(d, top_n=10, mode="dangerous", age_group="Kinder <14"):
    """
    Bar chart for the same Top-N selection as fig_children_ranking:
    - mode='dangerous' -> highest Kinder_0_14
    - mode='safe'      -> lowest non-zero Kinder_0_14
    """
    # Default behavior: keep bar chart empty until user selects a specific Top-N
    # (the dropdown default is -1 = "Alle Städte")
    if top_n in (None, -1):
        return empty_fig("Bitte eine Top-N Auswahl treffen, um das Balkendiagramm zu sehen.")
    if d.empty:
        return empty_fig("Keine Daten verfügbar")

    # Selected age group column (falls back to Kinder <14)
    if age_group not in AGE_COLS:
        age_group = "Kinder <14"
    col_children = AGE_COLS[age_group]
    if col_children not in d.columns:
        return empty_fig(f"Keine Daten für {age_group} verfügbar.")

    g = (
        d.groupby(["Region", "Bundesland"])[col_children]
        .sum()
        .reset_index()
        .rename(columns={col_children: "Kinder_0_14"})
    )

    # Keep only cities with data (avoid irrelevant zeros for safe mode)
    g = g[g["Kinder_0_14"] > 0].copy()
    if g.empty:
        return empty_fig("Zu wenige Daten für Kinder (0–14).")

    ascending = True if mode == "safe" else False
    g = g.sort_values("Kinder_0_14", ascending=ascending)

    bar_n = top_n or 10

    if bar_n is not None and bar_n > 0:
        g = g.head(bar_n)

    # Title
    if mode == "safe":
        title_mode = "Sicherste Städte (wenigste Opfer)"
    else:
        title_mode = "Gefährlichste Städte (meiste Opfer)"

    title = f"{title_mode} – Opfer ({age_group}) – Top {bar_n}"

    # Use same color logic as the map
    if mode == "safe":
        bar_color_scale = COLOR_SCALE_SAFE
    else:
        bar_color_scale = COLOR_SCALE_UNSAFE

    fig = px.bar(
        g,
        x="Kinder_0_14",
        y="Region",
        orientation="h",
        color="Kinder_0_14",
        color_continuous_scale=bar_color_scale,
        hover_data={"Bundesland": True, "Kinder_0_14": True},
        labels={"Kinder_0_14": f"Opfer ({age_group})", "Region": "Stadt / Region"},
        title=title,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(coloraxis_showscale=False, height=STANDARD_HEIGHT)
    return fig

# viollence agains Women over time 
def fig_violence_women(d):
    if d.empty:
        return empty_fig("Keine Daten verfügbar")

    # violent crime categories
    violence_categories = [
        "Sexualstraftaten",
        "Einfache KV",
        "Schwere KV",
        "Gewaltkriminalität",
        "Raub & Erpressung",
        "Raub Geschäfte",
        "Raub auf Straßen",
        "Raub in Wohnungen",
    ]

    d2 = d[d["Straftat_kurz"].isin(violence_categories)]

    if d2.empty:
        return empty_fig("Keine Daten zur Gewalt gegen Frauen verfügbar")

    g = (
        d2.groupby("Jahr")["Opfer weiblich"]
        .sum()
        .reset_index()
        .sort_values("Jahr")
    )

    fig = px.line(
        g,
        x="Jahr",
        y="Opfer weiblich",
        markers=True,
        color_discrete_sequence=["#b91c1c"],  # dark red
        title="Gewalt gegen Frauen im Zeitverlauf (2019–2024)",
        labels={"Opfer weiblich": "Weibliche Opfer"},
    )

    fig.update_layout(height=STANDARD_HEIGHT)

    return fig


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
                "textAlign": "center",
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
                        "textAlign": "center",
                    },
                ),
                html.H4(
                    "Polizeiliche Kriminalstatistik Deutschland (2019–2024)",
                    style={
                        "fontSize": "18px",
                        "fontWeight": "450",
                        "color": HEADER_TEXT_SUB,
                        "marginTop": "0px",
                        "textAlign": "center",
                    },
                ),
                # Toggle button for sidebar (☰)
                html.Button(
                    "☰",
                    id="toggle-sidebar",
                    n_clicks=0,
                    title="Sidebar ein-/ausblenden",
                    style={
                        "position": "absolute",
                        "top": "30px",
                        "right": "30px",
                        "fontSize": "22px",
                        "background": "transparent",
                        "border": "none",
                        "color": "white",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),
        dcc.Location(id="url"),
        dcc.Store(id="sidebar-visible", data=True),
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
    if path == "/trends":                     # 🆕 NEW
        return layout_trends()
    if path == "/temporal":
        return layout_temporal()
    return html.Div([html.H2("404 – Seite nicht gefunden")])

# --------- SIDEBAR TOGGLE CALLBACKS ---------
# Toggle sidebar visibility store
@app.callback(
    Output("sidebar-visible", "data"),
    Input("toggle-sidebar", "n_clicks"),
    State("sidebar-visible", "data"),
)
def toggle_sidebar(n_clicks, visible):
    # Keep sidebar visible on initial load
    if not callback_context.triggered:
        return visible

    trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
    if trigger == "toggle-sidebar" and n_clicks and n_clicks > 0:
        return not visible

    return visible

# Dynamically update sidebar and content styles
@app.callback(
    Output("sidebar", "style"),
    Output("page-content", "style"),
    Input("sidebar-visible", "data"),
)
def update_sidebar_visibility(visible):
    if visible:
        return SIDEBAR_STYLE, CONTENT_STYLE
    else:
        hidden_sidebar = SIDEBAR_STYLE.copy()
        hidden_sidebar["display"] = "none"
        expanded_content = CONTENT_STYLE.copy()
        expanded_content["marginLeft"] = "0px"
        return hidden_sidebar, expanded_content


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
    Output("crime-pie", "figure"),
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
        fig_crime_pie(d),
    )


# --------- GEOGRAPHIC CALLBACKS ---------
@app.callback(
    Output("selected-state-store", "data"),
    Input("map", "clickData"),
    Input("back-to-germany", "n_clicks"),
    Input("filter-state", "value"),
    State("selected-state-store", "data"),
)
def update_selected_state(click_data, back_clicks, filter_states, current_state):
    """Handle state selection logic"""
    ctx = callback_context
    
    if not ctx.triggered:
        return current_state
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Reset if back button clicked or filter changed
    if trigger_id == 'back-to-germany' or trigger_id == 'filter-state':
        return None
    
    # Only process map clicks if we're at state level (not city level)
    if trigger_id == 'map' and click_data and current_state is None:
        # We're at state level, so process the click
        try:
            if click_data and 'points' in click_data and click_data['points']:
                point = click_data['points'][0]
                # Try to get state name from different possible locations
                if 'hovertext' in point:
                    return point['hovertext']
                elif 'location' in point:
                    return point['location']
                elif 'customdata' in point and point['customdata']:
                    return point['customdata'][0]
        except Exception as e:
            print(f"Error processing click: {e}")
    
    # If we're already in a state view (current_state is not None), 
    # clicking on the map should do nothing
    return current_state

@app.callback(
    Output("map", "figure"),
    Output("statebar", "figure"),
    Output("topregions", "figure"),
    Output("current-state-display", "children"),
    Output("state-back-button", "style"),
    Input("filter-year", "value"),
    Input("filter-crime", "value"),
    Input("filter-state", "value"),
    Input("selected-state-store", "data"),
    Input("geo-city-mode", "value"),
    Input("geo-age-group", "value"),
    Input("geo-safety-mode", "value"),
)
def update_geo_components(
    years, crimes, states, selected_state, city_mode, age_group, safety_mode
):
    d = filter_data(years or YEARS, crimes or [], states or [])

    map_fig = fig_geo_map(
        d,
        selected_state=selected_state,
        city_mode=city_mode,
        age_group=age_group,
        safety_mode=safety_mode,
    )

    state_bar_fig = fig_geo_state_bar(d)
    top_regions_fig = fig_geo_top(d)

    # Update info text
    if selected_state:
        text = f"Aktuelle Ansicht: {selected_state} – Städteansicht"
        back_style = {"display": "block"}
    else:
        text = (
            "Aktuelle Ansicht: Deutschland – Bundesländer"
            if city_mode == "bundesland"
            else "Aktuelle Ansicht: Deutschland – Städte"
        )
        back_style = {"display": "none"}

    return map_fig, state_bar_fig, top_regions_fig, text, back_style


# --------- CRIME TYPES CALLBACK ---------
@app.callback(
    Output("heat", "figure"),
    Output("stacked", "figure"),
    Output("agechart", "figure"),
    Output("top5-crime", "figure"),
    Output("donut-crime", "figure"),
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
    top5_fig = fig_top5(d)
    donut_fig = fig_donut(d)

    return heat_fig, stacked_fig, age_fig, top5_fig, donut_fig


# Trends Callback (city danger)
@app.callback(
    Output("city-danger", "figure"),
    Input("filter-year", "value"),
    Input("filter-crime", "value"),
    Input("filter-state", "value"),
    Input("city-count", "value"),
    Input("city-color-scale", "value"),
)
def update_city_danger(years, crimes, states, top_n, color_scale):
    d = filter_data(years or YEARS, crimes or [], states or [])
    return fig_city_danger(
        d,
        top_n=top_n or 10,
        color_scale=color_scale or "OrRd",
    )


# Trends Callback: Children 0–14 ranking (map + bar)
@app.callback(
    Output("trend-children-cities", "figure"),
    Output("trend-children-bar", "figure"),
    Input("filter-year", "value"),
    Input("filter-crime", "value"),
    Input("filter-state", "value"),
    Input("trend-children-topn", "value"),
    Input("trend-children-mode", "value"),
    Input("trend-age-group", "value"),
)
def update_trend_children_cities(years, crimes, states, top_n, mode, age_group):
    d = filter_data(years or YEARS, crimes or [], states or [])
    map_fig = fig_children_ranking(
        d,
        top_n=top_n or 10,
        mode=mode or "dangerous",
        age_group=age_group or "Kinder <14",
    )
    bar_fig = fig_children_bar(
        d,
        top_n=top_n or 10,
        mode=mode or "dangerous",
        age_group=age_group or "Kinder <14",
    )
    return map_fig, bar_fig


#viollence against Women callback
@app.callback(
    Output("trend-women-violence", "figure"),
    Input("filter-year", "value"),
    Input("filter-crime", "value"),
    Input("filter-state", "value"),
)
def update_trend_violence_women(years, crimes, states):
    d = filter_data(years or YEARS, crimes or [], states or [])
    return fig_violence_women(d)


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


