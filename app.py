import json
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
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

    # ===== GENERAL =====
    "Straftaten insgesamt": "Alle Straftaten",

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


# --------- GEOGRAPHIC FIGURES ---------
def prepare_state_geo_data(d):
    """Prepare state-level geographic data"""
    if d.empty or gdf_states is None:
        return None, None
    
    # Filter data
    victims_df = d[d["Straftat_kurz"] != "Straftaten insgesamt"]
    
    # Aggregate by state
    victims = (
        victims_df.groupby("Bundesland")["Oper insgesamt"]
        .sum()
        .reset_index()
        .rename(columns={"Oper insgesamt": "Opfer_insgesamt"})
    )
    
    # Merge with geographic data
    gdf_merged = gdf_states.merge(victims, on="Bundesland", how="left")
    gdf_merged["Opfer_insgesamt"] = gdf_merged["Opfer_insgesamt"].fillna(0)
    
    # Create GeoJSON
    geojson_data = json.loads(gdf_merged.to_json())
    
    return gdf_merged, geojson_data


def prepare_city_geo_data(d, selected_state):
    """Prepare city-level geographic data for selected state"""
    if d.empty or gdf_cities is None or not selected_state:
        return None, None, None
    
    # Filter data for selected state
    state_data = d[d["Bundesland"] == selected_state]
    if state_data.empty:
        return None, None, None
    
    # Filter geographic data for selected state
    gdf_state_cities = gdf_cities[gdf_cities["Bundesland"] == selected_state]
    
    if gdf_state_cities.empty:
        return None, None, None
    
    # Aggregate victim data by region
    city_victims = (
        state_data.groupby("Region")["Oper insgesamt"]
        .sum()
        .reset_index()
        .rename(columns={"Oper insgesamt": "Opfer_insgesamt"})
    )
    
    # Try to match city names
    gdf_merged = gdf_state_cities.copy()
    
    # Simple matching: check if any part of the city name matches
    def find_matching_city(region_name):
        region_lower = str(region_name).lower()
        for city in gdf_state_cities["City"].unique():
            city_lower = str(city).lower()
            if region_lower in city_lower or city_lower in region_lower:
                return city
        return None
    
    # Create mapping
    city_mapping = {}
    for _, row in city_victims.iterrows():
        region = row["Region"]
        matching_city = find_matching_city(region)
        if matching_city:
            city_mapping[matching_city] = row["Opfer_insgesamt"]
        else:
            # Store with original region name
            city_mapping[region] = row["Opfer_insgesamt"]
    
    # Map values to GeoDataFrame
    gdf_merged["Opfer_insgesamt"] = gdf_merged["City"].map(city_mapping).fillna(0)
    
    # Create GeoJSON
    geojson_data = json.loads(gdf_merged.to_json())
    
    # Calculate center - FIXED: Use proper projection for centroid calculation
    # First project to a meter-based CRS (UTM zone 32N for Germany)
    try:
        gdf_projected = gdf_merged.to_crs("EPSG:32632")  # UTM zone 32N
        centroid = gdf_projected.geometry.centroid
        # Convert back to WGS84 (lat/lon)
        centroid_wgs84 = centroid.to_crs("EPSG:4326")
        center_lat = centroid_wgs84.y.mean()
        center_lon = centroid_wgs84.x.mean()
    except Exception as e:
        print(f"Warning: Could not calculate proper centroid for {selected_state}, using simple mean: {e}")
        # Fallback: use simple mean of coordinates
        center_lat = gdf_merged.geometry.centroid.y.mean()
        center_lon = gdf_merged.geometry.centroid.x.mean()
    
    return gdf_merged, geojson_data, (center_lat, center_lon)


def fig_geo_map(d, selected_state=None):
    """Create interactive map using the new choropleth_map function"""
    if d.empty or gdf_states is None:
        return empty_fig("Keine Geodaten verfügbar")
    
    if selected_state:
        # City-level view for selected state
        gdf_data, geojson_data, center_coords = prepare_city_geo_data(d, selected_state)
        
        if gdf_data is None or geojson_data is None:
            return empty_fig(f"Keine Geodaten für {selected_state}")
        
        center_lat, center_lon = center_coords
        
        # Determine zoom level based on state size
        zoom_levels = {
            "Berlin": 10, "Bremen": 10, "Hamburg": 10, "Saarland": 9,
        }
        zoom = zoom_levels.get(selected_state, 7)
        
        # Use the new choropleth_map function
        fig = px.choropleth_map(
            gdf_data,
            geojson=geojson_data,
            locations=gdf_data.index,
            color="Opfer_insgesamt",
            hover_name="City",
            hover_data={"Opfer_insgesamt": True, "Bundesland": False},
            opacity=0.7,
            map_style="open-street-map",
            color_continuous_scale="Reds",
            zoom=zoom,
            center={"lat": center_lat, "lon": center_lon},
            title=f"Opfer in {selected_state} - Städte/Landkreise"
        )
        
        # Disable click events on cities - only Bundesland clicks are allowed
        fig.update_layout(clickmode='none')  # Disable all click interactions
        
    else:
        # State-level view for all Germany
        gdf_data, geojson_data = prepare_state_geo_data(d)
        
        if gdf_data is None or geojson_data is None:
            return empty_fig()
        
        # Use the new choropleth_map function
        fig = px.choropleth_map(
            gdf_data,
            geojson=geojson_data,
            locations=gdf_data.index,
            color="Opfer_insgesamt",
            hover_name="Bundesland",
            hover_data={"Opfer_insgesamt": True},
            opacity=0.7,
            map_style="open-street-map",
            color_continuous_scale="Reds",
            zoom=4.5,
            center={"lat": 51.0, "lon": 10.2},
            title="Opfer nach Bundesland ",
        )
        
        # Enable click events only at state level
        fig.update_layout(clickmode='event+select')
    
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
        d.groupby(["Region", "Bundesland"])["Oper insgesamt"]
        .sum()
        .reset_index()
        .nlargest(10, "Oper insgesamt")
        .sort_values("Oper insgesamt")
    )
    g["Label"] = g["Region"] + " (" + g["Bundesland"] + ")"
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
                "Vergleich der Opferzahlen nach Bundesland und Region. Klicken Sie auf ein Bundesland, um die Städte/Landkreise anzuzeigen.",
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
                    html.Div(id="current-state-display", children="Aktuelle Ansicht: Deutschland"),
                    html.Div(id="state-back-button", style={"display": "none"}, children=[
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
                    ])
                ]
            ),
            dcc.Store(id="selected-state-store", data=None),
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
)
def update_geo_components(years, crimes, states, selected_state):
    """Update all geographic components"""
    d = filter_data(years or YEARS, crimes or [], states or [])
    
    # Create figures - only 3 charts now
    map_fig = fig_geo_map(d, selected_state)
    state_bar_fig = fig_geo_state_bar(d)
    top_regions_fig = fig_geo_top(d)
    
    # Update display text and back button visibility
    if selected_state:
        display_text = f"Aktuelle Ansicht: {selected_state} (Städte/Landkreise) - Kein weiterer Klick möglich"
        back_button_style = {"display": "block"}
    else:
        display_text = "Aktuelle Ansicht: Deutschland (übersicht) - Klicken Sie auf ein Bundesland"
        back_button_style = {"display": "none"}
    
    return map_fig, state_bar_fig, top_regions_fig, display_text, back_button_style


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