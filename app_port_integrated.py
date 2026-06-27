import os
import re
from glob import glob

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_folium import st_folium


st.set_page_config(
    page_title="MSR Port CCD and Pollution Observatory",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    :root {
        --ink: #17212b;
        --muted: #667085;
        --sea: #176b87;
        --green: #28765b;
        --amber: #c98220;
        --red: #b5473c;
        --line: #d8dee5;
        --soft: #f5f7f8;
    }
    .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
    .app-header {
        border-top: 5px solid var(--sea);
        border-bottom: 1px solid var(--line);
        padding: 1.1rem 0 1rem 0;
        margin-bottom: 1rem;
    }
    .app-header h1 {
        color: var(--ink);
        font-size: 2rem;
        margin: 0;
        letter-spacing: 0;
    }
    .app-header p {color: var(--muted); margin: .35rem 0 0 0;}
    .app-credit {
        color: var(--sea);
        font-size: .95rem;
        font-weight: 600;
        margin-top: .45rem;
    }
    .section-note {
        border-left: 4px solid var(--sea);
        background: var(--soft);
        color: #344054;
        padding: .75rem .9rem;
        margin: .5rem 0 1rem 0;
    }
    .method-box {
        border: 1px solid var(--line);
        padding: .85rem 1rem;
        background: white;
        border-radius: 6px;
    }
    .stTabs [data-baseweb="tab-list"] {gap: .4rem;}
    .stTabs [data-baseweb="tab"] {border-radius: 4px 4px 0 0;}
    div[data-testid="stMetric"] {
        border-top: 3px solid var(--sea);
        border-bottom: 1px solid var(--line);
        padding: .6rem .75rem;
        background: white;
    }
</style>
""",
    unsafe_allow_html=True,
)


APP_DIR = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-10

NATURAL_INDICATORS = {
    "elevation": "Elevation",
    "slope": "Slope",
    "terrain_ruggedness_tri": "Terrain ruggedness",
    "evi_vegetation": "EVI vegetation",
    "coastal_water_occur": "Coastal water occurrence",
    "lu_forest_ratio": "Forest ratio",
    "lu_grass_ratio": "Grassland ratio",
    "lu_cropland_ratio": "Cropland ratio",
    "lu_water_ratio": "Water ratio",
}

SOCIO_INDICATORS = {
    "population_density_2020": "Population density",
    "urban_ratio": "Urban ratio",
    "economic_activity_proxy": "Economic activity",
    "port_accessibility": "Accessibility travel time",
    "infra_network_density": "Infrastructure density",
}

# True means that a larger raw value contributes positively to the system score.
# Accessibility is travel time, so a smaller value means better accessibility.
POSITIVE_DIRECTION = {
    "elevation": True,
    "slope": False,
    "terrain_ruggedness_tri": False,
    "evi_vegetation": True,
    "coastal_water_occur": True,
    "lu_forest_ratio": True,
    "lu_grass_ratio": True,
    "lu_cropland_ratio": True,
    "lu_water_ratio": True,
    "population_density_2020": True,
    "urban_ratio": True,
    "economic_activity_proxy": True,
    "port_accessibility": False,
    "infra_network_density": True,
}

POLLUTANTS = ("SO2", "NO2", "CO")
POLLUTANT_COLORS = {"SO2": "#b5473c", "NO2": "#c98220", "CO": "#176b87"}
EXCLUDED_PORTS = {"Jiangyin", "Hankow", "Nanjing", "Chang Sha"}
LEVEL_ORDER = [
    "Coordinated",
    "Environment-lagging",
    "Economy-lagging",
    "Dual-lagging",
]


def normalize_port_name(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def safe_minmax(series, positive=True):
    values = pd.to_numeric(series, errors="coerce").astype(float)
    minimum = values.min()
    maximum = values.max()
    if pd.isna(minimum) or pd.isna(maximum) or np.isclose(maximum, minimum):
        normalized = pd.Series(0.5, index=values.index, dtype=float)
    else:
        normalized = (values - minimum) / (maximum - minimum)
    return normalized if positive else 1 - normalized


def entropy_weights(normalized_df):
    n = len(normalized_df)
    if n <= 1 or normalized_df.shape[1] == 0:
        return np.ones(normalized_df.shape[1]) / max(normalized_df.shape[1], 1)
    matrix = normalized_df.to_numpy(dtype=float)
    proportions = matrix / (matrix.sum(axis=0) + EPS)
    proportions = np.clip(proportions, EPS, 1 - EPS)
    entropy = -(proportions * np.log(proportions)).sum(axis=0) / np.log(n)
    divergence = np.clip(1 - entropy, 0, None)
    if np.isclose(divergence.sum(), 0):
        return np.ones(len(divergence)) / len(divergence)
    return divergence / divergence.sum()


def coordination_level(value):
    if value >= 0.8:
        return "Good coordination"
    if value >= 0.6:
        return "Primary coordination"
    if value >= 0.5:
        return "Barely coordinated"
    if value >= 0.4:
        return "Near disorder"
    if value >= 0.3:
        return "Mild disorder"
    if value >= 0.2:
        return "Moderate disorder"
    return "Severe disorder"


def development_type(row, threshold):
    geo = row["geo_score"]
    socio = row["socio_score"]
    if geo >= threshold and socio >= threshold:
        return "Coordinated"
    if geo < threshold <= socio:
        return "Environment-lagging"
    if socio < threshold <= geo:
        return "Economy-lagging"
    return "Dual-lagging"


def classify_three(series):
    valid = pd.to_numeric(series, errors="coerce")
    if valid.notna().sum() < 3 or valid.nunique(dropna=True) < 3:
        median = valid.median()
        return pd.Series(
            np.where(valid >= median, "High", "Low"),
            index=series.index,
        )
    ranks = valid.rank(method="first")
    return pd.qcut(ranks, q=3, labels=["Low", "Medium", "High"]).astype(str)


def classify_quadrant(row, concentration_cut, exposure_cut):
    concentration = "High pollution" if row["pollution_index"] >= concentration_cut else "Low pollution"
    exposure = "High exposure" if row["exposure_index"] >= exposure_cut else "Low exposure"
    return f"{concentration} / {exposure}"


def indicator_name_from_file(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"^MSR_Port_", "", stem, flags=re.IGNORECASE)
    return re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()


@st.cache_data(show_spinner=False)
def load_indicator_table(folder):
    frames = []
    quality = []
    expected = set(NATURAL_INDICATORS) | set(SOCIO_INDICATORS)

    for path in sorted(glob(os.path.join(folder, "MSR_Port_*.csv"))):
        indicator = indicator_name_from_file(path)
        if indicator not in expected:
            continue
        source = pd.read_csv(path)
        source.columns = source.columns.str.strip().str.lower()
        if "port_name" not in source or "mean" not in source:
            continue
        source["port_name"] = normalize_port_name(source["port_name"])
        source = source[~source["port_name"].isin(EXCLUDED_PORTS)].copy()
        source["mean"] = pd.to_numeric(source["mean"], errors="coerce")
        duplicate_rows = int(source.duplicated("port_name", keep=False).sum())
        frame = source.groupby("port_name", as_index=False)["mean"].mean()
        frame = frame.rename(columns={"mean": indicator})
        missing_rate = float(frame[indicator].isna().mean())
        quality.append(
            {
                "indicator": indicator,
                "label": NATURAL_INDICATORS.get(indicator, SOCIO_INDICATORS.get(indicator)),
                "source_file": os.path.basename(path),
                "ports": len(frame),
                "missing_rate": missing_rate,
                "duplicate_rows_aggregated": duplicate_rows,
                "status": "Excluded: all missing" if missing_rate == 1 else "Included",
            }
        )
        if missing_rate < 1:
            frames.append(frame)

    if not frames:
        return None, pd.DataFrame(quality)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="port_name", how="outer")

    coordinates_path = os.path.join(folder, "port_coordinates.csv")
    if os.path.exists(coordinates_path):
        coordinates = pd.read_csv(coordinates_path)
        coordinates.columns = coordinates.columns.str.strip().str.lower()
        if {"port_name", "latitude", "longitude"}.issubset(coordinates.columns):
            coordinates["port_name"] = normalize_port_name(coordinates["port_name"])
            coordinates = coordinates[
                ~coordinates["port_name"].isin(EXCLUDED_PORTS)
            ].copy()
            coordinates = coordinates.groupby("port_name", as_index=False).agg(
                latitude=("latitude", "mean"),
                longitude=("longitude", "mean"),
                location_count=("port_name", "size"),
            )
            merged = merged.merge(coordinates, on="port_name", how="left")

    return merged, pd.DataFrame(quality)


@st.cache_data(show_spinner=False)
def calculate_ccd(indicator_table, alpha):
    table = indicator_table.copy()
    all_cols = [c for c in NATURAL_INDICATORS if c in table.columns]
    all_cols += [c for c in SOCIO_INDICATORS if c in table.columns]
    for col in all_cols:
        table[col] = pd.to_numeric(table[col], errors="coerce")
        table[col] = table[col].fillna(table[col].median())

    geo_cols = [c for c in NATURAL_INDICATORS if c in table.columns]
    socio_cols = [c for c in SOCIO_INDICATORS if c in table.columns]
    geo_norm = pd.DataFrame(
        {col: safe_minmax(table[col], POSITIVE_DIRECTION[col]) for col in geo_cols}
    )
    socio_norm = pd.DataFrame(
        {col: safe_minmax(table[col], POSITIVE_DIRECTION[col]) for col in socio_cols}
    )

    geo_weights = entropy_weights(geo_norm)
    socio_weights = entropy_weights(socio_norm)
    geo_score = geo_norm.to_numpy().dot(geo_weights)
    socio_score = socio_norm.to_numpy().dot(socio_weights)
    coupling = 2 * np.sqrt(
        (geo_score * socio_score) / ((geo_score + socio_score) ** 2 + EPS)
    )
    comprehensive = alpha * geo_score + (1 - alpha) * socio_score
    coordination = np.sqrt(coupling * comprehensive)

    result = table[["port_name"]].copy()
    for col in ("latitude", "longitude", "location_count"):
        if col in table:
            result[col] = table[col]
    result["geo_score"] = geo_score
    result["socio_score"] = socio_score
    result["coupling_C"] = coupling
    result["comprehensive_T"] = comprehensive
    result["coordination_D"] = coordination
    result["coordination_level"] = result["coordination_D"].map(coordination_level)
    score_cut = float(pd.concat([result["geo_score"], result["socio_score"]]).median())
    result["ccd_type"] = result.apply(development_type, axis=1, threshold=score_cut)

    weight_table = pd.concat(
        [
            pd.DataFrame(
                {
                    "system": "Geo-environmental",
                    "indicator": geo_cols,
                    "indicator_label": [NATURAL_INDICATORS[c] for c in geo_cols],
                    "direction": ["Positive" if POSITIVE_DIRECTION[c] else "Negative" for c in geo_cols],
                    "weight": geo_weights,
                }
            ),
            pd.DataFrame(
                {
                    "system": "Socio-economic",
                    "indicator": socio_cols,
                    "indicator_label": [SOCIO_INDICATORS[c] for c in socio_cols],
                    "direction": ["Positive" if POSITIVE_DIRECTION[c] else "Negative" for c in socio_cols],
                    "weight": socio_weights,
                }
            ),
        ],
        ignore_index=True,
    )
    normalized = pd.concat(
        [
            table[["port_name"]].reset_index(drop=True),
            geo_norm.add_suffix("_normalized"),
            socio_norm.add_suffix("_normalized"),
        ],
        axis=1,
    )
    return result.sort_values("coordination_D", ascending=False).reset_index(drop=True), weight_table, normalized


@st.cache_data(show_spinner=False)
def load_pollution(folder):
    frames = []
    for pollutant in POLLUTANTS:
        path = os.path.join(
            folder,
            f"Ports_{pollutant}_population_weighted_exposure_2018_2025.csv",
        )
        if not os.path.exists(path):
            continue
        data = pd.read_csv(path)
        data.columns = data.columns.str.strip()
        required = {
            "port_name",
            "year",
            f"{pollutant}_area_mean_mol_per_m2",
            f"{pollutant}_population_weighted_exposure_mol_per_m2",
        }
        if not required.issubset(data.columns):
            continue
        data["port_name"] = normalize_port_name(data["port_name"])
        data = data[~data["port_name"].isin(EXCLUDED_PORTS)].copy()
        data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
        data["concentration"] = pd.to_numeric(
            data[f"{pollutant}_area_mean_mol_per_m2"], errors="coerce"
        )
        data["exposure"] = pd.to_numeric(
            data[f"{pollutant}_population_weighted_exposure_mol_per_m2"],
            errors="coerce",
        )
        data["population"] = pd.to_numeric(data.get("total_population"), errors="coerce")
        data["pollutant"] = pollutant
        frames.append(
            data[["port_name", "year", "pollutant", "concentration", "exposure", "population"]]
        )

    if not frames:
        return None
    pollution = pd.concat(frames, ignore_index=True)
    pollution = pollution.dropna(subset=["port_name", "year"])
    return (
        pollution.groupby(["port_name", "year", "pollutant"], as_index=False)
        .agg(
            concentration=("concentration", "mean"),
            exposure=("exposure", "mean"),
            population=("population", "mean"),
        )
    )


def pollution_snapshot(pollution, year_mode):
    if year_mode == "2018-2025 mean":
        snapshot = pollution.groupby(["port_name", "pollutant"], as_index=False).agg(
            concentration=("concentration", "mean"),
            exposure=("exposure", "mean"),
            population=("population", "mean"),
        )
    else:
        year = int(year_mode)
        snapshot = pollution[pollution["year"] == year].copy()

    wide = snapshot.pivot_table(
        index="port_name",
        columns="pollutant",
        values=["concentration", "exposure"],
        aggfunc="mean",
    )
    wide.columns = [f"{metric}_{pollutant}" for metric, pollutant in wide.columns]
    wide = wide.reset_index()

    concentration_norm = []
    exposure_norm = []
    for pollutant in POLLUTANTS:
        concentration_col = f"concentration_{pollutant}"
        exposure_col = f"exposure_{pollutant}"
        if concentration_col in wide:
            normalized = safe_minmax(wide[concentration_col])
            wide[f"{concentration_col}_normalized"] = normalized
            concentration_norm.append(f"{concentration_col}_normalized")
        if exposure_col in wide:
            normalized = safe_minmax(wide[exposure_col])
            wide[f"{exposure_col}_normalized"] = normalized
            exposure_norm.append(f"{exposure_col}_normalized")

    wide["pollution_index"] = wide[concentration_norm].mean(axis=1)
    wide["exposure_index"] = wide[exposure_norm].mean(axis=1)
    wide["pollution_level"] = classify_three(wide["pollution_index"])
    wide["exposure_level"] = classify_three(wide["exposure_index"])
    concentration_cut = wide["pollution_index"].median()
    exposure_cut = wide["exposure_index"].median()
    wide["exposure_quadrant"] = wide.apply(
        classify_quadrant,
        axis=1,
        concentration_cut=concentration_cut,
        exposure_cut=exposure_cut,
    )
    return wide, concentration_cut, exposure_cut


def merge_analysis(ccd, pollution_wide):
    merged = ccd.merge(pollution_wide, on="port_name", how="inner")
    merged["hidden_cost_type"] = (
        merged["ccd_type"]
        + " / "
        + merged["exposure_level"]
        + " exposure"
    )
    return merged


def metric_row(ccd, pollution):
    columns = st.columns(5)
    columns[0].metric("Ports in CCD", f"{len(ccd):,}")
    columns[1].metric("Mean coordination D", f"{ccd['coordination_D'].mean():.3f}")
    columns[2].metric("Pollution ports", f"{pollution['port_name'].nunique():,}")
    columns[3].metric("Years", f"{pollution['year'].min()}–{pollution['year'].max()}")
    columns[4].metric("Pollutants", pollution["pollutant"].nunique())


def add_map_legend(fmap, title, items):
    rows = "".join(
        f'<div><span style="color:{color};font-size:18px;">●</span> {label}</div>'
        for label, color in items
    )
    html = f"""
    <div style="position:fixed;bottom:25px;right:25px;z-index:9999;
        background:white;border:1px solid #cfd6dc;padding:10px 13px;
        font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);">
        <b>{title}</b>{rows}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(html))


def render_point_map(data, value_col, label_col, color_map, popup_fields, radius_col=None):
    map_data = data.dropna(subset=["latitude", "longitude"]).copy()
    if map_data.empty:
        st.warning("No matching coordinates are available for this map.")
        return
    fmap = folium.Map(
        location=[map_data["latitude"].mean(), map_data["longitude"].mean()],
        zoom_start=3,
        tiles="CartoDB positron",
        control_scale=True,
    )
    for _, row in map_data.iterrows():
        label = row[label_col]
        color = color_map.get(label, "#667085")
        radius_value = row[radius_col] if radius_col else row[value_col]
        radius = 5 + 10 * float(np.clip(radius_value, 0, 1))
        lines = [f"<b>{row['port_name']}</b>"]
        for field, display in popup_fields:
            value = row.get(field)
            if isinstance(value, (float, np.floating)):
                lines.append(f"{display}: {value:.4f}")
            else:
                lines.append(f"{display}: {value}")
        folium.CircleMarker(
            [row["latitude"], row["longitude"]],
            radius=radius,
            color=color,
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=.72,
            tooltip=f"{row['port_name']}: {row[value_col]:.3f}",
            popup=folium.Popup("<br>".join(lines), max_width=330),
        ).add_to(fmap)
    add_map_legend(fmap, label_col.replace("_", " ").title(), list(color_map.items()))
    st_folium(fmap, height=590, use_container_width=True, returned_objects=[])


def render_ccd_module(ccd, weights, normalized):
    st.markdown(
        '<div class="section-note">Port-level CCD measures the balance between the geo-environmental and socio-economic subsystems. Pollution is evaluated separately to avoid mixing development capacity with environmental cost.</div>',
        unsafe_allow_html=True,
    )
    subtab1, subtab2, subtab3, subtab4 = st.tabs(
        ["Map and ranking", "System balance", "Indicator weights", "Port profile"]
    )

    level_colors = {
        "Good coordination": "#28765b",
        "Primary coordination": "#4d8d72",
        "Barely coordinated": "#c98220",
        "Near disorder": "#d29a4b",
        "Mild disorder": "#b95d50",
        "Moderate disorder": "#9d3e37",
        "Severe disorder": "#6f2b28",
    }

    with subtab1:
        render_point_map(
            ccd,
            "coordination_D",
            "coordination_level",
            level_colors,
            [
                ("coordination_D", "Coordination D"),
                ("geo_score", "Environmental score"),
                ("socio_score", "Socio-economic score"),
                ("ccd_type", "Development type"),
            ],
        )
        left, right = st.columns([3, 2])
        with left:
            ranking = ccd[
                [
                    "port_name",
                    "coordination_D",
                    "coordination_level",
                    "ccd_type",
                    "geo_score",
                    "socio_score",
                ]
            ].copy()
            ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
            st.dataframe(
                ranking.round(4),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "coordination_D": st.column_config.ProgressColumn(
                        "Coordination D", min_value=0, max_value=1, format="%.3f"
                    )
                },
            )
        with right:
            counts = ccd["coordination_level"].value_counts().reset_index()
            counts.columns = ["level", "ports"]
            fig = px.bar(
                counts,
                x="ports",
                y="level",
                orientation="h",
                color="level",
                color_discrete_map=level_colors,
                title="Coordination level distribution",
            )
            fig.update_layout(showlegend=False, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

    with subtab2:
        type_colors = {
            "Coordinated": "#28765b",
            "Environment-lagging": "#176b87",
            "Economy-lagging": "#c98220",
            "Dual-lagging": "#b5473c",
        }
        fig = px.scatter(
            ccd,
            x="geo_score",
            y="socio_score",
            size="coordination_D",
            color="ccd_type",
            hover_name="port_name",
            color_discrete_map=type_colors,
            labels={
                "geo_score": "Geo-environmental score F",
                "socio_score": "Socio-economic score G",
                "ccd_type": "CCD type",
            },
            title="Environmental and socio-economic balance",
        )
        threshold = pd.concat([ccd["geo_score"], ccd["socio_score"]]).median()
        fig.add_vline(x=threshold, line_dash="dash", line_color="#667085")
        fig.add_hline(y=threshold, line_dash="dash", line_color="#667085")
        fig.update_layout(height=570)
        st.plotly_chart(fig, use_container_width=True)

    with subtab3:
        fig = px.bar(
            weights.sort_values("weight"),
            x="weight",
            y="indicator_label",
            color="system",
            orientation="h",
            facet_col="system",
            facet_col_wrap=2,
            text_auto=".3f",
            title="Entropy weights by subsystem",
            color_discrete_map={
                "Geo-environmental": "#28765b",
                "Socio-economic": "#176b87",
            },
        )
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(height=520, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(weights.round(5), use_container_width=True, hide_index=True)

    with subtab4:
        selected = st.selectbox("Port", ccd["port_name"].tolist(), key="ccd_port")
        port = ccd[ccd["port_name"] == selected].iloc[0]
        row = normalized[normalized["port_name"] == selected]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Coordination D", f"{port['coordination_D']:.3f}")
        c2.metric("Geo score", f"{port['geo_score']:.3f}")
        c3.metric("Socio score", f"{port['socio_score']:.3f}")
        c4.metric("CCD type", port["ccd_type"])
        if not row.empty:
            values = row.drop(columns="port_name").iloc[0]
            radar = pd.DataFrame(
                {
                    "indicator": [
                        NATURAL_INDICATORS.get(
                            name.replace("_normalized", ""),
                            SOCIO_INDICATORS.get(name.replace("_normalized", ""), name),
                        )
                        for name in values.index
                    ],
                    "value": values.values,
                }
            )
            fig = go.Figure(
                go.Scatterpolar(
                    r=radar["value"],
                    theta=radar["indicator"],
                    fill="toself",
                    line_color="#176b87",
                    fillcolor="rgba(23,107,135,.22)",
                )
            )
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
                height=570,
                title=f"{selected}: normalized indicator profile",
            )
            st.plotly_chart(fig, use_container_width=True)


def render_pollution_module(pollution, pollution_wide, ccd):
    st.markdown(
        '<div class="section-note">The pollution module reports SO2, NO2 and CO concentration and population-weighted exposure. Select ports and years to generate presentation-ready charts directly from the exported GEE CSV files.</div>',
        unsafe_allow_html=True,
    )
    subtab1, subtab2, subtab3, subtab4 = st.tabs(
        ["Time series", "Pollution map", "Port comparison", "GEE figure gallery"]
    )

    with subtab1:
        controls = st.columns([2, 1, 1])
        ports = sorted(pollution["port_name"].unique())
        preferred = [p for p in ["Gwadar", "Shanghai", "Hong Kong", "Malacca", "Colombo"] if p in ports]
        selected_ports = controls[0].multiselect(
            "Ports", ports, default=preferred, max_selections=8
        )
        pollutant = controls[1].selectbox("Pollutant", POLLUTANTS)
        metric = controls[2].radio(
            "Metric", ["Concentration", "Population-weighted exposure"]
        )
        value_col = "concentration" if metric == "Concentration" else "exposure"
        chart_data = pollution[
            pollution["port_name"].isin(selected_ports)
            & (pollution["pollutant"] == pollutant)
        ]
        fig = px.line(
            chart_data,
            x="year",
            y=value_col,
            color="port_name",
            markers=True,
            labels={
                value_col: f"{metric} (mol/m²)",
                "year": "Year",
                "port_name": "Port",
            },
            title=f"{pollutant} {metric.lower()}, 2018–2025",
        )
        fig.update_layout(height=580, hovermode="x unified")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)

    with subtab2:
        map_metric = st.radio(
            "Map variable",
            ["Pollution index", "Exposure index"],
            horizontal=True,
            key="pollution_map_metric",
        )
        map_data = ccd.merge(pollution_wide, on="port_name", how="inner")
        if map_metric == "Pollution index":
            value_col = "pollution_index"
            label_col = "pollution_level"
        else:
            value_col = "exposure_index"
            label_col = "exposure_level"
        colors = {"Low": "#28765b", "Medium": "#c98220", "High": "#b5473c"}
        render_point_map(
            map_data,
            value_col,
            label_col,
            colors,
            [
                ("pollution_index", "Multi-pollutant index"),
                ("exposure_index", "Exposure index"),
                ("pollution_level", "Pollution level"),
                ("exposure_level", "Exposure level"),
            ],
        )

    with subtab3:
        metric_choice = st.selectbox(
            "Comparison metric",
            [
                "Pollution index",
                "Exposure index",
                "SO2 concentration",
                "NO2 concentration",
                "CO concentration",
                "SO2 exposure",
                "NO2 exposure",
                "CO exposure",
            ],
        )
        metric_map = {
            "Pollution index": "pollution_index",
            "Exposure index": "exposure_index",
            "SO2 concentration": "concentration_SO2",
            "NO2 concentration": "concentration_NO2",
            "CO concentration": "concentration_CO",
            "SO2 exposure": "exposure_SO2",
            "NO2 exposure": "exposure_NO2",
            "CO exposure": "exposure_CO",
        }
        value_col = metric_map[metric_choice]
        top_n = st.slider("Ports displayed", 10, min(50, len(pollution_wide)), 25)
        bars = pollution_wide.nlargest(top_n, value_col).sort_values(value_col)
        fig = px.bar(
            bars,
            x=value_col,
            y="port_name",
            orientation="h",
            color=value_col,
            color_continuous_scale=["#d8e8df", "#c98220", "#b5473c"],
            labels={value_col: metric_choice, "port_name": "Port"},
            title=f"Top {top_n} ports: {metric_choice.lower()}",
        )
        fig.update_layout(height=max(500, top_n * 24), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with subtab4:
        figure_root = os.path.join(APP_DIR, "GEE_Figures")
        available = {
            pollutant: sorted(glob(os.path.join(figure_root, pollutant, "*.png")))
            for pollutant in POLLUTANTS
        }
        gallery_pollutant = st.segmented_control(
            "GEE output", POLLUTANTS, default="SO2"
        )
        files = available.get(gallery_pollutant, [])
        sample_files = [f for f in files if os.path.basename(f).lower() != "full map.png"]
        if not sample_files:
            st.info("No GEE PNG samples were found for this pollutant.")
        else:
            chosen = st.selectbox(
                "Sample port",
                sample_files,
                format_func=lambda p: os.path.splitext(os.path.basename(p))[0],
            )
            st.image(chosen, use_container_width=True)
        if gallery_pollutant == "SO2":
            full_map = os.path.join(figure_root, "SO2", "full map.png")
            if os.path.exists(full_map):
                with st.expander("Open the original GEE full-port map"):
                    st.image(full_map, use_container_width=True)


def render_hidden_cost_module(combined, concentration_cut, exposure_cut):
    st.markdown(
        '<div class="section-note">This module identifies hidden costs by comparing development coordination with pollution and human exposure. Thresholds are relative to the observed MSR port sample and should be interpreted as comparative, not regulatory, categories.</div>',
        unsafe_allow_html=True,
    )
    subtab1, subtab2, subtab3 = st.tabs(
        ["Exposure quadrants", "CCD × exposure", "Priority ports"]
    )

    quadrant_colors = {
        "High pollution / High exposure": "#8f2f2a",
        "High pollution / Low exposure": "#c98220",
        "Low pollution / High exposure": "#176b87",
        "Low pollution / Low exposure": "#28765b",
    }

    with subtab1:
        fig = px.scatter(
            combined,
            x="pollution_index",
            y="exposure_index",
            color="exposure_quadrant",
            size="coordination_D",
            hover_name="port_name",
            color_discrete_map=quadrant_colors,
            labels={
                "pollution_index": "Multi-pollutant concentration index",
                "exposure_index": "Population-weighted exposure index",
                "exposure_quadrant": "Hidden-cost quadrant",
            },
            title="Pollution concentration and population exposure",
        )
        fig.add_vline(x=concentration_cut, line_dash="dash", line_color="#667085")
        fig.add_hline(y=exposure_cut, line_dash="dash", line_color="#667085")
        fig.update_layout(height=590)
        st.plotly_chart(fig, use_container_width=True)
        render_point_map(
            combined,
            "exposure_index",
            "exposure_quadrant",
            quadrant_colors,
            [
                ("exposure_quadrant", "Hidden-cost quadrant"),
                ("pollution_index", "Pollution index"),
                ("exposure_index", "Exposure index"),
                ("coordination_D", "Coordination D"),
            ],
            radius_col="exposure_index",
        )

    with subtab2:
        cross = pd.crosstab(
            combined["ccd_type"],
            combined["exposure_level"],
        ).reindex(index=LEVEL_ORDER, columns=["Low", "Medium", "High"], fill_value=0)
        fig = px.imshow(
            cross,
            text_auto=True,
            color_continuous_scale=["#eef3f5", "#c98220", "#8f2f2a"],
            labels={"x": "Exposure level", "y": "CCD type", "color": "Ports"},
            title="CCD development type × exposure level",
            aspect="auto",
        )
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

        sunburst = px.sunburst(
            combined,
            path=["ccd_type", "exposure_level"],
            values=np.ones(len(combined)),
            color="exposure_index",
            color_continuous_scale=["#28765b", "#c98220", "#8f2f2a"],
            title="Composition of development and exposure types",
        )
        sunburst.update_layout(height=570)
        st.plotly_chart(sunburst, use_container_width=True)

    with subtab3:
        priority = combined.copy()
        priority["priority_score"] = (
            0.4 * priority["pollution_index"]
            + 0.4 * priority["exposure_index"]
            + 0.2 * (1 - priority["coordination_D"])
        )
        priority = priority.sort_values("priority_score", ascending=False)
        columns = [
            "port_name",
            "priority_score",
            "coordination_D",
            "ccd_type",
            "pollution_index",
            "pollution_level",
            "exposure_index",
            "exposure_level",
            "exposure_quadrant",
        ]
        st.dataframe(
            priority[columns].round(4),
            use_container_width=True,
            hide_index=True,
            column_config={
                "priority_score": st.column_config.ProgressColumn(
                    "Priority score", min_value=0, max_value=1, format="%.3f"
                )
            },
        )
        st.caption(
            "Exploratory priority score = 40% pollution index + 40% exposure index + 20% coordination deficit."
        )


def render_data_quality(quality, ccd, pollution, combined):
    with st.expander("Data quality and methodology", expanded=False):
        st.markdown(
            """
<div class="method-box">
<b>CCD formula</b><br>
Each indicator is direction-adjusted and min-max normalized. Entropy weights are calculated separately for the environmental and socio-economic systems. The model then calculates
<code>C = 2√(FG)/(F+G)</code>, <code>T = αF + (1-α)G</code>, and <code>D = √(C×T)</code>.
<br><br>
<b>Pollution indices</b><br>
SO2, NO2 and CO are normalized separately before averaging. This prevents CO's larger numerical scale from dominating the combined index. Population-weighted exposure remains distinct from area-mean concentration.
</div>
""",
            unsafe_allow_html=True,
        )
        st.dataframe(quality, use_container_width=True, hide_index=True)
        st.write(
            f"CCD ports: {len(ccd)}. Pollution ports: {pollution['port_name'].nunique()}. "
            f"Matched ports used in integrated analysis: {len(combined)}."
        )
        excluded = quality[quality["status"] != "Included"]
        if not excluded.empty:
            st.warning(
                "Indicators excluded because all values were missing: "
                + ", ".join(excluded["label"].tolist())
            )


def main():
    st.markdown(
        """
<div class="app-header">
    <h1>Maritime Silk Road Port Coordination and Hidden Pollution Costs</h1>
    <p>Port-level CCD, multi-pollutant trends, population exposure, and integrated risk classification</p>
    <div class="app-credit">Created by ZHAO Dingitan | Student ID: 3035945807</div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.caption("MSR-CCD port-level model by ZHAO Dingitan | Student ID: 3035945807")
        st.header("Analysis controls")
        data_folder = st.text_input("Data folder", value=APP_DIR)
        alpha = st.slider(
            "Environmental contribution α",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Socio-economic contribution is 1 − α.",
        )
        year_options = ["2018-2025 mean"] + [str(year) for year in range(2018, 2026)]
        year_mode = st.selectbox("Pollution snapshot", year_options, index=0)
        st.divider()
        st.caption(
            "The app reads MSR_Port_*.csv and Ports_[SO2/NO2/CO]_population_weighted_exposure_2018_2025.csv automatically."
        )
        refresh = st.button("Reload CSV data", use_container_width=True)
        if refresh:
            st.cache_data.clear()

    indicator_table, quality = load_indicator_table(data_folder)
    pollution = load_pollution(data_folder)
    if indicator_table is None:
        st.error("No usable MSR_Port indicator CSV files were found.")
        st.stop()
    if pollution is None:
        st.error("SO2, NO2 and CO port-year CSV files were not found or could not be read.")
        st.stop()

    ccd, weights, normalized = calculate_ccd(indicator_table, alpha)
    pollution_wide, concentration_cut, exposure_cut = pollution_snapshot(
        pollution, year_mode
    )
    combined = merge_analysis(ccd, pollution_wide)

    metric_row(ccd, pollution)
    st.caption(
        f"Pollution snapshot: {year_mode}. Integrated analysis uses {len(combined)} matched ports."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "1. Port-level CCD",
            "2. Port pollution and exposure",
            "3. Hidden environmental and health costs",
        ]
    )
    with tab1:
        render_ccd_module(ccd, weights, normalized)
    with tab2:
        render_pollution_module(pollution, pollution_wide, ccd)
    with tab3:
        render_hidden_cost_module(combined, concentration_cut, exposure_cut)

    st.divider()
    render_data_quality(quality, ccd, pollution, combined)

    download1, download2, download3 = st.columns(3)
    download1.download_button(
        "Download CCD results",
        ccd.to_csv(index=False, encoding="utf-8-sig"),
        "MSR_port_CCD_app_results.csv",
        "text/csv",
        use_container_width=True,
    )
    download2.download_button(
        "Download pollution snapshot",
        pollution_wide.to_csv(index=False, encoding="utf-8-sig"),
        "MSR_port_pollution_snapshot.csv",
        "text/csv",
        use_container_width=True,
    )
    download3.download_button(
        "Download integrated classification",
        combined.to_csv(index=False, encoding="utf-8-sig"),
        "MSR_port_hidden_cost_classification.csv",
        "text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
