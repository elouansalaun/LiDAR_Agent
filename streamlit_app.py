"""Web interface.
"""

from __future__ import annotations

import json
import os

import geopandas as gpd
import streamlit as st

from src import config
from src.lidar import io
from src.places import PLACES, get_place
from src.visualization.maps import point_cloud_figure

st.set_page_config(page_title="LiDAR AI Agent", page_icon="🛰️", layout="wide")

CLAUDE, OFFLINE = "Claude (LLM)", "Offline demo (rule-based)"

EXAMPLE_QUESTIONS = [
    "Find buildings taller than 20 m around Saint Joseph's Oratory",
    "And above 50 m?",
    "Which buildings are between 20 and 40 m high near Université de Montréal?",
    "Which area does the data cover?",
]


# --- Cached loaders ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_buildings(path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(config.PROJECT_ROOT / path)


@st.cache_data(show_spinner=False)
def load_point_cloud_figure(path: str, color_by: str):
    cloud = io.read_point_cloud(config.PROJECT_ROOT / path)
    return point_cloud_figure(cloud, color_by=color_by)


# --- Results panel ------------------------------------------------------------

def show_results(artifacts: dict) -> None:
    """Statistics, map, table and 3D view for the latest results."""
    buildings_path = artifacts.get("buildings_path")
    if not buildings_path:
        return
    buildings = load_buildings(buildings_path)

    st.subheader("Results")
    cols = st.columns(4)
    if len(buildings):
        cols[0].metric("Buildings", len(buildings))
        cols[1].metric("Average height", f"{buildings['height_m'].mean():.1f} m")
        cols[2].metric("Maximum height", f"{buildings['height_m'].max():.1f} m")
        cols[3].metric("Total footprint", f"{buildings['area_m2'].sum():,.0f} m²")
    else:
        cols[0].metric("Buildings", 0)

    map_tab, table_tab, cloud_tab = st.tabs(["Map", "Table", "3D point cloud"])
    with map_tab:
        map_file = (config.PROJECT_ROOT / artifacts.get("map_path", "")).resolve()
        # Only embed maps our create_map tool wrote (the iframe runs the HTML's JavaScript).
        if map_file.suffix == ".html" and map_file.parent == config.OUTPUTS_DIR and map_file.exists():
            st.iframe(map_file, height=560)
            st.caption(f"Saved to `{artifacts['map_path']}` · switch to the satellite layer with the control at the top right.")
        else:
            st.info("No map was created for these results.")
    with table_tab:
        columns = ["building_id", "height_m", "max_height_m", "area_m2", "point_count", "lat", "lon"]
        st.dataframe(buildings[columns] if len(buildings) else buildings, hide_index=True, width="stretch")
        st.download_button("Download GeoJSON", (config.PROJECT_ROOT / buildings_path).read_bytes(),
                           file_name=os.path.basename(buildings_path), mime="application/geo+json")
    with cloud_tab:
        cloud_path = artifacts.get("point_cloud_path")
        if cloud_path and cloud_path.endswith("_classified.laz"):
            color_by = st.radio("Colour by", ["class", "height"], horizontal=True)
            with st.spinner("Loading point cloud..."):
                st.plotly_chart(load_point_cloud_figure(cloud_path, color_by), width="stretch")
            st.caption("Random sample of 150,000 points. Height is the height above ground.")
        else:
            st.info("The 3D view needs a classified point cloud.")


def describe_event(event: dict) -> str | None:
    if event["type"] == "assistant_text":
        return f"💬 {event['text']}"
    if event["type"] == "tool_call":
        status = "❌" if event["is_error"] else "✅"
        arguments = ", ".join(f"{k}={v!r}" for k, v in event["input"].items() if "path" not in k)
        return f"{status} `{event['name']}`({arguments}) — {event['seconds']} s"
    if event["type"] == "validation":
        return ("⚠️ Validator rejected a result" if not event["valid"] else "ℹ️ Validator warning") + f"\n\n```\n{event['feedback']}\n```"
    return None


# --- Modes ----------------------------------------------------------------------

def agent_mode(planner: str) -> None:
    from src.agent.graph import LidarAgent
    from src.agent.llm import ClaudeLLM, has_credentials
    from src.agent.offline import OfflinePlanner

    if planner == CLAUDE and not has_credentials():
        st.warning("Set `ANTHROPIC_API_KEY` in a `.env` file to use Claude (see `.env.example`), "
                   f"or choose **{OFFLINE}** in the sidebar.")
        return
    if planner == OFFLINE:
        st.info("**Offline demo planner.** A rule-based stand-in replaces Claude (no API key needed). It understands "
                "building-height and coverage questions only, but the tools, validator, map and 3D view below are "
                "the real pipeline.", icon="🧩")

    # One conversation per planner; switching planner starts a new one.
    if st.session_state.get("planner") != planner:
        st.session_state.planner = planner
        st.session_state.agent = LidarAgent(llm=OfflinePlanner() if planner == OFFLINE else ClaudeLLM())
        st.session_state.chat = []

    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.caption("Try an example:")
    columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, example in zip(columns, EXAMPLE_QUESTIONS):
        if column.button(example, width="stretch"):
            st.session_state.pending_question = example

    question = st.chat_input("e.g. Find buildings taller than 20 m around Université de Montréal")
    question = question or st.session_state.pop("pending_question", None)
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.status("Working...", expanded=True) as status:
                def on_event(event: dict) -> None:
                    line = describe_event(event)
                    if line:
                        status.write(line)

                result = st.session_state.agent.ask(question, on_event=on_event)
                status.update(label=f"Done in {result.steps} steps", state="complete", expanded=False)
            st.markdown(result.answer)
        st.session_state.chat.append({"role": "assistant", "content": result.answer})

    show_results(st.session_state.agent.artifacts)


def pipeline_mode(place: str, size_m: int, min_height: float) -> None:
    from src.tools import ToolError
    from src.workflow import run_building_query

    lon, lat = get_place(place).lon, get_place(place).lat
    st.write(f"Fixed workflow on a **{size_m} m × {size_m} m** square around **{place}**, "
             f"keeping buildings **≥ {min_height:g} m**. No LLM involved.")

    if st.button("Run analysis", type="primary"):
        with st.status("Running pipeline...", expanded=True) as status:
            def on_step(step: dict) -> None:
                status.write(f"✅ `{step['name']}` — {step['seconds']} s" + (f"  ⚠️ {step['issues']}" if step["issues"] else ""))
            try:
                result = run_building_query({"center_lon": lon, "center_lat": lat, "size_m": size_m}, min_height=min_height, on_step=on_step)
            except ToolError as error:
                status.update(label="Failed", state="error")
                st.error(str(error))
                return
            status.update(label="Done", state="complete", expanded=False)
        st.session_state.pipeline_artifacts = {
            "buildings_path": result["selected_buildings"]["buildings_path"],
            "map_path": result["map_path"],
            "point_cloud_path": result["classified_point_cloud_path"],
        }
        st.session_state.class_percentages = result["class_percentages"]

    if "pipeline_artifacts" in st.session_state:
        st.caption("Point classes: " + " · ".join(f"{k} {v:.1f}%" for k, v in st.session_state.class_percentages.items()))
        show_results(st.session_state.pipeline_artifacts)


# --- Page -------------------------------------------------------------------------

st.title("🛰️ LiDAR AI Agent")
st.caption("Natural-language questions about 4 km² of Montreal LiDAR (Côte-des-Neiges) → "
           "Random Forest point classification → building heights → map.")

with st.sidebar:
    mode = st.radio("Mode", ["AI agent", "Pipeline"], help="Pipeline runs the same tools in a fixed order, with sliders.")
    if mode == "AI agent":
        from src.agent.llm import has_credentials

        planner = st.radio("Planner", [CLAUDE, OFFLINE], index=0 if has_credentials() else 1,
                           help="Who decides which tools to call. The offline planner is a rule-based stand-in "
                                "for demos without an API key.")
        if st.button("New conversation"):
            st.session_state.pop("planner", None)
    else:
        place = st.selectbox("Area", [p.name for p in PLACES])
        size_m = st.slider("Area size (m)", 200, 1000, 500, step=100)
        min_height = st.slider("Minimum height (m)", 0, 60, 20)

    st.divider()
    st.markdown("**Data**")
    tiles = io.find_tiles()
    st.markdown(f"{len(tiles)} tiles · {sum(t.point_count for t in tiles) / 1e6:.0f} M points · EPSG:2950")
    if config.MODEL_METRICS_PATH.exists():
        metrics = json.loads(config.MODEL_METRICS_PATH.read_text())
        best = metrics[metrics["selected"]]
        st.markdown(f"**Classifier** Random Forest · building F1 {best['per_class']['building']['f1']:.2f} · mIoU {best['mean_iou']:.2f}")

if mode == "AI agent":
    agent_mode(planner)
else:
    pipeline_mode(place, size_m, min_height)
