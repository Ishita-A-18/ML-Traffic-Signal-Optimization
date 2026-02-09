import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os

st.set_page_config(
    page_title="Smart Traffic Signal Dashboard",
    layout="wide"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🚦 Traffic Control Dashboard")

controller = st.sidebar.radio(
    "Select Controller",
    ["RL Controller", "Fixed-Time Controller"]
)

auto_refresh = st.sidebar.checkbox("Auto refresh", value=False)

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_data_from_pickle(pickle_path):
    """Load log data from pickle and convert to DataFrame"""
    try:
        with open(pickle_path, "rb") as f:
            log = pickle.load(f)
        return pd.DataFrame(log)
    except FileNotFoundError:
        return None

# Try to load from pickle
if controller == "RL Controller":
    pickle_file = "logs/run_log.pkl"
    controller_name = "RL (Hybrid Vision+SUMO)"
else:
    pickle_file = "logs/fixed_control_log.pkl"
    controller_name = "Fixed-Time (30s baseline)"

if os.path.exists(pickle_file):
    df = load_data_from_pickle(pickle_file)
    if df is None or df.empty:
        st.error(f"No data in log file. Run `python control/{'hybrid_control.py' if controller == 'RL Controller' else 'fixed_control.py'}` first.")
        st.stop()
else:
    st.error(f"📊 Log file not found. Run `python control/{'hybrid_control.py' if controller == 'RL Controller' else 'fixed_control.py'}` first.")
    st.stop()

# Add decision counter if not present
if "decision" not in df.columns:
    df["decision"] = range(1, len(df) + 1)

latest = df.iloc[-1]

# Display controller info
if controller == "RL Controller":
    st.info("📹 **Hybrid Sensor Setup**: Video (North Lane) + SUMO (East, South, West)")
else:
    st.info("🚦 **Fixed-Time Baseline**: 30s green per phase (standard timing)")

st.title(f"Controller: {controller_name}")

# ================================
# COMPARISON SECTION
# ================================
if os.path.exists("logs/run_log.pkl") and os.path.exists("logs/fixed_control_log.pkl"):
    st.divider()
    st.subheader("📊 RL vs Fixed-Time Comparison")
    
    rl_df = load_data_from_pickle("logs/run_log.pkl")
    fixed_df = load_data_from_pickle("logs/fixed_control_log.pkl")
    
    if rl_df is not None and fixed_df is not None:
        col1, col2, col3 = st.columns(3)
        
        rl_avg_queue = rl_df["queue"].mean()
        fixed_avg_queue = fixed_df["queue"].mean()
        queue_improvement = ((fixed_avg_queue - rl_avg_queue) / fixed_avg_queue * 100) if fixed_avg_queue > 0 else 0
        
        rl_avg_reward = rl_df["reward"].mean()
        fixed_avg_reward = fixed_df["reward"].mean()
        reward_improvement = ((rl_avg_reward - fixed_avg_reward) / abs(fixed_avg_reward) * 100) if fixed_avg_reward != 0 else 0
        
        rl_total_wait = rl_df["queue"].sum()
        fixed_total_wait = fixed_df["queue"].sum()
        wait_reduction = ((fixed_total_wait - rl_total_wait) / fixed_total_wait * 100) if fixed_total_wait > 0 else 0
        
        col1.metric("Avg Queue (RL)", f"{rl_avg_queue:.2f}", f"-{queue_improvement:.1f}% vs Fixed")
        col2.metric("Avg Reward (RL)", f"{rl_avg_reward:.2f}", f"+{reward_improvement:.1f}% vs Fixed")
        col3.metric("Total Wait Reduction", f"{wait_reduction:.1f}%", "RL is better")
        
        # Side-by-side queue comparison
        st.subheader("Queue Length Comparison")
        comparison_df = pd.DataFrame({
            "Decision": range(1, min(len(rl_df), len(fixed_df)) + 1),
            "RL Queue": rl_df["queue"][:min(len(rl_df), len(fixed_df))].values,
            "Fixed Queue": fixed_df["queue"][:min(len(rl_df), len(fixed_df))].values
        })
        
        fig_compare_queue = px.line(
            comparison_df,
            x="Decision",
            y=["RL Queue", "Fixed Queue"],
            title="Queue Length: RL vs Fixed-Time",
            labels={"value": "Queue Length", "variable": "Controller"},
            color_discrete_map={"RL Queue": "green", "Fixed Queue": "red"}
        )
        st.plotly_chart(fig_compare_queue, use_container_width='stretch')
        
        # Reward comparison
        st.subheader("Reward Comparison")
        reward_comparison_df = pd.DataFrame({
            "Decision": range(1, min(len(rl_df), len(fixed_df)) + 1),
            "RL Reward": rl_df["reward"][:min(len(rl_df), len(fixed_df))].values,
            "Fixed Reward": fixed_df["reward"][:min(len(rl_df), len(fixed_df))].values
        })
        
        fig_compare_reward = px.line(
            reward_comparison_df,
            x="Decision",
            y=["RL Reward", "Fixed Reward"],
            title="Reward: RL vs Fixed-Time",
            labels={"value": "Reward", "variable": "Controller"},
            color_discrete_map={"RL Reward": "green", "Fixed Reward": "red"}
        )
        st.plotly_chart(fig_compare_reward, use_container_width='stretch')

# -------------------------------
# Top metrics
# -------------------------------
st.divider()
st.subheader("📈 Current Controller Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("🚥 Phase", int(latest["phase"]))
col2.metric("⏱ Green Time (s)", int(latest["green"]))
col3.metric("🚗 Queue Length", int(latest["queue"]))
if "video_count" in df.columns:
    col4.metric("📹 Video Count", int(latest["video_count"]))
else:
    col4.metric("🎯 Reward", round(latest["reward"], 2))

st.divider()

# -------------------------------
# Signal visualization
# -------------------------------
st.subheader("🚦 Signal State")

phase = int(latest["phase"])

phase_map = {
    0: "NS Green",
    1: "NS Yellow",
    2: "EW Green",
    3: "EW Yellow"
}

color_map = {
    "NS Green": "green",
    "EW Green": "green",
    "NS Yellow": "orange",
    "EW Yellow": "orange"
}

signal_label = phase_map.get(phase, "Unknown")
signal_color = color_map.get(signal_label, "red")

st.markdown(
    f"""
    <div style="font-size:30px; font-weight:bold; color:{signal_color};">
        {signal_label}
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------------
# Queue over time
# -------------------------------
st.subheader("📊 Queue Length Over Time")

fig_queue = px.line(
    df,
    x="step",
    y="queue",
    title="Total Queue Length",
    labels={"queue": "Vehicles", "step": "Simulation Step"}
)

st.plotly_chart(fig_queue, use_container_width='stretch')

# -------------------------------
# Green duration decisions
# -------------------------------
st.subheader("⏱ Green Duration Decisions")

fig_green = px.line(
    df,
    x="decision",
    y="green",
    title="Green Time per Decision",
    labels={"green": "Seconds", "decision": "Decision #"},
    markers=True
)

st.plotly_chart(fig_green, use_container_width='stretch')

# -------------------------------
# Reward plot
# -------------------------------
st.subheader("🎯 Reward Trend")

fig_reward = px.line(
    df,
    x="decision",
    y="reward",
    title="Reward per Decision",
    labels={"reward": "Reward", "decision": "Decision #"}
)

st.plotly_chart(fig_reward, use_container_width='stretch')

# -------------------------------
# Video vehicle count (hybrid sensor)
# -------------------------------
if "video_count" in df.columns:
    st.subheader("📹 Video Lane Vehicle Count")
    
    fig_video = px.line(
        df,
        x="decision",
        y="video_count",
        title="Vehicles Detected in Video (North Lane)",
        labels={"video_count": "Vehicles", "decision": "Decision #"},
        markers=True
    )
    
    st.plotly_chart(fig_video, use_container_width='stretch')

# ================================
# PHASE TRANSITION ANALYSIS
# ================================
st.divider()
st.subheader("🚦 Phase Transitions & Timeline")

# Phase over time (line chart)
fig_phase = px.line(
    df,
    x="decision",
    y="phase",
    title="Traffic Light Phase Over Time",
    labels={"phase": "Phase Number", "decision": "Decision #"},
    markers=True
)
fig_phase.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=["NS Green", "NS Yellow", "EW Green", "EW Yellow"])
st.plotly_chart(fig_phase, use_container_width='stretch')

# Phase distribution (bar chart)
phase_counts = df["phase"].value_counts().sort_index()
phase_labels = {0: "NS Green", 1: "NS Yellow", 2: "EW Green", 3: "EW Yellow"}
phase_counts.index = phase_counts.index.map(lambda x: phase_labels.get(x, f"Phase {x}"))

fig_phase_dist = px.bar(
    x=phase_counts.index,
    y=phase_counts.values,
    title="Traffic Light Phase Distribution",
    labels={"x": "Phase", "y": "Count"},
    color=phase_counts.index,
    color_discrete_map={
        "NS Green": "green",
        "NS Yellow": "orange",
        "EW Green": "green",
        "EW Yellow": "orange"
    }
)
st.plotly_chart(fig_phase_dist, use_container_width='stretch')

# Phase + Queue correlation
col1, col2 = st.columns(2)

with col1:
    st.subheader("Queue by Phase")
    queue_by_phase = df.groupby("phase")["queue"].agg(["mean", "max"])
    queue_by_phase.index = queue_by_phase.index.map(lambda x: phase_labels.get(x, f"Phase {x}"))
    st.dataframe(queue_by_phase, use_container_width=True)

with col2:
    st.subheader("Green Time by Phase")
    green_by_phase = df.groupby("phase")["green"].agg(["mean", "min", "max"])
    green_by_phase.index = green_by_phase.index.map(lambda x: phase_labels.get(x, f"Phase {x}"))
    st.dataframe(green_by_phase, use_container_width=True)

# Phase transition table
st.subheader("Phase Transition Log (Last 20)")
phase_log = df[["decision", "phase", "green", "queue", "reward"]].tail(20).copy()
phase_log["phase"] = phase_log["phase"].map(lambda x: phase_labels.get(x, f"Phase {x}"))
st.dataframe(phase_log, use_container_width=True)

# -------------------------------
if auto_refresh:
    st.experimental_rerun()
