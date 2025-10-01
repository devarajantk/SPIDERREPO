import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib import colors
import matplotlib as mpl
import io

# 👉 Modern font globally for matplotlib
mpl.rcParams['font.family'] = 'Inter'       # can also pick 'Roboto', 'Segoe UI', 'Helvetica Neue'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titleweight'] = 'semibold'

# 🪄 CricViz-style light mode UI with modern font
st.set_page_config(page_title="Wagon Wheel Views", layout="centered")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    .stApp {
        background-color: #f5f5f5;
        color: #000000;
        font-family: 'Inter', sans-serif;
    }
    div[data-testid="stSidebar"] {
        background-color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding: 2rem;
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        font-family: 'Inter', sans-serif;
    }
    .stDataFrame { color: #000000; font-family: 'Inter', sans-serif; }
    .stMetric { font-family: 'Inter', sans-serif; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏏 Wagon Wheel – Batter vs Bowler View")

# 🎨 Colour Palettes (unchanged)
color_schemes = {
    "🔴 Reds vs 🔵 Blues": ("Reds_r", "Blues_r"),
    "🟣 Purples vs 🟠 Oranges": ("Purples_r", "Oranges_r"),
    "🟢 Greens vs 🟠 Oranges": ("Greens_r", "Oranges_r"),
    "🟢 Greens vs 🟣 Purples": ("Greens_r", "Purples_r"),
    "🌈 Viridis vs Plasma": ("viridis", "plasma"),
    "💜 Inferno vs Magma": ("inferno", "magma"),
    "🟡 Cividis vs Turbo": ("cividis", "turbo"),
    "🧊 Cool vs 🔥 Hot": ("cool", "hot"),
    "🌪️ Coolwarm vs seismic": ("coolwarm", "seismic"),
    "🧬 Set1 vs Dark2": ("Set1", "Dark2"),
    "🎭 Paired vs Accent": ("Paired", "Accent"),
    "🍬 Pastel1 vs Set2": ("Pastel1", "Set2"),
    "🎨 Tab10 vs Tab20": ("tab10", "tab20"),
    "🌸 Spring vs Autumn": ("spring", "autumn"),
    "🍁 BrBG vs PiYG": ("BrBG", "PiYG"),
    "🌍 Spectral vs RdYlBu": ("Spectral", "RdYlBu"),
    "🥇 Copper vs 🌊 Ocean": ("copper", "ocean"),
    "🟡 Wistia vs ❄️ Winter": ("Wistia", "winter"),
    "💎 Blues vs YlOrBr": ("Blues_r", "YlOrBr_r"),
    "💖 Pink vs YlGnBu": ("pink", "YlGnBu"),
}

# 🔧 Contrast text
def get_contrast_text(color):
    r, g, b, _ = color
    r_, g_, b_ = int(r*255), int(g*255), int(b*255)
    yiq = ((r_*299) + (g_*587) + (b_*114)) / 1000
    return "black" if yiq >= 128 else "white"

# ⚙️ Sidebar Controls
st.sidebar.header("⚙️ Options")
palette_choice = st.sidebar.selectbox("🎨 Colour Palette", list(color_schemes.keys()))
leg_palette_name, off_palette_name = color_schemes[palette_choice]

alpha_val = st.sidebar.slider("Wedge Transparency", 0.4, 1.0, 0.9, 0.05)
show_labels = st.sidebar.checkbox("Show Labels", value=True)
min_label_val = st.sidebar.slider("Min Threshold for Labels", 0, 20, 5, 1)
show_grid = st.sidebar.checkbox("Show Gridlines", value=True)
grid_alpha = st.sidebar.slider("Grid Visibility", 0.0, 1.0, 0.3)

# 📂 Upload CSV
uploaded_file = st.file_uploader("Upload ball-by-ball CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = {"batsmanName", "wagonZone", "batsmanRuns", "isFour", "isSix", "ballNumber", "shotControl"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        batsmen = df['batsmanName'].dropna().unique()
        batsman = st.selectbox("Select Batsman", sorted(batsmen))
        df_bat = df[df['batsmanName'] == batsman].copy()
        

        if df_bat.empty:
            st.warning(f"No valid wagon data for {batsman}.")
        else:
            stance = df_bat['batsmanBattingStyle'].iloc[0] if "batsmanBattingStyle" in df_bat else "Unknown"
            is_left = "Left" in str(stance)
            st.write(f"Batting stance: **{stance}**")

            # ===== METRICS =====
            zone_runs = df_bat.groupby("wagonZone")["batsmanRuns"].sum()
            total_runs = zone_runs.sum()
            zone_pct_runs = (zone_runs / total_runs * 100).replace([np.inf, np.nan], 0)

            zone_stats = (
    			df_bat.groupby("wagonZone")
    			.agg(
        		runs=('batsmanRuns', 'sum'),
        		balls=('ballNumber', 'count'),
        		fours=('isFour', 'sum'),
        		sixes=('isSix', 'sum'),
        		controlled=('shotControl', lambda x: (x == 1).sum()),
        		wide_controlled=('shotControl', lambda x: ((x == 1) & (df_bat.loc[x.index, "wides"] == 1)).sum()),
    	    )
             .reindex(range(1, 9), fill_value=0)
    	     .reset_index()
	    )
            zone_stats["sr"] = (zone_stats["runs"] / zone_stats["balls"] * 100).replace([np.inf, np.nan], 0)
            zone_stats["boundary_balls"] = zone_stats["fours"] + zone_stats["sixes"]
            zone_stats["boundary %"] = (zone_stats["boundary_balls"] / zone_stats["balls"] * 100).replace([np.inf, np.nan], 0)
            zone_stats["adjusted_control"] = zone_stats["controlled"] - zone_stats["wide_controlled"]
            zone_stats["control %"] = (zone_stats["adjusted_control"] / zone_stats["balls"] * 100).replace([np.inf, np.nan], 0)

            labels_rh = ["Fine Leg", "Behind Sq Leg", "In front Sq Leg", "Mid Wicket",
                         "Straight/Mid Off", "Cover", "Backward Point", "Third Man"]
            labels = labels_rh if not is_left else labels_rh[::-1]
            zone_name_map = dict(zip(range(1, 9), labels))

            valid_zones = range(1, 9)
            metric_options = {"% of Runs": zone_pct_runs.reindex(valid_zones, fill_value=0),"Strike Rate": zone_stats.set_index("wagonZone")["sr"].reindex(valid_zones, fill_value=0),"Boundary %": zone_stats.set_index("wagonZone")["boundary %"].reindex(valid_zones, fill_value=0),"Control %": zone_stats.set_index("wagonZone")["control %"].reindex(valid_zones, fill_value=0)}
            plot_choice = st.sidebar.selectbox("📊 Select Metric for Wagon Wheel", list(metric_options.keys()))
            zone_vals = metric_options[plot_choice]

            # ===== WAGON WHEEL PLOT =====
            max_val = zone_vals.max() if not zone_vals.empty else 1
            wedge_zoom = 0.8 if max_val >= 60 else 1.0 if max_val >= 40 else 1.2 if max_val >= 20 else 1.5

            view = st.radio("Choose View:", ["Batter’s End", "Bowler’s End"])
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6), dpi=120)
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#f5f5f5")
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location("S" if view == "Batter’s End" else "N")

            wedge_angle = 2 * np.pi / 8
            leg_cmap, off_cmap = cm.get_cmap(leg_palette_name), cm.get_cmap(off_palette_name)
            norm = colors.Normalize(vmin=0.3, vmax=1.0)

            for z in range(1, 9):
                val = zone_vals.get(z, 0.0)
                theta1, theta2 = (z - 1) * wedge_angle, z * wedge_angle
                color = leg_cmap(norm(val / max_val)) if z <= 4 else off_cmap(norm(val / max_val))
                height = (val / max_val) * wedge_zoom if max_val > 0 else 0
                ax.bar((theta1 + theta2) / 2, height, width=wedge_angle,
                       color=color, alpha=alpha_val, edgecolor="#dddddd", linewidth=1.0)

            # Zone labels modern style
            for i, zone_name in enumerate(labels):
                ang = (i + 0.5) * wedge_angle
                zone_val = zone_vals.get(i + 1, 0.0)
                if plot_choice != "Strike Rate":
                    label_text = f"{zone_name}\n({zone_val:.0f}%)"
                else:
                    label_text = f"{zone_name}\n({zone_val:.0f})"

                ax.text(
                    ang, 1.18 * wedge_zoom, label_text,
                    ha="center", va="center",
                    fontsize=9, fontweight="medium", color="#2f3e46"
                )

            if show_grid:
                for frac in [0.25, 0.5, 0.75, 1.0]:
                    ax.plot(np.linspace(0, 2 * np.pi, 200), [frac * wedge_zoom] * 200,
                            color="gray", linestyle="--", linewidth=0.8, alpha=grid_alpha)

            outer_circle = plt.Circle((0, 0), 1.0 * wedge_zoom,
                                      transform=ax.transData._b, fill=False,
                                      color="#bbbbbb", linewidth=1.2, alpha=0.7)
            ax.add_artist(outer_circle)

            ax.spines['polar'].set_visible(False)
            ax.set_ylim(0, 1.1 * wedge_zoom)
            ax.set_xticks([]), ax.set_yticks([])

            plt.title(f"{batsman} – {plot_choice} Across Wagon",
                      fontsize=15, weight="semibold", pad=25, color="#1a237e")
            st.pyplot(fig)

            # PNG Download
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
            buf.seek(0)
            st.download_button(
                label="📥 Download Wagon Wheel as PNG",
                data=buf,
                file_name=f"{batsman}_wagon_wheel.png",
                mime="image/png"
            )

            # ===== TABLES =====
            st.subheader("📊 Zone Contributions (% of Runs)")
            zone_df = zone_pct_runs.reindex(valid_zones, fill_value=0).reset_index()
            zone_df.columns = ["wagonZone", "% of Runs"]
            zone_df["Zone"] = zone_df["wagonZone"].map(zone_name_map)
            st.dataframe(zone_df[["Zone", "% of Runs"]].round(1))

            st.subheader("📊 Extended Zone-wise Batting Stats")
            zone_stats["Zone"] = zone_stats["wagonZone"].map(zone_name_map)
            st.dataframe(zone_stats[[
                "Zone", "runs", "balls", "sr", "boundary %", "control %"
            ]].rename(columns={
                "runs": "Runs",
                "balls": "Balls",
                "sr": "Strike Rate",
                "boundary %": "Boundary %",
                "control %": "Control %"
            }).round(1))

            # ===== SUMMARY =====
            overall_sr = (df_bat["batsmanRuns"].sum() / df_bat.shape[0] * 100) if df_bat.shape[0] > 0 else 0
            overall_boundary_pct = ((df_bat["isFour"].sum() + df_bat["isSix"].sum()) / df_bat.shape[0] * 100) if df_bat.shape[0] > 0 else 0

# Use the original df (uploaded file) for overall control %, ignore wagonZone filter
            df_full = df[df['batsmanName'] == batsman].copy()
            df_full["wides"] = df_full["wides"].fillna(0)
            df_full["noballs"] = df_full["noballs"].fillna(0)
            df_full["shotControl"] = df_full["shotControl"].fillna(0)

# Count only valid balls (excluding wides)
            df_full["shotControl"] = pd.to_numeric(df_full["shotControl"], errors='coerce').fillna(0).astype(int)
            df_full["wides"] = pd.to_numeric(df_full["wides"], errors='coerce').fillna(0).astype(int)
            df_full["noballs"] = pd.to_numeric(df_full["noballs"], errors='coerce').fillna(0).astype(int)
            
            controlled_balls = (df_full["shotControl"] == 1).sum()

# Subtract cases where it's a wide AND shotControl==1
            wide_controlled = ((df_full["shotControl"] == 1) & (df_full["wides"] == 1)).sum()

# Adjusted controlled balls
            adjusted_controlled = controlled_balls - wide_controlled

# Total valid balls (still counting wides)
            total_valid_balls = len(df_full)

# Control percentage
            overall_control_pct = adjusted_controlled / total_valid_balls * 100

            st.subheader("📌 Overall Batting Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("💥 Overall Boundary %", f"{overall_boundary_pct:.1f}%")
            col2.metric("⚡ Overall SR", f"{overall_sr:.1f}")
            col3.metric("🎯 Overall Control %", f"{overall_control_pct:.1f}%")
