'''
Created on 18 Apr 2025

@author: yang hu
'''

import re, io
import os

import json

from matplotlib.ticker import FuncFormatter

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative

import numpy as np


# 1) Read the CSV --------------------------------------------------------
csv_path = "f_model-source.csv" # change if the file is elsewhere
# csv_path = "f_model-source.csv"
with open(csv_path, "r", encoding="utf-8") as f:
    text = f.read()

# Remove commas inside numbers (1,300,000 -> 1300000)
text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)

# 2) Load into DataFrame --------------------------------------------------
df = pd.read_csv(io.StringIO(text))

# 2) Clean numeric columns (remove commas, convert to float) -------------
for col in ["slides", "tiles"]:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
                   .str.replace(",", "", regex=False)
                   .replace({"": None})
                   .astype(float)
        )

# 3) Formatter that shows numbers as K / M / B ---------------------------
def sci_formatter(x, pos=None):
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:.1f}B"
    elif x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    elif x >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:g}"

abbr_fmt = FuncFormatter(sci_formatter)

# 4) Plot US state choropleth map ----------------------------------------
def plot_us_state_map(df: pd.DataFrame):
    """Plot US states colored by Foundation Models individually (no merging)."""
    # 1) explode multi‑state cells: "CA;WA" → two rows
    states_df = df.dropna(subset=["state"]).copy()
    states_df["state"] = states_df["state"].str.split(r";\s*")
    states_df = states_df.explode("state")
    states_df["state"] = states_df["state"].str.strip()

    # 2) build choropleth with each model as its own category
    fig = px.choropleth(
        states_df,
        locations="state",                # USPS codes
        locationmode="USA-states",
        color="model_name",               # use model_name directly
        scope="usa",
        # color_discrete_sequence=px.colors.qualitative.Set2,
        color_discrete_sequence=[
            "#FFA500",  # (orange)
            "#FFD700",  # (gold)
            "#32CD32",  # (lime-green)
        ],
        category_orders={"model_name": sorted(states_df["model_name"].unique())},
        title="Location of Foundation Model Training Data Source"
    )

    # 3) layout tweaks
    fig.update_layout(
        title=dict(
            text="<b>Location of Foundation Model Training Data Source</b>",
            x=0.5,
            xanchor="center",
            yanchor="top",
            pad=dict(t=10),
            font=dict(size=50, family="Arial Black, Arial, sans-serif")
        ),
        font=dict(
            family="Arial Black, Arial, sans-serif",
            size=13
        ),
        # legend=dict(
        #     title_font=dict(size=30),
        #     font=dict(size=30),
        #     yanchor="top",
        #     y=0.98,
        #     xanchor="left",
        #     x=0.02,
        #     bgcolor="rgba(255,255,255,0.6)",
        #     borderwidth=0
        # ),
        margin=dict(l=5, r=5, t=40, b=5)
    )

    # 4) export & show
    fig.write_html("foundation_states_map.html", include_plotlyjs="cdn")
    fig.show()
    
def plot_world_highlight_map():
    """Highlight China and USA on a square world map in mediumseagreen."""
    import plotly.graph_objects as go

    # Create a choropleth trace for USA and China
    fig = go.Figure(go.Choropleth(
        locations=['USA', 'CHN'],       # ISO-3 codes
        locationmode='ISO-3',
        z=[1, 1],                        # dummy values
        colorscale=[[0, 'mediumseagreen'], [1, 'mediumseagreen']],
        showscale=False,                 # no colorbar
        marker_line_color='black',
        marker_line_width=0.5
    ))

    # Layout: square aspect, no title, no legend, plain map
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='natural earth'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        width=800,
        height=600
    )

    fig.show()

# --- global font settings (size 20, normal weight) ----------------------
plt.rcParams.update({
    "font.size": 20,
    "font.weight": "normal",         # no bold
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

# 5) Plot bar chart for slides (sorted by slides descending) ------------
def plot_slides_bar(df):
    """Bar chart of slides sorted by value (highest to lowest)."""
    slides_df = df.dropna(subset=["slides"]).sort_values(by="slides", ascending=True)
    fig, ax = plt.subplots(figsize=(6, 8))
    bars = ax.bar(slides_df["model_name"], slides_df["slides"], color="tomato")

    # add value labels
    for bar in bars:
        height = bar.get_height()
        label = sci_formatter(height)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=16
        )

    # adjust y-limit to leave space for labels
    max_val = slides_df["slides"].max()
    ax.set_ylim(0, max_val * 1.15)
    fig.subplots_adjust(top=0.95)

    ax.set_title("Training Slides")
    ax.set_ylabel("")
    ax.yaxis.set_major_formatter(abbr_fmt)
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.show()


# 6) Plot bar chart for tiles (sorted by tiles descending) --------------
def plot_tiles_bar(df):
    """Bar chart of tiles sorted by value (highest to lowest)."""
    tiles_df = df.dropna(subset=["tiles"]).sort_values(by="tiles", ascending=True)
    fig, ax = plt.subplots(figsize=(5, 8))
    bars = ax.bar(tiles_df["model_name"], tiles_df["tiles"], color="cyan")

    # add value labels
    for bar in bars:
        height = bar.get_height()
        label = sci_formatter(height)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=16
        )

    # adjust y-limit to leave space for labels
    max_val = tiles_df["tiles"].max()
    ax.set_ylim(0, max_val * 1.15)
    fig.subplots_adjust(top=0.95)

    ax.set_title("Training Tiles")
    ax.set_ylabel("")
    ax.yaxis.set_major_formatter(abbr_fmt)
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.show()
    
# 8) Plot hollow radial bar chart ----------------------------------------
def plot_concentric_circular_bars(data):
    """
    Concentric circular bar chart with numeric labels at bar heads:
    - Inner empty center ring.
    - One ring per model, starting after the empty ring.
    - Bars start at center_radius + j * ring_width.
    - Height = ring_width * (percent / (max_percent + 1)).
    - Numeric labels represent category index, placed at each bar's head for the outermost ring only.
    - Only title in main plot; legend and mapping textbox drawn separately.
    """
    # 1) Prepare categories and models
    categories = sorted({cat for dist in data.values() for cat in dist.keys()})
    models = list(data.keys())
    N = len(categories)
    M = len(models)

    # 2) Set radial parameters
    center_radius = 1.5               # leave inner ring empty
    outer_radius = 2.5                # outer boundary radius
    ring_width = (outer_radius - center_radius) / M
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # 3) Create figure and configure polar plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='polar')
    ax.set_theta_zero_location('N')  # zero at top
    ax.set_theta_direction(-1)       # clockwise
    ax.set_facecolor('white')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)

    # 4) Plot each model's ring
    # colors = plt.cm.Dark2.colors
    colors = ['#FFD700', '#FFA500', '#32CD32', '#FF0000']
    for j, model in enumerate(models):
        bottom = center_radius + j * ring_width
        percents = [float(data[model].get(cat, '0%').strip('%')) for cat in categories]
        max_p = max(percents)
        scale = max_p + 1.0
        heights = [(p / scale) * ring_width for p in percents]
        bars = ax.bar(
            angles,
            heights,
            width=2 * np.pi / N * 0.8,
            bottom=bottom,
            color=colors[j % len(colors)],
            edgecolor='k',
            align='edge'
        )
        # annotate only outermost ring at bar heads
        if j == M - 1:
            for i, bar in enumerate(bars):
                angle = bar.get_x() + bar.get_width() / 2
                r_head = bar.get_y() + bar.get_height()
                ax.text(
                    angle,
                    r_head,
                    str(i + 1),
                    ha='center',   # horizontal center
                    va='bottom',   # just above the bar
                    fontsize=10
                )

    # 5) Only title in main plot, closer to plot
    # plt.title('Training Data Cancer Types', fontsize=20, y=1.03)

    # 6) Draw legend separately on figure
    fig.legend(models, loc='upper right', bbox_to_anchor=(0.32, 0.95), fontsize=20)

    # 7) Mapping textbox under legend
    mapping = '\n'.join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])
    fig.text(
        0.8, 0.95,
        mapping,
        ha='left',
        va='top',
        fontsize=14,
        transform=fig.transFigure
    )

    plt.tight_layout()
    plt.show()





# 7) Execute all three plots ---------------------------------------------
plot_us_state_map(df)
# plot_world_highlight_map()
# plot_slides_bar(df)
# plot_tiles_bar(df)

# with open("f_model-type_dist.json", "r") as f:
#     data = json.load(f)
# plot_concentric_circular_bars(data)