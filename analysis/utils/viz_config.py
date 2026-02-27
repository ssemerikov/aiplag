"""Shared visualization configuration for all analysis scripts."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Output directories
import os
FIGURE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'source_new', 'figures')
ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), '..')
os.makedirs(FIGURE_DIR, exist_ok=True)

# Figure dimensions
FIG_SINGLE = (7, 5)
FIG_WIDE = (10, 5)
FIG_TALL = (7, 8)
FIG_SQUARE = (7, 7)
FIG_DOUBLE = (14, 5)

# DPI
DPI = 300

# Font configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'text.usetex': False,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palettes
YEAR_COLORS = {2022: '#66c2a5', 2023: '#fc8d62', 2024: '#8da0cb', 2025: '#e78ac3'}
TOPIC_CMAP = plt.cm.Set2
NODE_TYPE_COLORS = {
    'Paper': '#4e79a7',
    'Author': '#f28e2b',
    'Institution': '#e15759',
    'Country': '#76b7b2',
    'Topic': '#59a14f',
    'Keyword': '#edc948',
    'Technology': '#b07aa1',
    'Method': '#ff9da7',
    'Venue': '#9c755f',
    'Year': '#bab0ac',
}
CLUSTER_COLORS = list(plt.cm.tab10.colors)

def get_topic_color(idx, n_topics=8):
    return TOPIC_CMAP(idx / max(n_topics - 1, 1))

def savefig(fig, name, tight=True):
    path = os.path.join(FIGURE_DIR, name)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
