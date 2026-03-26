import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
import dash_bootstrap_components as dbc

from layouts import create_layout
from callbacks import register_callbacks

# ── Initialize App ─────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Energy Grid Analyzer",
    suppress_callback_exceptions=True,
)

server = app.server 

# ── Layout ─────────────────────────────────────────────────
app.layout = create_layout()

# ── Callbacks ──────────────────────────────────────────────
register_callbacks(app)

# ── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Energy Grid Analyzer Dashboard")
    print("  Open: http://localhost:8050")
    print("=" * 50 + "\n")
    app.run(debug=True, port=8050)
 