"""CSS styles for the Fengshen GraphRAG Streamlit app — academic-scholarly theme."""

FENGSHEN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #f8fafb;
    --surface: #ffffff;
    --surface-alt: #f4f6f9;
    --border: #e2e8f0;
    --border-strong: #cbd5e1;
    --text: #1e293b;
    --text-muted: #64748b;
    --text-faint: #94a3b8;
    --accent: #3b6e71;
    --accent-light: #e8f0f1;
    --accent-dark: #2d5659;
    --gold: #b8860b;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-lg: 0 4px 16px rgba(0,0,0,0.08);
    --radius: 8px;
    --radius-lg: 12px;
}

html, body, [class*="css"] {
    font-family: 'Noto Sans SC', sans-serif;
    color: var(--text);
}

/* ── Global overrides ── */
.stApp {
    background: var(--bg);
}

section[data-testid="stSidebar"] {
    background: #fafbfc;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

h1, h2, h3, h4 { font-family: 'Noto Serif SC', serif; letter-spacing: -0.01em; }

/* ── Hero header ── */
.hero-shell {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafb 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}

.hero-shell::before {
    content: "";
    position: absolute;
    top: 0; right: 0;
    width: 200px; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(59,110,113,0.03));
    pointer-events: none;
}

.hero-kicker {
    color: var(--accent);
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 10px;
    font-weight: 600;
}

.hero-title {
    font-family: 'Noto Serif SC', serif;
    color: #0f172a;
    font-size: 1.55rem;
    line-height: 1.2;
    margin: 0 0 8px 0;
    font-weight: 700;
}

.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 640px;
}

.hero-pill-row {
    display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px;
}

.hero-pill {
    border: 1px solid var(--border);
    background: #ffffff;
    color: var(--accent);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.73rem;
    font-weight: 500;
}

/* ── Metric grid ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 10px;
    margin: 16px 0 0 0;
}

.metric-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 14px;
}

.metric-label {
    color: var(--text-faint);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}

.metric-value {
    color: #0f172a;
    font-size: 1.05rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 12px 16px !important;
    box-shadow: var(--shadow-sm);
    margin-bottom: 12px;
}

.chat-message-note {
    color: var(--text-faint);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}

.chat-message-body {
    border-radius: var(--radius);
    padding: 14px 16px;
    line-height: 1.7;
    font-size: 0.92rem;
}

.chat-message-body.user {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    color: var(--text);
}

.chat-message-body.assistant {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    color: var(--text);
}

/* ── Panels ── */
.panel-shell {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: var(--shadow-sm);
}

.panel-title {
    color: #0f172a;
    font-family: 'Noto Serif SC', serif;
    font-size: 0.98rem;
    font-weight: 700;
    margin-bottom: 6px;
}

.panel-subtitle {
    color: var(--text-muted);
    font-size: 0.8rem;
    line-height: 1.55;
}

/* ── Source evidence cards ── */
.original-source-card {
    background: #fafcfd;
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 14px 16px;
    margin: 10px 0;
    box-shadow: var(--shadow-sm);
}

.original-source-head {
    display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;
}

.source-index-badge {
    background: var(--accent);
    color: #ffffff;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.original-source-title {
    color: #0f172a;
    font-weight: 600;
    font-size: 0.88rem;
}

.original-source-text {
    color: var(--text-muted);
    font-size: 0.85rem;
    line-height: 1.75;
}

/* ── Graph shell ── */
.graph-shell {
    background: #fafcfd;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 8px;
    margin: 12px 0;
}

.triple-cloud {
    background: #f8fafb;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px 12px;
    margin-bottom: 12px;
}

.triple-tag {
    display: inline-block;
    background: var(--accent-light);
    border: 1px solid #d4e0e4;
    border-radius: 6px;
    padding: 3px 9px;
    margin: 3px;
    font-size: 0.73em;
    color: var(--accent-dark);
    font-family: 'JetBrains Mono', 'Noto Sans SC', monospace;
}

/* ── Sidebar ── */
.sidebar-section {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 14px;
    margin-bottom: 10px;
    box-shadow: var(--shadow-sm);
}

.sidebar-kicker {
    color: var(--accent);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
    font-weight: 600;
}

/* ── Buttons ── */
.stButton > button {
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius) !important;
    background: #ffffff !important;
    color: var(--text) !important;
    font-weight: 500 !important;
    box-shadow: var(--shadow-sm);
    transition: all 0.15s ease;
}

.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: var(--shadow);
    transform: translateY(-1px);
}

.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #ffffff !important;
}

.stButton > button[kind="primary"]:hover {
    background: var(--accent-dark) !important;
    color: #ffffff !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input {
    border-radius: var(--radius) !important;
    border-color: var(--border-strong) !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(59,110,113,0.15) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius);
    padding: 6px 14px;
    font-size: 0.85rem;
    border: 1px solid var(--border);
    background: var(--surface-alt);
    color: var(--text-muted);
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: var(--accent) !important;
    border-color: var(--accent) !important;
    font-weight: 600;
}

/* ── Query note (contextual hint) ── */
.query-note {
    background: #f8fafb;
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    padding: 10px 14px;
    color: var(--text-muted);
    font-size: 0.8rem;
    line-height: 1.55;
    margin: 6px 0 12px 0;
}

/* ── Evidence lead ── */
.evidence-lead {
    background: linear-gradient(135deg, #f4f8f9, #ffffff);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
    margin-bottom: 14px;
}

.evidence-lead-title {
    color: #0f172a;
    font-size: 0.92rem;
    font-weight: 700;
    margin-bottom: 4px;
}

.evidence-lead-body {
    color: var(--text-muted);
    font-size: 0.8rem;
    line-height: 1.55;
}

/* ── Responsive ── */
@media (max-width: 900px) {
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
    .hero-title { font-size: 1.3rem; }
}

@media (max-width: 600px) {
    .metric-grid { grid-template-columns: 1fr; }
    .hero-shell { padding: 14px 16px; }
}

/* ── Misc ── */
hr { border-color: var(--border) !important; }
.block-caption { color: #334155; font-weight: 600; margin: 8px 0 8px 0; font-size: 0.82rem; }
.soft-caption { color: var(--text-faint); font-size: 0.73rem; }
</style>
"""
