"""
ScoreIQ — Academic Performance Intelligence
Refined · Hierarchical · Professional
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib, os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="ScoreIQ",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── PAGE STATE via query params ────────────────────────────────
params = st.query_params
if "page" not in st.session_state:
    st.session_state.page = params.get("page", "predict")

def go_to(p):
    st.session_state.page = p
    st.query_params["page"] = p
    st.rerun()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:         #F8F7F4;
    --surface:    #FFFFFF;
    --border:     #E5E2DC;
    --border2:    #EEEBE5;
    --text:       #18160F;
    --text2:      #5C5852;
    --text3:      #9C9890;
    --accent:     #2A5F49;
    --accent2:     #000000;
    --accent-bg:  #EDF5F1;
    --accent-brd: #B8D9CB;
    --gold:       #C47C0A;
}

html, body, [class*="css"], .stApp {
    font-family: 'Geist', system-ui, sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    -webkit-font-smoothing: antialiased;
    letter-spacing: -0.015em;
}

[data-testid="collapsedControl"],
section[data-testid="stSidebar"],
#MainMenu, footer, header { display: none !important; visibility: hidden !important; }

.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* Hide all nav stButtons */
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button {
    display: none !important;
}

/* ── NAV ── */
.nav {
    background: rgba(248,247,244,0.94);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 2.5rem 0;
    position: sticky; top: 0; z-index: 999;
}
.nav-brand { display: flex; align-items: center; gap: 12px; }
.nav-wordmark {
    font-family: 'Instrument Serif', serif;
    font-size: 3rem; color: var(--text);
    letter-spacing: -0.02em; line-height: 1;
}
.nav-wordmark em { color: var(--accent); font-style: italic; }
.nav-badge {
    font-size: 0.73rem; font-weight: 600;
    color: var(--accent); background: var(--accent-bg);
    border: 1px solid var(--accent-brd);
    padding: 2px 9px; border-radius: 20px;
    letter-spacing: 0.05em; text-transform: uppercase;
}
.nav-links { display: flex; gap: 10px; align-items: center; }
.nav-link {
    font-size: 0.9rem; font-weight: 500;
    color: var(--text3); padding: 7px 16px;
    border-radius: 5px; transition: all .15s;
    cursor: pointer; text-decoration: none;
    user-select: none;
}
.nav-link:hover { color: var(--text); background: var(--border2); }
.nav-link.active {
    background: var(--text); color: var(--bg);
    font-weight: 700;
}

/* ── SHELL ── */
.shell {
    max-width: 1140px; margin: 0 auto;
    padding: 0.5rem 0.25rem 2rem;
    position: relative; z-index: 1;
}

/* ── PAGE HEADER ── */
# .page-header { margin-bottom: 2.75rem; }
.page-eyebrow {
    font-size: 0.67rem; font-weight: 700;
    color: var(--accent); letter-spacing: 0.13em;
    text-transform: uppercase; margin-bottom: 0.65rem;
    display: flex; align-items: center; gap: 8px;
}
.page-eyebrow::before {
    content: ''; width: 18px; height: 2px;
    background: var(--accent); border-radius: 2px;
}
.page-title {
    font-family: 'Instrument Serif', serif;
    font-size: 2.7rem; color: var(--text);
    letter-spacing: -0.03em; line-height: 1.08;
    margin-bottom: 0.7rem;
}
.page-title em { color: var(--accent); }
.page-desc {
    font-size: 0.9rem; color: var(--text2);
    max-width: 460px; line-height: 1.7; font-weight: 400;
}

/* ── SECTION LABELS ── */
.sec-label {
    font-size: 0.67rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.13em;
    color: var(--text3); margin-bottom: 0.9rem;
    display: flex; align-items: center; gap: 10px;
}
.sec-label::after {
    content: ''; flex: 1; height: 1px; background: var(--border2);
}

/* ── CARD ── */
# .card {
#     background: var(--surface);
#     border: 1px solid var(--border);
#     border-radius: 16px; padding: 1.75rem;
#     box-shadow: 0 1px 4px rgba(0,0,0,0.05);
# }

/* ── SCORE BLOCK ── */
.score-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px; padding: 2.25rem 2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    position: relative; overflow: hidden;
}
.score-block::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2) 60%, transparent 100%);
}
.score-glow {
    position: absolute; bottom: -40px; right: -40px;
    width: 160px; height: 160px; border-radius: 50%;
    background: radial-gradient(circle, rgba(42,95,73,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.score-primary {
    font-family: 'Geist Mono', monospace;
    font-size: 6.5rem; font-weight: 600; line-height: 1;
    letter-spacing: -0.07em; color: var(--text);
}
.score-primary .dec { font-size: 3rem; color: var(--text3); font-weight: 400; }
.score-label {
    font-size: 0.7rem; font-weight: 700; color: var(--text3);
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 6px;
}
.grade-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 16px; border-radius: 24px;
    font-size: 0.78rem; font-weight: 700;
    margin-top: 16px; border: 1.5px solid transparent;
}

/* ── GRADE SCALE ── */
.gs-divider { height: 1px; background: var(--border2); margin: 1.25rem 0 0.9rem; }
.gs-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 5px 10px; border-radius: 8px; margin-bottom: 3px;
}
.gs-row.active { background: var(--accent-bg); }
.gs-range { font-family: 'Geist Mono', monospace; font-size: 0.69rem; color: var(--text3); font-weight: 500; }
.gs-letter { font-size: 0.85rem; font-weight: 700; }

/* ── INSIGHTS ── */
.insight {
    display: flex; gap: 11px; padding: 11px 13px;
    border-radius: 11px; margin-bottom: 7px; border: 1px solid transparent;
}
.insight.ok   { background: #EDF8F2; border-color: #B8D9CB; }
.insight.warn { background: #FDF8EC; border-color: #F0D898; }
.insight.bad  { background: #FEF2F2; border-color: #FDC5C5; }
.insight.info { background: #EFF5FF; border-color: #BFCFFE; }
.insight-ico  { font-size: 13px; margin-top: 1px; flex-shrink: 0; }
.insight-title { font-size: 0.78rem; font-weight: 700; color: var(--text); margin-bottom: 2px; }
.insight-body  { font-size: 0.71rem; color: var(--text2); line-height: 1.55; }

/* ── GAIN BOX ── */
.gain-box {
    background: linear-gradient(140deg, #183828 0%, #2A5F49 100%);
    border-radius: 14px; padding: 1.4rem 1.6rem;
    margin-top: 1rem; position: relative; overflow: hidden;
}
.gain-box::before {
    content: ''; position: absolute; top: -30px; right: -30px;
    width: 100px; height: 100px; border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.gain-eyebrow { font-size: 0.62rem; font-weight: 700; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 5px; }
.gain-val { font-family: 'Geist Mono', monospace; font-size: 2.4rem; font-weight: 600; color: #fff; letter-spacing: -0.05em; line-height: 1; }
.gain-sub { font-size: 0.69rem; color: rgba(255,255,255,0.38); margin-top: 6px; }

/* ── SIM CARDS ── */
.sim-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 13px; padding: 1.2rem 0.9rem; text-align: center;
    transition: border-color .2s, box-shadow .2s;
}
.sim-card:hover { border-color: var(--accent-brd); box-shadow: 0 4px 14px rgba(42,95,73,0.09); }
.sim-lbl { font-size: 0.62rem; font-weight: 700; color: var(--text3); text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 9px; }
.sim-score { font-family: 'Geist Mono', monospace; font-size: 2.2rem; font-weight: 600; letter-spacing: -0.05em; color: var(--text); }
.sim-delta { font-family: 'Geist Mono', monospace; font-size: 0.74rem; font-weight: 600; margin-top: 5px; }
.pos { color: #059669; } .neg { color: #DC2626; } .neu { color: var(--text3); }

/* ── BASELINE BANNER ── */
.baseline-banner {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 0 14px 14px 0;
    padding: 1.1rem 1.75rem; margin: 1.5rem 0 2.25rem;
    display: flex; align-items: center; gap: 2rem;
}
.baseline-num {
    font-family: 'Geist Mono', monospace;
    font-size: 2.8rem; font-weight: 600;
    color: var(--accent); letter-spacing: -0.06em;
}

/* ── SLIDERS ── */
.stSlider label { font-size: 0.76rem !important; font-weight: 600 !important; color: var(--text) !important; }
div[data-testid="stSlider"] > div > div > div { background: var(--border) !important; height: 4px !important; }
div[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(42,95,73,0.15) !important;
}
.stSelectbox label { font-size: 0.76rem !important; font-weight: 600 !important; color: var(--text) !important; }
.stSelectbox > div > div {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 9px !important; font-size: 0.84rem !important;
    font-weight: 500 !important; font-family: 'Geist', sans-serif !important;
}

/* ── ABOUT STAT ── */
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1.75rem; }
.stat-val { font-family: 'Instrument Serif', serif; font-size: 2.5rem; color: var(--text); letter-spacing: -0.03em; margin-bottom: 5px; line-height: 1; }
.stat-lbl { font-size: 0.7rem; font-weight: 700; color: var(--text3); text-transform: uppercase; letter-spacing: 0.08em; }
.stat-sub { font-size: 0.69rem; color: var(--text3); margin-top: 4px; }

/* ── FOOTER ── */
.app-footer {
    text-align: center; font-size: 0.66rem; color: var(--text3);
    padding: 2.5rem 0 1rem; margin-top: 4rem;
    border-top: 1px solid var(--border2);
    letter-spacing: 0.06em; font-weight: 500;
}
.footer-sep { color: var(--border); margin: 0 8px; }
</style>
""", unsafe_allow_html=True)

# ── MODEL ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists('student_model.pkl') and os.path.exists('student_scaler.pkl'):
        return joblib.load('student_model.pkl'), joblib.load('student_scaler.pkl')
    np.random.seed(42); n = 5000
    X = np.column_stack([
        np.random.randint(1,44,n), np.random.randint(60,100,n),
        np.random.randint(4,10,n), np.random.randint(50,100,n),
        np.random.randint(0,8,n),  np.random.randint(0,6,n),
        np.random.randint(0,3,n),  np.random.randint(0,3,n),
        np.random.randint(0,3,n),  np.random.randint(0,2,n),
        np.random.randint(0,3,n),  np.random.randint(0,3,n),
    ])
    y = (40 + X[:,0]*0.85 + (X[:,1]-75)*0.3 + X[:,3]*0.25 + X[:,4]*1.2
           + X[:,5]*0.4 + np.random.normal(0,3,n)).clip(40,100)
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    m  = Ridge(alpha=1.0); m.fit(Xs, y)
    return m, sc

model, scaler = load_model()

def predict(f):
    return float(np.clip(model.predict(scaler.transform(np.array([f])))[0], 40, 100))

def enc(v):  return {"Low":0,"Medium":1,"High":2}.get(v,1)
def encb(v): return 1 if v == "Yes" else 0

def grade_info(s):
    if   s >= 90: return "A+", "#059669", "#EDF8F2", "#B8D9CB"
    elif s >= 80: return "A",  "#1D4ED8", "#EFF5FF", "#BFCFFE"
    elif s >= 70: return "B",  "#C47C0A", "#FDF8EC", "#F0D898"
    elif s >= 60: return "C",  "#C2490A", "#FFF5EE", "#FECDAA"
    else:         return "D",  "#DC2626", "#FEF2F2", "#FDC5C5"

CHART_DEFAULTS = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Geist', color='#9C9890', size=11),
)

# ── NAV ────────────────────────────────────────────────────────
page = st.session_state.page
pages = [("predict","Predict"), ("simulator","Simulator"), ("analytics","Analytics"), ("about","About")]

nav_links = "".join(
    f'<a class="nav-link {"active" if page==k else ""}" href="?page={k}" target="_self">{v}</a>'
    for k, v in pages
)
st.markdown(f"""
<div class="nav">
  <div class="nav-brand">
    <span class="nav-wordmark">Score<em>IQ</em></span>
    <span class="nav-badge">Ridge · 5K</span>
  </div>
  <div class="nav-links">{nav_links}</div>
</div>
""", unsafe_allow_html=True)

_nc = st.columns(len(pages))
for _c, (_k, _l) in zip(_nc, pages):
    with _c:
        if st.button(_l, key=f"nav_{_k}"):
            go_to(_k)

# ── INPUT BLOCK ────────────────────────────────────────────────
def input_block(prefix=""):
    st.markdown('<div class="sec-label">Academic Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: hours  = st.slider("Hours Studied / Week",   1, 44, 20, key=f"{prefix}h")
    with c2: attend = st.slider("Attendance Rate (%)",   60,100, 85, key=f"{prefix}a")
    with c3: prev   = st.slider("Previous Test Score",   50,100, 75, key=f"{prefix}p")
    with c4: tutor  = st.slider("Tutoring Sessions / Mo", 0,  8,  2, key=f"{prefix}t")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Lifestyle & Environment</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    l1,l2,l3,l4,l5,l6,l7,l8 = st.columns(8)
    with l1: sleep    = st.slider("Sleep (hrs/night)",  4,10, 7, key=f"{prefix}sl")
    with l2: phys     = st.slider("Physical Activity",  0, 6, 2, key=f"{prefix}ph")
    with l3: parental = st.selectbox("Parental Involvement", ["Low","Medium","High"], index=1, key=f"{prefix}par")
    with l4: res      = st.selectbox("Access to Resources",  ["Low","Medium","High"], index=1, key=f"{prefix}res")
    with l5: motiv    = st.selectbox("Motivation Level",     ["Low","Medium","High"], index=1, key=f"{prefix}mot")
    with l6: inet     = st.selectbox("Internet Access",      ["Yes","No"],            key=f"{prefix}inet")
    with l7: tq       = st.selectbox("Teacher Quality",      ["Low","Medium","High"], index=1, key=f"{prefix}tq")
    with l8: income   = st.selectbox("Family Income",        ["Low","Medium","High"], index=1, key=f"{prefix}inc")
    st.markdown('</div>', unsafe_allow_html=True)

    feats = [hours, attend, sleep, prev, tutor, phys,
             enc(parental), enc(res), enc(motiv), encb(inet), enc(income), enc(tq)]
    return feats, hours, attend, sleep, prev, tutor, phys, motiv, tq

# ══════════════════════════════════════════════════════════════
#  PAGE: PREDICT
# ══════════════════════════════════════════════════════════════
if page == "predict":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Score Prediction</div>
      <div class="page-title">Student Performance<br><em>Intelligence</em></div>
      <div class="page-desc">Adjust the profile below — results update live as you move the sliders.</div>
    </div>
    """, unsafe_allow_html=True)
    
    feats, hours, attend, sleep, prev, tutor, phys, motiv, tq = input_block("pr_")
    score = predict(feats)
    grade, gcol, gbg, gbd = grade_info(score)

    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Prediction Results</div>', unsafe_allow_html=True)

    col_score, col_charts, col_insights = st.columns([1.1, 2.3, 1.4], gap="medium")

    with col_score:
        i_part = int(score)
        d_part = f"{score:.1f}".split('.')[1]
        st.markdown(f"""
        <div class="score-block">
          <div class="score-glow"></div>
          <div class="score-primary">{i_part}<span class="dec">.{d_part}</span></div>
          <div class="score-label">Predicted Score · out of 100</div>
          <span class="grade-badge" style="background:{gbg};color:{gcol};border-color:{gbd};">
            ● &nbsp;Grade {grade}
          </span>
          <div class="gs-divider"></div>
        """, unsafe_allow_html=True)
        for rng, g, c in [
            ("90–100","A+","#059669"),("80–89","A","#1D4ED8"),
            ("70–79","B","#C47C0A"),("60–69","C","#C2490A"),("<60","D","#DC2626")
        ]:
            active = "active" if g == grade else ""
            st.markdown(f'<div class="gs-row {active}"><span class="gs-range">{rng}</span><span class="gs-letter" style="color:{c}">{g}</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_charts:
        fv = {
            "Study Hours":   min(hours/44,1),
            "Attendance":    (attend-60)/40,
            "Sleep Quality": (sleep-4)/6,
            "Prev Score":    (prev-50)/50,
            "Tutoring":      tutor/8,
            "Motivation":    enc(motiv)/2,
            "Teacher Q.":    enc(tq)/2,
        }
        labs = list(fv.keys()); vals = list(fv.values())
        bar_c = ["#2A5F49" if v>=.7 else "#C47C0A" if v>=.4 else "#DC2626" for v in vals]

        fig = go.Figure(go.Bar(
            x=vals, y=labs, orientation='h',
            marker=dict(color=bar_c, opacity=0.88, line=dict(width=0), cornerradius=4),
            text=[f"{v:.0%}" for v in vals], textposition='outside',
            textfont=dict(size=10, color='#9C9890', family='Geist Mono'),
            hovertemplate='%{y}: %{x:.1%}<extra></extra>', showlegend=False, width=0.52,
        ))
        fig.update_layout(
            **CHART_DEFAULTS,
            margin=dict(l=0, r=52, t=4, b=4), height=225,
            xaxis=dict(range=[0,1.32], gridcolor='#EEEBE5', zeroline=False,
                       tickformat='.0%', tickfont=dict(size=9.5, family='Geist Mono')),
            yaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=11.5, color='#18160F')),
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})

        x_s = np.arange(1, 45)
        y_s = [predict([h,attend,sleep,prev,tutor,phys,1,1,enc(motiv),1,1,enc(tq)]) for h in x_s]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x_s, y=y_s, mode='lines',
            line=dict(color='#2A5F49', width=2.5, shape='spline'),
            fill='tozeroy', fillcolor='rgba(42,95,73,0.07)',
            hovertemplate='%{x}h/wk → %{y:.1f}<extra></extra>', showlegend=False
        ))
        fig2.add_vline(x=hours, line=dict(color='#C47C0A', width=1.5, dash='dot'))
        fig2.add_annotation(x=hours, y=score+2, text=f"  {score:.0f}", showarrow=False,
                            font=dict(color='#C47C0A', size=11, family='Geist Mono'), xanchor='left')
        fig2.update_layout(
            **CHART_DEFAULTS,
            margin=dict(l=0, r=16, t=34, b=4), height=225,
            title=dict(text="Score Sensitivity · Study Hours",
                       font=dict(size=11.5, color='#5C5852', family='Geist'), x=0),
            xaxis=dict(gridcolor='#EEEBE5', zeroline=False, tickfont=dict(size=9.5, family='Geist Mono')),
            yaxis=dict(range=[38,104], gridcolor='#EEEBE5', zeroline=False, tickfont=dict(size=9.5, family='Geist Mono')),
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar':False})

    with col_insights:
        tips = []
        if hours < 10:    tips.append(("bad","⚠","Low Study Hours","Increase to 20+ hrs/week for a meaningful boost."))
        elif hours >= 30: tips.append(("ok","✓","Strong Study Habit","Top-percentile — primary performance driver."))
        else:             tips.append(("warn","→","Moderate Study","Targeting 25+ hrs/week could push the score higher."))

        if attend < 75:    tips.append(("bad","⚠","Low Attendance","Below 75% strongly predicts lower scores."))
        elif attend >= 90: tips.append(("ok","✓","Excellent Attendance","Top tier — one of the highest-impact factors."))
        else:              tips.append(("warn","→","Good Attendance","Reaching 90%+ would unlock full benefit."))

        if sleep < 6:    tips.append(("warn","⚠","Sleep Deprivation","7–8 hrs/night improves cognitive performance."))
        elif sleep >= 7: tips.append(("ok","✓","Healthy Sleep","Consistent sleep supports sustained output."))

        if tutor >= 4:   tips.append(("info","★","Active Tutoring","Frequent sessions positively lift the prediction."))
        elif tutor == 0: tips.append(("warn","→","No Tutoring","1–2 sessions/month can improve focused learning."))

        if enc(motiv) == 0: tips.append(("bad","⚠","Low Motivation","Key behavioural predictor — address this first."))

        for sev, ico, title, body in tips[:5]:
            st.markdown(f"""
            <div class="insight {sev}">
              <span class="insight-ico">{ico}</span>
              <div><div class="insight-title">{title}</div><div class="insight-body">{body}</div></div>
            </div>""", unsafe_allow_html=True)

        max_s = predict([44,100,9,100,8,6,2,2,2,1,2,2])
        gain  = max(0, max_s - score)
        st.markdown(f"""
        <div class="gain-box">
          <div class="gain-eyebrow">Potential Gain</div>
          <div class="gain-val">+{gain:.1f}</div>
          <div class="gain-sub">points with an optimised profile</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: SIMULATOR
# ══════════════════════════════════════════════════════════════
elif page == "simulator":
    st.markdown('<div class="shell">', unsafe_allow_html=True)
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">What-If Analysis</div>
      <div class="page-title">Scenario <em>Simulator</em></div>
      <div class="page-desc">Set a baseline profile, then compare how targeted improvements shift the predicted score.</div>
    </div>
    """, unsafe_allow_html=True)

    feats, hours, attend, sleep, prev, tutor, phys, motiv, tq = input_block("sim_")
    score = predict(feats)
    grade, gcol, gbg, gbd = grade_info(score)

    st.markdown(f"""
    <div class="baseline-banner">
      <div>
        <div style="font-size:.65rem;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:.12em;margin-bottom:3px;">Baseline Score</div>
        <div class="baseline-num">{score:.1f}</div>
      </div>
      <span class="grade-badge" style="background:{gbg};color:{gcol};border-color:{gbd};">● &nbsp;Grade {grade}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Improvement Scenarios</div>', unsafe_allow_html=True)

    scenarios = [
        ("+5 Study Hrs",     [min(hours+5,44),  attend,sleep,prev,tutor,phys,1,1,enc(motiv),1,1,enc(tq)]),
        ("+10 Study Hrs",    [min(hours+10,44), attend,sleep,prev,tutor,phys,1,1,enc(motiv),1,1,enc(tq)]),
        ("95% Attendance",   [hours,95,sleep,prev,tutor,phys,1,1,enc(motiv),1,1,enc(tq)]),
        ("8 hrs Sleep",      [hours,attend,8,prev,tutor,phys,1,1,enc(motiv),1,1,enc(tq)]),
        ("4 Tutor Sessions", [hours,attend,sleep,prev,4,phys,1,1,enc(motiv),1,1,enc(tq)]),
        ("High Motivation",  [hours,attend,sleep,prev,tutor,phys,1,1,2,1,1,enc(tq)]),
        ("All Improved",     [min(hours+10,44),min(attend+10,100),8,prev,max(tutor,4),phys,2,2,2,1,2,2]),
    ]

    sim_vals = []
    sim_cols = st.columns(len(scenarios), gap="small")
    for col, (lbl, sf) in zip(sim_cols, scenarios):
        s = predict(sf); sim_vals.append(s)
        d = s - score
        ds = f"+{d:.1f}" if d > 0 else f"{d:.1f}"
        dc = "pos" if d > .1 else ("neg" if d < -.1 else "neu")
        with col:
            st.markdown(f"""
            <div class="sim-card">
              <div class="sim-lbl">{lbl}</div>
              <div class="sim-score">{s:.0f}</div>
              <div class="sim-delta {dc}">{ds}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
    all_labs = ["Baseline"] + [s[0] for s in scenarios]
    all_vals = [score] + sim_vals
    all_c = ["#DEDAD4"] + ["#2A5F49" if v > score+.3 else "#C8C3BB" for v in sim_vals]

    fig3 = go.Figure(go.Bar(
        x=all_labs, y=all_vals,
        marker=dict(color=all_c, cornerradius=5, line=dict(width=0)),
        text=[f"{v:.0f}" for v in all_vals], textposition='outside',
        textfont=dict(size=11, color='#9C9890', family='Geist Mono'),
        hovertemplate='%{x}: %{y:.1f}<extra></extra>', showlegend=False, width=0.65,
    ))
    fig3.add_hline(y=score, line=dict(color='#C47C0A', width=1.5, dash='dot'))
    fig3.update_layout(
        **CHART_DEFAULTS, margin=dict(l=10,r=10,t=20,b=10), height=235,
        xaxis=dict(gridcolor='rgba(0,0,0,0)', zeroline=False, tickfont=dict(size=10.5)),
        yaxis=dict(range=[35,112], gridcolor='#EEEBE5', zeroline=False, tickfont=dict(size=10, family='Geist Mono')),
    )
    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar':False})
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "analytics":
    st.markdown('<div class="shell">', unsafe_allow_html=True)
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Advanced Analytics</div>
      <div class="page-title">Sensitivity <em>Analysis</em></div>
      <div class="page-desc">Curves and a 2D score map computed against your profile in real time.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Reference Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    q1,q2,q3,q4,q5,q6 = st.columns(6)
    with q1: hours  = st.slider("Study Hrs/Wk",  1, 44, 20, key="aq1")
    with q2: attend = st.slider("Attendance %", 60,100, 85, key="aq2")
    with q3: sleep  = st.slider("Sleep Hrs",     4, 10,  7, key="aq3")
    with q4: prev   = st.slider("Prev. Score",  50,100, 75, key="aq4")
    with q5: tutor  = st.slider("Tutoring/Mo",   0,  8,  2, key="aq5")
    with q6: motiv  = st.selectbox("Motivation", ["Low","Medium","High"], index=1, key="aq6")
    st.markdown('</div>', unsafe_allow_html=True)

    base = [hours, attend, sleep, prev, tutor, 2, 1,1,enc(motiv),1,1,1]
    base_s = predict(base)

    st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Sensitivity Curves</div>', unsafe_allow_html=True)

    def curve(idx, lo, hi):
        xs = np.arange(lo, hi+1)
        ys = [predict([v if i==idx else base[i] for i in range(len(base))]) for v in xs]
        return xs.tolist(), ys

    palette = ["#2A5F49","#1D4ED8","#C47C0A","#7C3AED"]
    curve_defs = [
        (0,(1,44),  hours,  "Study Hours / Week",       "hrs/week",       palette[0]),
        (1,(60,100),attend, "Attendance Rate",           "% attendance",   palette[1]),
        (2,(4,10),  sleep,  "Sleep Hours / Night",       "hrs/night",      palette[2]),
        (4,(0,8),   tutor,  "Tutoring Sessions / Month", "sessions/month", palette[3]),
    ]

    r1 = st.columns(2, gap="medium")
    r2 = st.columns(2, gap="medium")
    for col, (idx, rng, cur, title, xlabel, color) in zip([r1[0],r1[1],r2[0],r2[1]], curve_defs):
        xs, ys = curve(idx, rng[0], rng[1])
        r = int(color[1:3],16); g = int(color[3:5],16); b = int(color[5:7],16)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color=color, width=2.5, shape='spline'),
            fill='tozeroy', fillcolor=f'rgba({r},{g},{b},0.07)',
            hovertemplate=f'{xlabel}: %{{x}} → %{{y:.1f}}<extra></extra>', showlegend=False
        ))
        fig.add_vline(x=cur, line=dict(color='#C47C0A', width=1.5, dash='dot'))
        fig.add_annotation(x=cur, y=base_s+2, text=f"  {base_s:.0f}", showarrow=False,
                           font=dict(color='#C47C0A', size=11, family='Geist Mono'), xanchor='left')
        fig.update_layout(
            **CHART_DEFAULTS,
            margin=dict(l=0, r=16, t=42, b=6), height=215,
            title=dict(text=title, font=dict(size=12.5, color='#18160F', family='Geist'), x=0),
            xaxis=dict(gridcolor='#EEEBE5', zeroline=False, tickfont=dict(size=10, family='Geist Mono')),
            yaxis=dict(range=[38,104], gridcolor='#EEEBE5', zeroline=False, tickfont=dict(size=10, family='Geist Mono')),
        )
        with col:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})

    st.markdown('<div class="sec-label">Score Map · Study Hours × Attendance</div>', unsafe_allow_html=True)
    h_grid = np.arange(5, 45, 5)
    a_grid = np.arange(65, 102, 5)
    Z = np.array([[predict([h,a,sleep,prev,tutor,2,1,1,enc(motiv),1,1,1]) for h in h_grid] for a in a_grid])

    fig_hm = go.Figure(go.Heatmap(
        x=h_grid, y=a_grid, z=Z,
        colorscale=[[0,'#FEF2F2'],[0.3,'#FDF8EC'],[0.65,'#EDF5F1'],[1,'#183828']],
        hovertemplate='Study: %{x}h · Attend: %{y}%% → %{z:.1f}<extra></extra>',
        colorbar=dict(tickfont=dict(size=10, family='Geist Mono', color='#9C9890'), thickness=12, outlinewidth=0)
    ))
    fig_hm.add_trace(go.Scatter(
        x=[hours], y=[attend], mode='markers',
        marker=dict(color='#C47C0A', size=14, symbol='cross-thin', line=dict(color='#C47C0A', width=2.5)),
        showlegend=False, hoverinfo='skip'
    ))
    fig_hm.update_layout(
        **CHART_DEFAULTS, margin=dict(l=10,r=10,t=10,b=10), height=310,
        xaxis=dict(title="Study Hours / Week", gridcolor='rgba(0,0,0,0)', tickfont=dict(size=10, family='Geist Mono')),
        yaxis=dict(title="Attendance (%)", gridcolor='rgba(0,0,0,0)', tickfont=dict(size=10, family='Geist Mono')),
    )
    st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar':False})
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "about":
    st.markdown('<div class="shell">', unsafe_allow_html=True)
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Documentation</div>
      <div class="page-title">About <em>ScoreIQ</em></div>
      <div class="page-desc">Model architecture, feature descriptions, and grade scale reference.</div>
    </div>
    """, unsafe_allow_html=True)

    s1,s2,s3,s4 = st.columns(4, gap="medium")
    for col, val, lbl, sub in [
        (s1,"12","Input Factors","Academic · Lifestyle · Environment"),
        (s2,"5,000","Training Records","Synthetic · seed 42"),
        (s3,"40–100","Output Range","Score clipped to range"),
        (s4,"Ridge α=1","Model Type","scikit-learn · linear"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-val">{val}</div>
              <div class="stat-lbl">{lbl}</div>
              <div class="stat-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Feature Reference</div>', unsafe_allow_html=True)

    features = [
        ("Hours Studied / Week",      "1 – 44",    "Numeric", "Total weekly study hours. Highest model weight."),
        ("Attendance Rate",           "60 – 100%", "Numeric", "Percentage of classes attended."),
        ("Sleep Hours / Night",       "4 – 10",    "Numeric", "Average nightly sleep. Cognitive performance proxy."),
        ("Previous Test Score",       "50 – 100",  "Numeric", "Last exam score. Strong continuity predictor."),
        ("Tutoring Sessions / Month", "0 – 8",     "Numeric", "Number of private tutoring sessions per month."),
        ("Physical Activity",         "0 – 6 hrs", "Numeric", "Weekly exercise hours."),
        ("Parental Involvement",      "Low/Med/Hi","Ordinal", "Level of parental academic engagement."),
        ("Access to Resources",       "Low/Med/Hi","Ordinal", "Educational material availability."),
        ("Motivation Level",          "Low/Med/Hi","Ordinal", "Self-reported academic motivation."),
        ("Internet Access",           "Yes / No",  "Binary",  "Home internet availability."),
        ("Teacher Quality",           "Low/Med/Hi","Ordinal", "Perceived quality of teaching staff."),
        ("Family Income",             "Low/Med/Hi","Ordinal", "Household income category."),
    ]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    hc = st.columns([2.5,1.2,1,4])
    for h, lbl in zip(hc, ["Feature","Range","Type","Description"]):
        with h:
            st.markdown(f'<div style="font-size:.63rem;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:.12em;padding:.4rem 0;">{lbl}</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:var(--border2);margin:.25rem 0 .5rem;"></div>', unsafe_allow_html=True)
    for i,(name,rng,typ,desc) in enumerate(features):
        fc = st.columns([2.5,1.2,1,4])
        bg = "background:#FAFAF7;" if i%2==0 else ""
        for col,txt,mono in zip(fc,[name,rng,typ,desc],[True,True,False,False]):
            with col:
                ff = "font-family:'Geist Mono',monospace;font-size:.77rem;" if mono else "font-size:.79rem;"
                st.markdown(f'<div style="{ff}color:var(--text);padding:.44rem 0;{bg}">{txt}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Grade Scale</div>', unsafe_allow_html=True)
    gc = st.columns(5, gap="medium")
    for col,(rng,g,c,bg) in zip(gc,[
        ("90–100","A+","#059669","#EDF8F2"),("80–89","A","#1D4ED8","#EFF5FF"),
        ("70–79","B","#C47C0A","#FDF8EC"),("60–69","C","#C2490A","#FFF5EE"),("<60","D","#DC2626","#FEF2F2"),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1.5px solid {c}28;border-radius:16px;padding:1.75rem 1rem;text-align:center;">
              <div style="font-family:'Instrument Serif',serif;font-size:3rem;color:{c};line-height:1;">{g}</div>
              <div style="font-family:'Geist Mono',monospace;font-size:.72rem;color:var(--text3);margin-top:9px;font-weight:500;">{rng}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────
st.markdown("""
<div class="shell" style="padding-top:0;padding-bottom:1rem;">
  <div class="app-footer">
    ScoreIQ v6.0
    <span class="footer-sep">·</span>Ridge Regression
    <span class="footer-sep">·</span>5,000 records
    <span class="footer-sep">·</span>12 features
    <span class="footer-sep">·</span>@abdel2ty
  </div>
</div>
""", unsafe_allow_html=True)
