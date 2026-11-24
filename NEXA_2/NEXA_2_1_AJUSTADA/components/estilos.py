import streamlit as st

def apply_style():
    st.markdown("""<style>
    .stApp {background: #090F1D; color: #E7EFFB;}
    .block-container {padding-top:1.6rem; padding-bottom:3rem; max-width:1500px;}
    [data-testid="stSidebar"] {background:#0E1829; border-right:1px solid #26344C;}
    [data-testid="stMetric"] {background:#121D30; padding:18px 20px; border:1px solid #26344C; border-radius:14px;}
    [data-testid="stMetricLabel"] {color:#A9BAD2;}
    [data-testid="stMetricValue"] {color:#F1F7FF;}
    h1,h2,h3 {color:#F0F6FF !important; letter-spacing:-.035em;}
    .nexa-hero {background:linear-gradient(115deg,#122A4C,#14203A 58%,#112F39); border:1px solid #29476C; padding:24px 28px; border-radius:18px; margin-bottom:20px;}
    .nexa-hero h1 {font-size:2rem; margin:0 0 8px 0;}
    .nexa-hero p {color:#B6C9E5; margin:0;}
    .insight {background:#121D30; border:1px solid #26344C; border-left:4px solid #60A5FA; padding:14px 18px; border-radius:10px; margin:9px 0;}
    .insight strong {color:#EAF3FF;}
    .small-muted {color:#9CAFCB; font-size:.85rem;}
    div[data-baseweb="select"] > div {background:#15243A;}
    </style>""", unsafe_allow_html=True)
