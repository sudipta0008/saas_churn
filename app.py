"""
SaaS Customer Churn Prediction — Streamlit Web App
====================================================
Deployed on Streamlit Cloud
Author : Sudipta Sarkar  |  NIT Tiruchirappalli
Guide  : Dr. S. Saroja
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
import io, base64, datetime, os

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction — NIT Trichy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* overall background */
[data-testid="stAppViewContainer"] { background-color: #F8FAFC; }
[data-testid="stSidebar"]          { background-color: #1D3557; }
[data-testid="stSidebar"] *        { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stRadio label { color: #CADCFC !important; }

/* metric cards */
div[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
div[data-testid="metric-container"] label { color: #64748B !important; font-size:13px; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size:28px; font-weight:700; }

/* headings */
h1 { color: #1D3557 !important; }
h2 { color: #1D3557 !important; border-bottom: 2px solid #457B9D; padding-bottom:6px; }
h3 { color: #457B9D !important; }

/* risk badges */
.badge-high   { background:#FFEBEE; color:#C62828; padding:3px 10px; border-radius:99px; font-size:12px; font-weight:600; }
.badge-medium { background:#FFF9E6; color:#E65100; padding:3px 10px; border-radius:99px; font-size:12px; font-weight:600; }
.badge-low    { background:#E8F5E9; color:#2E7D32; padding:3px 10px; border-radius:99px; font-size:12px; font-weight:600; }

/* dataframe */
[data-testid="stDataFrame"] { border-radius:8px; overflow:hidden; }

/* tabs */
.stTabs [data-baseweb="tab"] { font-size:15px; font-weight:500; }
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
FEATURES = ["Account_Age_Days", "Daily_Usage_Mins", "login_freq_enc", "sentiment"]
HIGH_THRESH = 0.75
MED_THRESH  = 0.50

# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    base = os.path.dirname(__file__)
    train = pd.read_csv(os.path.join(base, "churn_dataset.csv"))
    test  = pd.read_csv(os.path.join(base, "charn_dataset2.csv"))
    return train, test

@st.cache_data(show_spinner=False)
def engineer(train_df, test_df):
    for df in [train_df, test_df]:
        df["sentiment"] = df["Last_Support_Ticket"].apply(
            lambda t: TextBlob(str(t)).sentiment.polarity
        )
    le = LabelEncoder()
    train_df["login_freq_enc"] = le.fit_transform(train_df["Login_Frequency"])
    test_df["login_freq_enc"]  = le.transform(test_df["Login_Frequency"])
    return train_df, test_df, le

@st.cache_resource(show_spinner=False)
def train_models(train_df):
    X, y = train_df[FEATURES], train_df["Churn"]
    models = {
        "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    for m in models.values():
        m.fit(X, y)
    return models

def score_customers(model, test_df, le):
    scored = test_df.copy()
    scored["churn_probability"] = model.predict_proba(test_df[FEATURES])[:, 1]
    scored["churn_score_pct"]   = (scored["churn_probability"]*100).round(1).astype(str) + "%"
    scored["risk_level"] = scored["churn_probability"].apply(
        lambda s: "HIGH" if s >= HIGH_THRESH else "MEDIUM" if s >= MED_THRESH else "LOW"
    )
    return scored

def badge(risk):
    cls = {"HIGH":"badge-high","MEDIUM":"badge-medium","LOW":"badge-low"}[risk]
    return f'<span class="{cls}">{risk}</span>'

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "NITT_logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)

    st.markdown("### NIT Tiruchirappalli")
    st.markdown("**Dept. of Computer Science**")
    st.markdown("---")
    st.markdown("**Project:** SaaS Churn Prediction")
    st.markdown("**Author:** Sudipta Sarkar")
    st.markdown("**Guide:** Dr. S. Saroja")
    st.markdown("**Accuracy:** 83.9%")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 Exploratory Analysis",
        "🔤 NLP Sentiment",
        "🤖 ML Models",
        "🚨 Risk Dashboard",
        "📧 Alert Simulator",
        "🔍 Predict Single Customer",
    ])

    st.markdown("---")
    st.markdown("""
    <small>
    📚 References:<br>
    IEEE Access 2024<br>
    DOI: 10.1109/ACCESS.2024.3402092
    </small>
    """, unsafe_allow_html=True)


# ── load & prepare data ───────────────────────────────────────────────────────
with st.spinner("Loading data and training models..."):
    train_raw, test_raw = load_data()
    train, test, le = engineer(train_raw.copy(), test_raw.copy())
    models = train_models(train)
    best_model = models["Logistic Regression"]
    scored_df  = score_customers(best_model, test, le)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("SaaS Customer Churn Prediction")
    st.markdown("**Using Machine Learning & Natural Language Processing**")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers Scored", f"{len(test):,}")
    col2.metric("Model Accuracy", "83.9%", "+17.4pp vs baseline")
    col3.metric("HIGH Risk Customers",
                f"{(scored_df['risk_level']=='HIGH').sum():,}",
                f"{(scored_df['risk_level']=='HIGH').mean()*100:.1f}% of total")
    col4.metric("Churners Identified (Recall)", "74%")

    st.markdown("---")
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.subheader("What This App Does")
        st.markdown("""
        This system predicts customer churn for a SaaS platform by combining:

        - **Exploratory Data Analysis** — understanding usage patterns
        - **NLP Sentiment Analysis** — extracting emotion from support tickets
        - **Machine Learning** — 3 models trained and compared
        - **Automated Alerts** — 4 business workflows fired at-risk customers

        **Research gap addressed:** The 2024 IEEE Access review of 212 churn prediction
        papers identified NLP features and automated workflows as under-explored.
        This project implements both.
        """)

    with c2:
        st.subheader("Key Findings")
        data = {
            "Login Frequency": ["Daily", "Weekly", "Rarely"],
            "Churn Rate": ["16.4%", "37.4%", "82.7%"],
            "Risk":       ["LOW 🟢", "MEDIUM 🟡", "HIGH 🔴"],
        }
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        st.info("Customers who rarely log in churn at **82.7%** — the strongest single signal.")

    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown("""
    ```
    CSV Data  →  Feature Engineering  →  NLP Sentiment
                         ↓
               Logistic Regression (83.9%)
                         ↓
               Customer Risk Scoring
                         ↓
    HIGH Risk → Email Alert + Slack + Retention Email + CRM Task
    MED  Risk → Slack + Retention Email
    ```
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Exploratory Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Churn Distribution", "Login Frequency", "Usage Patterns"])

    with tab1:
        st.subheader("How Many Customers Churned?")
        img_path = os.path.join(os.path.dirname(__file__), "assets", "step1_churn_distribution.png")
        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers", f"{len(train):,}")
        c2.metric("Churned (1)", f"{train['Churn'].sum():,}", f"{train['Churn'].mean()*100:.1f}%")
        c3.metric("Naive Baseline", "65.8%", "Our model beats this")
        st.info("💡 A naive model that always predicts 'retained' would be 65.8% accurate. Our ML model achieves **83.9%** — a 17.4 point improvement.")

    with tab2:
        st.subheader("Login Frequency vs Churn Rate")
        img_path = os.path.join(os.path.dirname(__file__), "assets", "step2_login_vs_churn.png")
        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)
        freq_data = train.groupby("Login_Frequency")["Churn"].agg(["mean","count"]).reset_index()
        freq_data["Churn Rate %"] = (freq_data["mean"]*100).round(1)
        freq_data = freq_data.rename(columns={"Login_Frequency":"Frequency","count":"Customers"})
        freq_data = freq_data[["Frequency","Customers","Churn Rate %"]]
        st.dataframe(freq_data, use_container_width=True, hide_index=True)
        st.warning("⚠️ Customers who **Rarely** log in churn at **82.7%** — almost certain to cancel.")

    with tab3:
        st.subheader("Usage Pattern Distributions")
        img_path = os.path.join(os.path.dirname(__file__), "assets", "step3_usage_distributions.png")
        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Retained avg daily usage", f"{train[train['Churn']==0]['Daily_Usage_Mins'].mean():.1f} min/day")
        c2.metric("Churned avg daily usage",  f"{train[train['Churn']==1]['Daily_Usage_Mins'].mean():.1f} min/day", delta="-24 min/day", delta_color="inverse")
        st.info("💡 Churned customers use the platform **less than half** as much as retained ones.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NLP SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔤 NLP Sentiment":
    st.title("NLP — Sentiment Analysis on Support Tickets")
    st.markdown("---")

    st.subheader("What is Sentiment Analysis?")
    st.markdown("""
    We use **TextBlob** to assign every support ticket a polarity score:
    - **-1.0** = Very Negative (frustrated customer)
    - **0.0**  = Neutral
    - **+1.0** = Very Positive (happy customer)
    """)

    img_path = os.path.join(os.path.dirname(__file__), "assets", "step4_sentiment_analysis.png")
    if os.path.exists(img_path):
        st.image(img_path, use_column_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Retained customers avg sentiment", "+0.154")
    c2.metric("Churned customers avg sentiment",  "-0.091", delta="-0.245", delta_color="inverse")

    st.markdown("---")
    st.subheader("Try It — Live Sentiment Scorer")
    st.markdown("Type a support ticket below and see the sentiment score:")

    user_ticket = st.text_area(
        "Support ticket text:",
        placeholder="e.g. I cannot find the export button. Very frustrating.",
        height=100,
    )
    if user_ticket.strip():
        blob = TextBlob(user_ticket)
        score = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        col1, col2, col3 = st.columns(3)
        col1.metric("Polarity Score", f"{score:+.3f}")
        col2.metric("Subjectivity", f"{subjectivity:.3f}")
        if score < -0.1:
            col3.error("😠 Negative — HIGH CHURN RISK")
        elif score > 0.1:
            col3.success("😊 Positive — LOW CHURN RISK")
        else:
            col3.info("😐 Neutral — MODERATE RISK")

        if score < -0.3:
            st.error("🔴 This customer is very unhappy. Immediate outreach recommended.")
        elif score < -0.1:
            st.warning("🟡 This customer shows signs of dissatisfaction. Monitor closely.")
        elif score > 0.2:
            st.success("🟢 This customer appears satisfied. Low churn risk.")
        else:
            st.info("⚪ Neutral sentiment detected.")

    st.markdown("---")
    st.subheader("Sample Tickets from Dataset")
    sample = train[["Last_Support_Ticket","sentiment","Churn"]].sample(8, random_state=7)
    sample["Churn"] = sample["Churn"].map({0:"Retained ✅", 1:"Churned ❌"})
    sample["sentiment"] = sample["sentiment"].round(3)
    sample = sample.rename(columns={
        "Last_Support_Ticket":"Support Ticket",
        "sentiment":"Sentiment Score",
        "Churn":"Status"
    })
    st.dataframe(sample, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.title("Machine Learning — 3 Models Trained & Compared")
    st.markdown("---")

    # model results
    results = {}
    for name, model in models.items():
        preds = model.predict(test[FEATURES])
        acc   = accuracy_score(test["Churn"], preds)
        report = classification_report(test["Churn"], preds, output_dict=True)
        results[name] = {"acc": acc, "preds": preds, "report": report}

    col1, col2, col3 = st.columns(3)
    for col, (name, res) in zip([col1, col2, col3], results.items()):
        delta = f"+{(res['acc']-0.658)*100:.1f}pp vs baseline"
        col.metric(name, f"{res['acc']*100:.1f}%", delta)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Accuracy Comparison", "Confusion Matrix", "Feature Importance"])

    with tab1:
        img_path = os.path.join(os.path.dirname(__file__), "assets", "step5_model_comparison.png")
        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)

        st.markdown("#### Full Classification Report — Logistic Regression")
        report = results["Logistic Regression"]["report"]
        report_df = pd.DataFrame({
            "Class":     ["Retained (0)", "Churned (1)", "Macro Avg"],
            "Precision": [f"{report['0']['precision']:.3f}", f"{report['1']['precision']:.3f}", f"{report['macro avg']['precision']:.3f}"],
            "Recall":    [f"{report['0']['recall']:.3f}",    f"{report['1']['recall']:.3f}",    f"{report['macro avg']['recall']:.3f}"],
            "F1 Score":  [f"{report['0']['f1-score']:.3f}", f"{report['1']['f1-score']:.3f}", f"{report['macro avg']['f1-score']:.3f}"],
        })
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        st.info("""
        **Why Logistic Regression won:**
        With only 500 training rows and 4 features, simpler models generalise better.
        Ensemble methods (Random Forest, Gradient Boosting) need more data to outperform.
        With 40,000+ rows, the ensemble methods would likely win.
        """)

    with tab2:
        img_path = os.path.join(os.path.dirname(__file__), "assets", "step6_confusion_matrix.png")
        if os.path.exists(img_path):
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image(img_path, use_column_width=True)

        cm = confusion_matrix(test["Churn"], results["Logistic Regression"]["preds"])
        cm_df = pd.DataFrame({
            "Term":            ["True Negative","True Positive","False Positive","False Negative"],
            "Meaning":         ["Predicted retained, was retained",
                                "Predicted churned, was churned",
                                "Predicted churned, actually stayed",
                                "Predicted retained, actually churned"],
            "Count":           [cm[0,0], cm[1,1], cm[0,1], cm[1,0]],
            "Business Impact": ["Correct — no action needed",
                                "Correct — intervention triggered",
                                "Minor cost — unnecessary email",
                                "COSTLY — missed churner!"],
        })
        st.dataframe(cm_df, use_container_width=True, hide_index=True)

    with tab3:
        model_lr = models["Logistic Regression"]
        importances = np.abs(model_lr.coef_[0])
        importances = importances / importances.sum()
        feat_labels = ["Account Age", "Daily Usage", "Login Freq", "Sentiment"]

        fig, ax = plt.subplots(figsize=(7, 3))
        colors = ["#1D3557" if i == importances.argmax() else "#457B9D" for i in range(len(feat_labels))]
        bars = ax.barh(feat_labels, importances, color=colors, edgecolor="white")
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance — Logistic Regression", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, importances):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info("Login Frequency and Daily Usage are the dominant predictors. Sentiment adds a meaningful NLP signal.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RISK DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Risk Dashboard":
    st.title("Customer Risk Dashboard")
    st.markdown("---")

    # summary metrics
    high   = scored_df[scored_df["risk_level"] == "HIGH"]
    medium = scored_df[scored_df["risk_level"] == "MEDIUM"]
    low    = scored_df[scored_df["risk_level"] == "LOW"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",  f"{len(scored_df):,}")
    c2.metric("HIGH Risk 🔴",     f"{len(high):,}",   f"{len(high)/len(scored_df)*100:.1f}%")
    c3.metric("MEDIUM Risk 🟡",   f"{len(medium):,}", f"{len(medium)/len(scored_df)*100:.1f}%")
    c4.metric("LOW Risk 🟢",      f"{len(low):,}",    f"{len(low)/len(scored_df)*100:.1f}%")

    st.markdown("---")

    # filter controls
    col1, col2, col3 = st.columns(3)
    risk_filter  = col1.multiselect("Filter by Risk Level", ["HIGH","MEDIUM","LOW"], default=["HIGH","MEDIUM"])
    freq_filter  = col2.multiselect("Filter by Login Frequency", ["Daily","Weekly","Rarely"], default=["Daily","Weekly","Rarely"])
    top_n        = col3.slider("Show top N customers", 10, 200, 50)

    filtered = scored_df[
        (scored_df["risk_level"].isin(risk_filter)) &
        (scored_df["Login_Frequency"].isin(freq_filter))
    ].sort_values("churn_probability", ascending=False).head(top_n)

    st.markdown(f"**Showing {len(filtered):,} customers**")

    # display table
    cols_show = ["Name", "Email", "churn_score_pct", "risk_level",
                 "Login_Frequency", "Account_Age_Days", "Daily_Usage_Mins", "sentiment"]
    cols_show = [c for c in cols_show if c in filtered.columns]
    display   = filtered[cols_show].copy()
    display   = display.rename(columns={
        "churn_score_pct":  "Churn Score",
        "risk_level":       "Risk Level",
        "Login_Frequency":  "Login Freq",
        "Account_Age_Days": "Account Age",
        "Daily_Usage_Mins": "Daily Usage",
        "sentiment":        "Sentiment",
    })
    if "Sentiment" in display.columns:
        display["Sentiment"] = display["Sentiment"].round(3)

    st.dataframe(display, use_container_width=True, hide_index=True, height=420)

    # download button
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered List as CSV",
        data=csv,
        file_name=f"at_risk_customers_{datetime.date.today()}.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Risk Distribution Chart")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    risk_counts = scored_df["risk_level"].value_counts()[["HIGH","MEDIUM","LOW"]]
    colors = ["#E63946","#F4A261","#2A9D8F"]
    axes[0].pie(risk_counts.values, labels=risk_counts.index, autopct="%1.1f%%",
                colors=colors, startangle=90, textprops={"fontsize":11})
    axes[0].set_title("Risk Level Distribution", fontweight="bold")
    axes[1].hist(scored_df["churn_probability"], bins=30, color="#457B9D",
                 edgecolor="white", linewidth=0.5)
    axes[1].axvline(HIGH_THRESH, color="#E63946", linestyle="--", label=f"High threshold ({HIGH_THRESH})")
    axes[1].axvline(MED_THRESH,  color="#F4A261", linestyle="--", label=f"Med threshold ({MED_THRESH})")
    axes[1].set_title("Churn Probability Distribution", fontweight="bold")
    axes[1].set_xlabel("Churn Probability")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=9)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ALERT SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📧 Alert Simulator":
    st.title("Automated Alert Workflow Simulator")
    st.markdown("---")
    st.markdown("Configure thresholds and simulate what alerts would fire for at-risk customers.")

    col1, col2 = st.columns(2)
    high_thresh = col1.slider("HIGH Risk Threshold", 0.50, 0.95, 0.75, 0.01,
                               help="Customers above this score trigger all 4 workflows")
    med_thresh  = col2.slider("MEDIUM Risk Threshold", 0.30, 0.74, 0.50, 0.01,
                               help="Customers above this score trigger Slack + Retention Email")

    # re-score with custom thresholds
    sim_df = scored_df.copy()
    sim_df["risk_sim"] = sim_df["churn_probability"].apply(
        lambda s: "HIGH" if s >= high_thresh else "MEDIUM" if s >= med_thresh else "LOW"
    )
    high_sim   = sim_df[sim_df["risk_sim"] == "HIGH"]
    medium_sim = sim_df[sim_df["risk_sim"] == "MEDIUM"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HIGH Risk Customers",   f"{len(high_sim):,}")
    c2.metric("MEDIUM Risk Customers", f"{len(medium_sim):,}")
    c3.metric("CS Team Emails",        f"{len(high_sim):,}")
    c4.metric("Retention Emails",      f"{len(high_sim) + len(medium_sim):,}")

    st.markdown("---")
    st.subheader("Workflow Summary")

    wf_data = {
        "Workflow":       ["CS Team Email Alert","Slack Notification","Retention Email to Customer","HubSpot CRM Task"],
        "Triggers For":   ["HIGH only","HIGH + MEDIUM","HIGH + MEDIUM","HIGH only"],
        "Customers":      [len(high_sim), len(high_sim)+len(medium_sim), len(high_sim)+len(medium_sim), len(high_sim)],
        "Status":         ["🟢 Would Send","🟢 Would Send","🟢 Would Send","🟢 Would Send"],
    }
    st.dataframe(pd.DataFrame(wf_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Sample Alert Preview")

    top5 = high_sim.sort_values("churn_probability", ascending=False).head(5)
    for _, row in top5.iterrows():
        name  = row.get("Name","Unknown")
        score = row["churn_score_pct"]
        freq  = row["Login_Frequency"]
        sent  = round(row["sentiment"], 2)
        ticket = str(row.get("Last_Support_Ticket",""))[:90]

        with st.expander(f"🔴 {name}  —  Churn Score: {score}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Login Frequency", freq)
            c2.metric("Daily Usage",     f"{row['Daily_Usage_Mins']} min")
            c3.metric("Sentiment Score", f"{sent:+.2f}")

            st.markdown("**Last Support Ticket:**")
            st.info(f'"{ticket}..."')

            st.markdown("**Workflows that would fire:**")
            st.success("✅ CS Team Email Alert")
            st.success("✅ Slack #customer-success Notification")
            st.success("✅ Personalised Retention Email (with STAYWITHUS20 discount)")
            st.success("✅ HubSpot CRM Follow-up Task")

            first = name.split()[0]
            if freq == "Rarely":
                msg = f"Hi {first}, we noticed you haven't logged in recently. We'd love to help! Here's 20% off: STAYWITHUS20"
            elif freq == "Weekly":
                msg = f"Hi {first}, we want to help you get more value from the platform. Here are some tips — and 20% off: STAYWITHUS20"
            else:
                msg = f"Hi {first}, we want to make sure you have everything you need. As a token of appreciation: STAYWITHUS20"

            st.markdown("**Retention email preview:**")
            st.code(msg)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT SINGLE CUSTOMER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict Single Customer":
    st.title("Predict Churn for a Single Customer")
    st.markdown("Enter customer details below to get an instant churn probability score.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        name         = st.text_input("Customer Name", value="Sudipta Sarkar")
        login_freq   = st.selectbox("Login Frequency", ["Daily","Weekly","Rarely"])
        account_age  = st.slider("Account Age (days)", 1, 1825, 365)

    with col2:
        daily_usage  = st.slider("Daily Usage (minutes)", 1, 120, 30)
        ticket_text  = st.text_area("Last Support Ticket Text",
            value="The UI is confusing and I cannot find the export button.",
            height=100)

    if st.button("🔮 Predict Churn Probability", type="primary"):
        sentiment_score = TextBlob(ticket_text).sentiment.polarity
        freq_enc        = {"Daily": 0, "Rarely": 1, "Weekly": 2}[login_freq]
        features_input  = [[account_age, daily_usage, freq_enc, sentiment_score]]

        prob   = best_model.predict_proba(features_input)[0][1]
        risk   = "HIGH" if prob >= HIGH_THRESH else "MEDIUM" if prob >= MED_THRESH else "LOW"

        st.markdown("---")
        st.subheader(f"Results for: {name}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Churn Probability", f"{prob*100:.1f}%")
        c2.metric("Sentiment Score",   f"{sentiment_score:+.3f}")
        c3.metric("Risk Level",        risk)
        c4.metric("Account Age",       f"{account_age} days")

        if risk == "HIGH":
            st.error(f"🔴 **HIGH RISK** — {name} is very likely to churn. Immediate intervention recommended.")
            st.markdown("**Workflows that would fire automatically:**")
            st.markdown("- ✅ CS Team Email Alert\n- ✅ Slack Notification\n- ✅ Retention Email with STAYWITHUS20\n- ✅ HubSpot CRM Task")
        elif risk == "MEDIUM":
            st.warning(f"🟡 **MEDIUM RISK** — {name} shows some churn signals. Monitor and engage.")
            st.markdown("**Workflows that would fire automatically:**")
            st.markdown("- ✅ Slack Notification\n- ✅ Retention Email with STAYWITHUS20")
        else:
            st.success(f"🟢 **LOW RISK** — {name} appears likely to remain a customer.")

        # probability gauge
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh(["Churn Probability"], [prob], color="#E63946" if risk=="HIGH" else "#F4A261" if risk=="MEDIUM" else "#2A9D8F",
                height=0.5)
        ax.barh(["Churn Probability"], [1-prob], left=[prob], color="#E2E8F0", height=0.5)
        ax.axvline(HIGH_THRESH, color="#E63946", linestyle="--", alpha=0.7)
        ax.axvline(MED_THRESH,  color="#F4A261", linestyle="--", alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.text(prob/2, 0, f"{prob*100:.1f}%", ha="center", va="center",
                color="white", fontweight="bold", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
