# SaaS Customer Churn Prediction
### Using Machine Learning & Natural Language Processing

**NIT Tiruchirappalli — Department of Computer Science & Engineering**  
**Author:** Sudipta | **Guide:** Dr. S. Saroja

---

## 🚀 Live App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 📌 Project Overview

This project predicts customer churn for a SaaS platform by combining:
- **EDA** — Exploratory analysis of usage behaviour patterns
- **NLP** — TextBlob sentiment analysis on customer support tickets
- **ML** — 3 models: Logistic Regression, Random Forest, Gradient Boosting
- **Automation** — 4 alert workflows (Email, Slack, Retention Email, CRM)

**Best Model:** Logistic Regression — **83.9% accuracy** on 2,000 unseen customers  
**Baseline:** 65.8% (naive) → **+17.4 percentage points improvement**

---

## 📊 Key Findings

| Login Frequency | Churn Rate | Risk Level |
|---|---|---|
| Daily | 16.4% | 🟢 LOW |
| Weekly | 37.4% | 🟡 MEDIUM |
| Rarely | 82.7% | 🔴 HIGH |

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas, NumPy | Data processing |
| TextBlob | NLP sentiment analysis |
| scikit-learn | ML models |
| matplotlib, seaborn | Visualisation |
| Streamlit | Web app framework |

---

## 📂 Project Structure

```
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── churn_dataset.csv          # Training data (500 rows)
├── charn_dataset2.csv         # Test data (2,000 rows)
└── assets/                    # Chart images + NIT logo
    ├── nit_logo_rgb.png
    ├── step1_churn_distribution.png
    ├── step2_login_vs_churn.png
    ├── step3_usage_distributions.png
    ├── step4_sentiment_analysis.png
    ├── step5_model_comparison.png
    └── step6_confusion_matrix.png
```

---

## ▶️ Run Locally

```bash
# Clone the repo
git clone https://github.com/sudipta0008/saas-churn-prediction.git
cd saas-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📚 IEEE References

- Manzoor et al. (2024). *IEEE Access*, DOI: 10.1109/ACCESS.2024.3402092
- IEEE Xplore (2025). Document 10968946
- IEEE Xplore (2024). Document 10895079, 10592931, 10723880

---

## 📄 License

MIT License — feel free to use and adapt for your own projects.
