"""
Prédiction du churn client (Telco) — application Streamlit.

Charge le pipeline entraîné (voir src/train_model.py), prédit la probabilité de
résiliation d'un client, met en évidence les facteurs de risque et propose une
action de rétention.

Lancement : streamlit run streamlit_app.py
"""

import os
import json

import pandas as pd
import joblib
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")

st.set_page_config(page_title="Prédiction de churn", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "churn_model.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    metrics = {}
    mpath = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            metrics = json.load(f)
    return model, schema, metrics


model, SCHEMA, METRICS = load_artifacts()
THRESHOLD = SCHEMA.get("threshold", 0.5)
best_auc = METRICS.get("comparison", {}).get(METRICS.get("best_model", ""), {}).get("roc_auc", 0.847)

st.title("📉 Prédiction du churn client")
st.caption(
    f"Estime la probabilité qu'un client résilie son abonnement télécom. "
    f"Modèle : **XGBoost tuné** · ROC-AUC **{best_auc:.2f}** · seuil de décision **{THRESHOLD:.2f}**."
)

tab_pred, tab_perf = st.tabs(["🔮 Prédiction", "📊 Performance du modèle"])

with tab_pred:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("👤 Profil")
        gender = st.selectbox("Genre", ["Female", "Male"])
        senior = st.toggle("Senior (65+)")
        partner = st.selectbox("En couple", ["No", "Yes"])
        dependents = st.selectbox("Personnes à charge", ["No", "Yes"])
        tenure = st.slider("Ancienneté (mois)", 0, 72, 12)

    with c2:
        st.subheader("🌐 Services")
        internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
        has_net = internet != "No"
        phone = st.selectbox("Téléphone", ["Yes", "No"])
        multi = st.selectbox("Lignes multiples", ["No", "Yes"]) if phone == "Yes" else "No phone service"

        def net_opt(label):
            return st.selectbox(label, ["No", "Yes"]) if has_net else "No internet service"

        online_sec = net_opt("Sécurité en ligne")
        online_bak = net_opt("Sauvegarde en ligne")
        device = net_opt("Protection appareil")
        tech = net_opt("Support technique")
        tv = net_opt("Streaming TV")
        movies = net_opt("Streaming films")

    with c3:
        st.subheader("💳 Contrat & facturation")
        contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Facture dématérialisée", ["Yes", "No"])
        payment = st.selectbox("Paiement", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.slider("Charge mensuelle (€)", 18.0, 120.0, 70.0, 0.5)
        total = st.number_input("Charges totales (€)", 0.0, 9000.0,
                                float(round(monthly * max(tenure, 1), 2)), step=10.0)

    predict = st.button("Évaluer le risque de churn", type="primary", use_container_width=True)

    if predict:
        client = {
            "gender": gender, "SeniorCitizen": int(senior), "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multi, "InternetService": internet, "OnlineSecurity": online_sec,
            "OnlineBackup": online_bak, "DeviceProtection": device, "TechSupport": tech,
            "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }
        X = pd.DataFrame([client])[SCHEMA["columns"]]
        proba = float(model.predict_proba(X)[0, 1])
        churn = proba >= THRESHOLD

        st.divider()
        r1, r2 = st.columns([1, 1.3])
        with r1:
            color = "#ef5350" if proba >= 0.6 else ("#ffb74d" if proba >= 0.35 else "#81c784")
            st.markdown(
                f"<div style='text-align:center'>"
                f"<div style='font-size:3.2rem;font-weight:800;color:{color}'>{proba:.0%}</div>"
                f"<div style='color:#8892b0'>probabilité de churn</div></div>",
                unsafe_allow_html=True)
            st.progress(proba)
            if churn:
                st.error(f"⚠️ Client À RISQUE (≥ seuil {THRESHOLD:.0%}) — action de rétention recommandée.")
            else:
                st.success(f"✅ Client peu susceptible de partir (< seuil {THRESHOLD:.0%}).")

        with r2:
            # Facteurs de risque détectés (règles métier interprétables)
            factors = []
            if contract == "Month-to-month":
                factors.append("Contrat au mois (sans engagement) → proposer un contrat 1 ou 2 ans avec avantage.")
            if tenure < 12:
                factors.append("Client récent (< 12 mois) → phase critique, soigner l'onboarding.")
            if internet == "Fiber optic":
                factors.append("Fibre optique → segment au churn élevé, vérifier la qualité/prix perçu.")
            if has_net and tech == "No":
                factors.append("Pas de support technique → proposer l'option (facteur protecteur fort).")
            if has_net and online_sec == "No":
                factors.append("Pas de sécurité en ligne → offre de sécurité en cross-sell.")
            if payment == "Electronic check":
                factors.append("Paiement par chèque électronique → orienter vers le prélèvement automatique.")
            if monthly > 85:
                factors.append("Charge mensuelle élevée → risque de sensibilité au prix.")

            st.markdown("**🎯 Facteurs de risque & rétention**")
            if factors:
                for f in factors:
                    st.markdown(f"- {f}")
            else:
                st.markdown("- Profil stable : peu de leviers de churn identifiés. 👍")

with tab_perf:
    st.subheader("Comparaison des modèles")
    comp = METRICS.get("comparison", {})
    if comp:
        rows = [{
            "Modèle": n.replace("_", " ").title(),
            "ROC-AUC": round(m["roc_auc"], 3),
            "Recall": round(m["recall"], 3),
            "Précision": round(m["precision"], 3),
            "F1": round(m["f1"], 3),
        } for n, m in sorted(comp.items(), key=lambda kv: -kv[1]["roc_auc"])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info(
        "En rétention, on privilégie le **recall** (détecter un maximum de churners). "
        "Le **SMOTE** a été testé mais n'améliore pas les résultats ici ; le déséquilibre "
        "est géré par pondération des classes. Le **seuil de décision** est ajustable selon "
        "le coût d'un client perdu vs. celui d'une offre de rétention."
    )

    col1, col2 = st.columns(2)
    for col, img, cap in [
        (col1, "roc_curves.png", "Courbes ROC"),
        (col2, "confusion_matrix.png", "Matrice de confusion (seuil optimal)"),
    ]:
        path = os.path.join(REPORTS_DIR, img)
        if os.path.exists(path):
            col.image(path, caption=cap, use_container_width=True)

st.divider()
st.caption("Alexis Clerc · [GitHub](https://github.com/2Alexis) · [Portfolio](https://alexis-clerc.fr)")
