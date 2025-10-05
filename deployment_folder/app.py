import streamlit as st
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_PATH = 'lgbm_model_91day.pkl'
SCALER_PATH = 'feature_scaler.pkl'
FEATURES_PATH = 'model_features.pkl'
RMSE = 0.3140

@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        try:
            feature_names = joblib.load(FEATURES_PATH)
        except:
            feature_names = [
                'cbk_rate', 'domestic_debt', 'external_debt',
                'exchange_rate', 'total_debt', 'annual_inflation', 'weighted_average_rate'
            ]
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"⚠️ Error loading assets: {e}")
        return None, None, None

def interpret_rate(rate):
    if rate >= 12:
        return ("High Investment Appeal",
                "High returns may signal tight liquidity or borrowing pressure.",
                "success", "#38a700")
    elif rate >= 8:
        return ("Moderate & Stable Return",
                "Balanced conditions — investor confidence and stability.",
                "warning", "#ffa500")
    else:
        return ("Conservative Return",
                "Lower yields — likely due to high liquidity or easing stance.",
                "error", "#e80000")


st.set_page_config(page_title="Kenya 91-Day T-Bill Forecast Dashboard", layout="wide")

st.title("🇰🇪 91-Day Treasury Bill Forecasting Dashboard")
st.caption("Explore how macroeconomic indicators influence Kenya’s short-term T-Bill rate.")

# Loading the model
model, scaler, feature_names = load_model_assets()

if model and scaler:
    st.sidebar.header("📊 Adjust Economic Indicators")

  # User Inputs
    inputs = {
        "cbk_rate": st.sidebar.slider("CBK Policy Rate (%)", 6.0, 16.0, 12.5, 0.1),
        "domestic_debt": st.sidebar.slider("Domestic Debt (KES Trillions)", 2.0, 6.0, 4.2, 0.1),
        "external_debt": st.sidebar.slider("External Debt (USD Trillions)", 0.02, 0.05, 0.037, 0.001),
        "exchange_rate": st.sidebar.slider("Exchange Rate (KES/USD)", 100.0, 170.0, 153.0, 0.5),
        "total_debt": st.sidebar.slider("Total Public Debt (KES Trillions)", 7.0, 12.0, 9.8, 0.1),
        "annual_inflation": st.sidebar.slider("Annual Inflation (%)", 2.0, 15.0, 6.8, 0.1),
        "weighted_average_rate": st.sidebar.slider("Weighted Average Rate (%)", 8.0, 16.0, 12.2, 0.1)
    }

    X_input = np.array(list(inputs.values())).reshape(1, -1)
    X_scaled = scaler.transform(X_input)
    predicted_rate = float(model.predict(X_scaled)[0])

    title, interpretation, alert_type, color = interpret_rate(predicted_rate)

    # Tracking the scenario trend
    if "trend_data" not in st.session_state:
        st.session_state["trend_data"] = pd.DataFrame(columns=["Step", "Predicted Rate"])
    new_row = pd.DataFrame({"Step": [len(st.session_state["trend_data"]) + 1],
                            "Predicted Rate": [predicted_rate]})
    st.session_state["trend_data"] = pd.concat([st.session_state["trend_data"], new_row], ignore_index=True)

    col1, col2 = st.columns([0.55, 0.45])

    with col1:
        st.subheader("Predicted 91-Day Treasury Bill Rate")
        st.metric(label="Predicted Rate (%)", value=f"{predicted_rate:.2f}", delta=f"±{RMSE:.3f} RMSE")

        if alert_type == "success":
            st.success(f"### {title}\n{interpretation}")
        elif alert_type == "warning":
            st.warning(f"### {title}\n{interpretation}")
        else:
            st.error(f"### {title}\n{interpretation}")

    with col2:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_rate,
            title={'text': "91-Day T-Bill Rate (%)"},
            gauge={
                'axis': {'range': [0, 15]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 8], 'color': "#ffcccc"},
                    {'range': [8, 12], 'color': "#ffe699"},
                    {'range': [12, 15], 'color': "#ccffcc"}
                ]
            }
        ))
        st.plotly_chart(gauge, use_container_width=True)

# plotting the trend of predictions
    st.markdown("### 📈 Trend of Predicted Rates (Scenario Steps)")
    fig_trend = px.line(st.session_state["trend_data"], x="Step", y="Predicted Rate",
                        title="Predicted 91-Day T-Bill Trend", markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    # Plotting Feature Importance
    st.markdown("### 🔍 Model Feature Importance")
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    fig_imp = px.bar(importance, x="Feature", y="Importance", color="Importance",
                     text_auto=True, title="Feature Influence on Predictions")
    st.plotly_chart(fig_imp, use_container_width=True)

    # Statistical Visualizations Section
    st.markdown("### 📊 Statistical Visualizations")

    # Create data frame for correlation
    df_inputs = pd.DataFrame([inputs])
    df_inputs["Predicted Rate"] = predicted_rate

    tab1, tab2, tab3 = st.tabs(["Correlation Heatmap", "Distribution", "Pairwise Scatter"])

    with tab1:
        st.write("#### 🔗 Correlation between Inputs and Predicted Rate")
        corr = df_inputs.corr(numeric_only=True)
        fig_corr, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig_corr)

    with tab2:
        st.write("#### 📉 Distribution of Predicted T-Bill Rates")
        if len(st.session_state["trend_data"]) > 2:
            fig_dist = px.histogram(st.session_state["trend_data"], x="Predicted Rate",
                                    nbins=10, title="Distribution of Predicted Rates (across scenarios)")
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Run more scenarios to see the distribution histogram.")
   
    # --- Scenario Summary ---
    st.markdown("### 🧾 Scenario Summary")
    df_summary = pd.DataFrame(list(inputs.items()), columns=["Indicator", "Value"])
    df_summary.loc[len(df_summary)] = ["Predicted 91-Day T-Bill Rate (%)", f"{predicted_rate:.2f}"]
    st.dataframe(df_summary, use_container_width=True)

    if st.button("🔄 Reset Trend Data"):
        st.session_state["trend_data"] = pd.DataFrame(columns=["Step", "Predicted Rate"])
        st.success("Trend data cleared successfully!")

else:
    st.error("⚠️ Model or scaler files not found. Ensure `.pkl` files are available in the app directory.")
