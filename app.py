import streamlit as st
import pandas as pd
from data_utils import EXPECTED_COLUMNS, load_dataset, validate_dataset, preprocess_data, create_features
from ml_model import train_ml_model
from insights import generate_insights, generate_recommendations, extract_sms_transaction

st.set_page_config(page_title="SmartSpend AI", page_icon="💰", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-title {font-size:42px;font-weight:800;margin-bottom:5px;}
.subtitle {font-size:18px;color:#666;margin-bottom:25px;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("💰 SmartSpend AI")
st.sidebar.caption("Student Expense Analytics & Savings Assistant")
page = st.sidebar.radio("Navigation", [
    "🏠 Home", "📊 Dashboard", "📥 Transactions", "🧹 Data Cleaning",
    "📈 EDA", "🔧 Feature Engineering", "🤖 Machine Learning",
    "💡 AI Insights", "🎯 Saving Planner", "🔬 Data Science Process", "🔐 Privacy"
])
st.sidebar.markdown("---")
st.sidebar.write("**Dataset**")
st.sidebar.write("SmartSpendAI_cleaned_dataset.csv")
st.sidebar.caption("B.Sc. Data Science Academic Project")

try:
    raw_df = load_dataset()
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.info("Place SmartSpendAI_cleaned_dataset.csv inside the data/ folder.")
    st.stop()

missing = validate_dataset(raw_df)
if missing:
    st.error(f"Dataset is missing required columns: {', '.join(missing)}")
    st.stop()

df = preprocess_data(raw_df)
if df.empty:
    st.error("No valid transactions remain after preprocessing.")
    st.stop()
feature_df = create_features(df)

if page == "🏠 Home":
    st.markdown('<div class="main-title">💰 SmartSpend AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Intelligent Student Expense Analytics & Savings Assistant</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Total Spending", f"₹{df['amount'].sum():,.0f}")
    c3.metric("Categories", int(df["category"].nunique()))
    c4.metric("Cities", int(df["city"].nunique()))
    st.markdown("---")
    st.subheader("🎯 Project Objective")
    st.write("SmartSpend AI analyzes student transaction data, identifies spending patterns, and provides practical saving recommendations.")
    st.subheader("🔄 Workflow")
    workflow = ["Data Collection", "Data Cleaning", "Preprocessing", "EDA", "Feature Engineering",
                "Train-Test Split", "Machine Learning", "Model Evaluation", "AI Insights", "Saving Recommendations", "Dashboard"]
    for start in range(0, len(workflow), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = start + j
            if idx < len(workflow):
                col.info(f"**{idx + 1}.** {workflow[idx]}")
    st.subheader("✨ Main Features")
    a, b, c = st.columns(3)
    a.write("📥 **Transaction Management**\n\nDataset viewing, manual entry, and SMS extraction.")
    b.write("📊 **Expense Analytics**\n\nCategory, monthly, city, and payment-mode analysis.")
    c.write("🤖 **Machine Learning**\n\nRandom Forest transaction-category classification.")

elif page == "📊 Dashboard":
    st.title("📊 Expense Dashboard")
    total, average, maximum = df["amount"].sum(), df["amount"].mean(), df["amount"].max()
    success_rate = df["transaction_status"].astype(str).str.lower().eq("success").mean() * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Spending", f"₹{total:,.0f}")
    c2.metric("Average Transaction", f"₹{average:,.0f}")
    c3.metric("Highest Transaction", f"₹{maximum:,.0f}")
    c4.metric("Success Rate", f"{success_rate:.1f}%")
    left, right = st.columns(2)
    with left:
        st.subheader("💰 Spending by Category")
        st.bar_chart(df.groupby("category")["amount"].sum().sort_values(ascending=False))
    with right:
        st.subheader("💳 Payment Mode")
        st.bar_chart(df["payment_mode"].value_counts())
    st.subheader("📅 Monthly Spending")
    monthly = df.groupby(df["date"].dt.to_period("M"))["amount"].sum()
    monthly.index = monthly.index.astype(str)
    st.line_chart(monthly)
    st.subheader("🏙️ City-wise Spending")
    st.bar_chart(df.groupby("city")["amount"].sum().sort_values(ascending=False))
    st.subheader("📌 Transaction Status")
    st.bar_chart(df["transaction_status"].value_counts())

elif page == "📥 Transactions":
    st.title("📥 Transaction Management")
    tab1, tab2, tab3 = st.tabs(["📄 Dataset", "➕ Add Transaction", "📱 SMS Extraction"])
    with tab1:
        st.dataframe(df, use_container_width=True, height=450)
        st.download_button("⬇️ Download Dataset", df.to_csv(index=False).encode("utf-8"),
                           "SmartSpendAI_cleaned_dataset.csv", "text/csv")
    with tab2:
        with st.form("add_transaction_form"):
            col1, col2 = st.columns(2)
            payer = col1.text_input("Payer Name", "Student")
            payee = col2.text_input("Payee Name")
            tx_date = col1.date_input("Date")
            tx_time = col2.time_input("Time")
            amount = col1.number_input("Amount (₹)", min_value=0.0, step=10.0)
            category = col2.selectbox("Category", sorted(df["category"].dropna().unique().tolist()) + ["Others"])
            payment_mode = col1.selectbox("Payment Mode", sorted(df["payment_mode"].dropna().astype(str).unique()))
            city = col2.text_input("City")
            status = st.selectbox("Transaction Status", ["Success", "Failed", "Pending"])
            submitted = st.form_submit_button("Add transaction")
        if submitted:
            new_row = {col: None for col in EXPECTED_COLUMNS}
            new_row.update({"transaction_id": f"MANUAL-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
                            "payer_name": payer, "payee_name": payee, "date": str(tx_date),
                            "time": tx_time.strftime("%H:%M"), "amount": amount, "category": category,
                            "payment_mode": payment_mode, "city": city, "transaction_status": status})
            updated = pd.concat([raw_df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state["updated_df"] = updated
            st.success("Transaction added for this session. Download the updated dataset from the Dataset tab.")
            st.download_button("Download updated CSV", updated.to_csv(index=False).encode("utf-8"),
                               "SmartSpendAI_updated_dataset.csv", "text/csv")
    with tab3:
        sms = st.text_area("Paste bank/payment SMS")
        if st.button("Extract transaction"):
            result = extract_sms_transaction(sms)
            st.json(result)
            if result["amount"] is None:
                st.warning("Could not detect an amount. Please check the SMS format.")

elif page == "🧹 Data Cleaning":
    st.title("🧹 Data Cleaning")
    st.write("Raw dataset preview")
    st.dataframe(raw_df.head(20), use_container_width=True)
    st.write("Missing values before preprocessing")
    st.dataframe(raw_df.isna().sum().rename("Missing values"))
    st.write(f"Rows before cleaning: {len(raw_df):,}")
    st.write(f"Rows after cleaning: {len(df):,}")
    st.write(f"Rows removed: {len(raw_df) - len(df):,}")
    st.dataframe(df.head(20), use_container_width=True)

elif page == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Amount distribution")
        st.histogram if False else None
        st.bar_chart(df["amount"].round(-1).value_counts().sort_index().head(40))
    with col2:
        st.subheader("Transactions by category")
        st.bar_chart(df["category"].value_counts())
    st.subheader("Category summary")
    st.dataframe(df.groupby("category")["amount"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False))

elif page == "🔧 Feature Engineering":
    st.title("🔧 Feature Engineering")
    st.write("Derived fields include month, day, weekday, hour, weekend flag, and spending level.")
    st.dataframe(feature_df.head(100), use_container_width=True)
    st.download_button("Download feature dataset", feature_df.to_csv(index=False).encode("utf-8"),
                       "SmartSpendAI_features.csv", "text/csv")

elif page == "🤖 Machine Learning":
    st.title("🤖 Random Forest Classification")
    if feature_df["category"].nunique() < 2:
        st.warning("At least two categories are needed to train a classifier.")
    else:
        try:
            results = train_ml_model(feature_df)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{results['accuracy']:.3f}")
            c2.metric("Precision", f"{results['precision']:.3f}")
            c3.metric("Recall", f"{results['recall']:.3f}")
            c4.metric("F1 Score", f"{results['f1']:.3f}")
            st.subheader("Classification Report")
            st.code(results["classification_report"])
            st.subheader("Confusion Matrix")
            st.dataframe(pd.DataFrame(results["confusion_matrix"],
                                      index=results["class_names"], columns=results["class_names"]))
        except Exception as exc:
            st.error(f"Model training failed: {exc}")
            st.info("Check that each category has enough rows for a stratified train/test split.")

elif page == "💡 AI Insights":
    st.title("💡 AI Insights")
    for insight in generate_insights(feature_df):
        st.info(insight)

elif page == "🎯 Saving Planner":
    st.title("🎯 Saving Planner")
    monthly_budget = st.number_input("Monthly budget (₹)", min_value=0.0, value=10000.0, step=500.0)
    savings_goal = st.number_input("Monthly savings goal (₹)", min_value=0.0, value=2000.0, step=500.0)
    spend = float(df["amount"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Recorded spending", f"₹{spend:,.2f}")
    c2.metric("Budget remaining", f"₹{monthly_budget - spend:,.2f}")
    c3.metric("Savings goal", f"₹{savings_goal:,.2f}")
    if spend > monthly_budget:
        st.warning("Recorded spending exceeds the entered budget.")
    else:
        st.success("Recorded spending is within the entered budget.")
    for item in generate_recommendations(feature_df):
        st.write("• " + item)
    st.caption("Budget comparison uses all records in the loaded dataset; it is not automatically limited to one month.")

elif page == "🔬 Data Science Process":
    st.title("🔬 Data Science Process")
    steps = {
        "1. Data Collection": "Transaction records are loaded from a CSV dataset.",
        "2. Data Validation": "Required columns are checked before analysis.",
        "3. Data Cleaning": "Dates and amounts are converted; rows with invalid date, amount, or category are removed.",
        "4. EDA": "Spending is summarized by category, month, city, payment mode, and status.",
        "5. Feature Engineering": "Calendar and time-based features are created.",
        "6. Model Training": "A Random Forest classifier predicts transaction categories.",
        "7. Evaluation": "Accuracy, weighted precision/recall/F1, and confusion matrix are displayed.",
        "8. Insights": "Summary statistics are translated into readable insights.",
        "9. Saving Plan": "Users enter a budget and savings goal and review recommendations."
    }
    for heading, detail in steps.items():
        st.subheader(heading)
        st.write(detail)

elif page == "🔐 Privacy":
    st.title("🔐 Privacy")
    st.write("This app processes the CSV file locally in the Streamlit app session. Avoid uploading or sharing files containing sensitive banking details.")
    st.write("SMS extraction is rule-based and displays the parsed result; review it before using it.")
