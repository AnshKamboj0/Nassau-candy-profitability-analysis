import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Nassau Candy | Profit Intelligence",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #08080C;
    color: #F5F5F7;
}

[data-testid="stSidebar"] {
    background: #111116;
    border-right: 1px solid #292333;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #F5F5F7;
}

.subtitle {
    color: #A5A5AF;
    font-size: 14px;
    margin-bottom: 25px;
}

.metric-card {
    background: linear-gradient(145deg, #191522, #111116);
    border: 1px solid #332747;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 12px;
}

.metric-label {
    color: #A5A5AF;
    font-size: 13px;
    font-weight: 500;
}

.metric-value {
    color: #A678FF;
    font-size: 25px;
    font-weight: 800;
    margin-top: 8px;
}

.section-title {
    color: #F5F5F7;
    font-size: 20px;
    font-weight: 700;
    margin-top: 18px;
}

div[data-testid="stMetric"] {
    background: #14121B;
    border: 1px solid #332747;
    padding: 17px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #A5A5AF;
}

div[data-testid="stMetricValue"] {
    color: #A678FF;
}

.stTabs [data-baseweb="tab"] {
    color: #A5A5AF;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #A678FF !important;
}

div.stButton > button {
    background: #8B4DFF;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

div.stButton > button:hover {
    background: #6935D3;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🍬 Nassau Candy</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">PROFIT INTELLIGENCE | SALES ANALYTICS & MACHINE LEARNING</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR - FILE UPLOAD
# --------------------------------------------------

st.sidebar.title("Dashboard Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload your Nassau Candy CSV",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Upload a CSV file to explore your business performance.")


if uploaded_file is None:
    st.info("Please upload your Nassau Candy Distributor CSV file using the sidebar.")
    st.markdown("### Required CSV columns")
    st.write(
        "Product Name, Division, Sales, Cost, Gross Profit and Units"
    )
    st.stop()


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:
    raw = pd.read_csv(uploaded_file)
    original_row_count = len(raw)
except Exception as error:
    st.error(f"Unable to read the CSV file: {error}")
    st.stop()


# --------------------------------------------------c
# VALIDATE DATA
# --------------------------------------------------

required_columns = [
    "Product Name",
    "Division",
    "Sales",
    "Cost",
    "Gross Profit",
    "Units"
]

missing_columns = [
    column for column in required_columns
    if column not in raw.columns
]

if missing_columns:
    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

raw = raw.copy()

numeric_columns = [
    "Sales",
    "Cost",
    "Gross Profit",
    "Units"
]

for column in numeric_columns:
    raw[column] = pd.to_numeric(
        raw[column],
        errors="coerce"
    )

raw["Product Name"] = raw["Product Name"].fillna("Unknown")
raw["Division"] = raw["Division"].fillna("Unknown")

duplicate_rows = int(raw.duplicated().sum())

raw = raw.dropna(subset=numeric_columns)

if raw.empty:
    st.error("No valid records found after data cleaning.")
    st.stop()

raw = raw.reset_index(drop=True)


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

raw["Gross Margin (%)"] = np.where(
    raw["Sales"] != 0,
    (raw["Gross Profit"] / raw["Sales"]) * 100,
    np.nan
)

raw["Profit per Unit"] = np.where(
    raw["Units"] != 0,
    raw["Gross Profit"] / raw["Units"],
    np.nan
)

raw["Cost Ratio (%)"] = np.where(
    raw["Sales"] != 0,
    (raw["Cost"] / raw["Sales"]) * 100,
    np.nan
)


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.markdown("### Filters")

divisions = sorted(
    raw["Division"].astype(str).unique().tolist()
)

selected_divisions = st.sidebar.multiselect(
    "Select Division",
    options=divisions,
    default=divisions
)

filtered = raw[
    raw["Division"].astype(str).isin(selected_divisions)
].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()


# --------------------------------------------------
# PRODUCT AND DIVISION AGGREGATION
# --------------------------------------------------

products = (
    filtered.groupby("Product Name", as_index=False)
    .agg({
        "Sales": "sum",
        "Cost": "sum",
        "Gross Profit": "sum",
        "Units": "sum"
    })
)

products["Gross Margin (%)"] = np.where(
    products["Sales"] != 0,
    products["Gross Profit"] / products["Sales"] * 100,
    np.nan
)

products["Profit per Unit"] = np.where(
    products["Units"] != 0,
    products["Gross Profit"] / products["Units"],
    np.nan
)

division_data = (
    filtered.groupby("Division", as_index=False)
    .agg({
        "Sales": "sum",
        "Cost": "sum",
        "Gross Profit": "sum",
        "Units": "sum"
    })
)

division_data["Gross Margin (%)"] = np.where(
    division_data["Sales"] != 0,
    division_data["Gross Profit"] / division_data["Sales"] * 100,
    np.nan
)


# --------------------------------------------------
# MAIN KPI SECTION
# --------------------------------------------------

total_sales = filtered["Sales"].sum()
total_cost = filtered["Cost"].sum()
total_profit = filtered["Gross Profit"].sum()
total_units = filtered["Units"].sum()

overall_margin = (
    total_profit / total_sales * 100
    if total_sales != 0 else 0
)

st.markdown("### Business Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Sales",
    f"${total_sales:,.0f}"
)

c2.metric(
    "Total Cost",
    f"${total_cost:,.0f}"
)

c3.metric(
    "Gross Profit",
    f"${total_profit:,.0f}"
)

c4.metric(
    "Gross Margin",
    f"{overall_margin:.2f}%"
)

c5, c6, c7, c8 = st.columns(4)

c5.metric("Total Units", f"{total_units:,.0f}")
c6.metric("Products", f"{products['Product Name'].nunique():,}")
c7.metric("Divisions", f"{division_data['Division'].nunique():,}")
c8.metric("Duplicate Rows", f"{duplicate_rows:,}")


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Product Analysis",
    "Division Performance",
    "Machine Learning",
    "Profit Concentration",
    "Geography"
])


# ==================================================
# TAB 1 - OVERVIEW
# ==================================================

with tab1:

    st.subheader("Sales and Profit Overview")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            division_data,
            x="Division",
            y="Sales",
            title="Sales by Division",
            color="Sales",
            color_continuous_scale="Purples"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111116",
            plot_bgcolor="#111116"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.pie(
            division_data,
            names="Division",
            values="Gross Profit",
            title="Profit Contribution by Division",
            hole=0.55
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111116"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sales vs Gross Profit")

    fig = px.scatter(
        products,
        x="Sales",
        y="Gross Profit",
        size="Units",
        color="Gross Margin (%)",
        hover_name="Product Name",
        title="Product Sales and Profit Relationship",
        color_continuous_scale="Purples"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111116",
        plot_bgcolor="#111116"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")

    st.dataframe(
        filtered.head(20),
        use_container_width=True
    )


# ==================================================
# TAB 2 - PRODUCT ANALYSIS
# ==================================================

with tab2:

    st.subheader("Product Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:

        top_products = products.nlargest(
            10,
            "Sales"
        )

        fig = px.bar(
            top_products,
            x="Sales",
            y="Product Name",
            orientation="h",
            title="Top 10 Products by Sales",
            color="Sales",
            color_continuous_scale="Purples"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111116",
            plot_bgcolor="#111116",
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        profitable_products = products[
            products["Gross Margin (%)"].notna()
        ].nlargest(
            10,
            "Gross Margin (%)"
        )

        fig = px.bar(
            profitable_products,
            x="Gross Margin (%)",
            y="Product Name",
            orientation="h",
            title="Top 10 Products by Gross Margin",
            color="Gross Margin (%)",
            color_continuous_scale="Purples"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111116",
            plot_bgcolor="#111116",
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Product Performance Table")

    st.dataframe(
        products.sort_values(
            "Gross Profit",
            ascending=False
        ),
        use_container_width=True
    )


# ==================================================
# TAB 3 - DIVISION PERFORMANCE
# ==================================================

with tab3:

    st.subheader("Division Performance Analysis")

    fig = px.bar(
        division_data,
        x="Division",
        y=["Sales", "Cost", "Gross Profit"],
        barmode="group",
        title="Sales, Cost and Profit by Division"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111116",
        plot_bgcolor="#111116"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        division_data,
        x="Division",
        y="Gross Margin (%)",
        title="Gross Margin by Division",
        color="Gross Margin (%)",
        color_continuous_scale="Purples"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111116",
        plot_bgcolor="#111116"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        division_data.sort_values(
            "Gross Profit",
            ascending=False
        ),
        use_container_width=True
    )


# ==================================================
# TAB 4 - MACHINE LEARNING
# ==================================================

with tab4:

    st.subheader("Machine Learning Insights")

    ml_data = products[
        ["Sales", "Cost", "Gross Profit", "Units", "Gross Margin (%)"]
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(ml_data) < 2:

        st.warning(
            "At least two products with valid numeric values are required for machine learning."
        )

    else:

        # ------------------------------------------
        # K-MEANS CLUSTERING
        # ------------------------------------------

        st.markdown("### 1. K-Means Product Clustering")

        number_of_clusters = st.slider(
            "Select Number of Clusters",
            min_value=2,
            max_value=min(4, len(ml_data)),
            value=min(3, len(ml_data))
        )

        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(ml_data)

        kmeans = KMeans(
            n_clusters=number_of_clusters,
            random_state=42,
            n_init=10
        )

        cluster_labels = kmeans.fit_predict(
            scaled_data
        )

        cluster_data = products.loc[
            ml_data.index
        ].copy()

        cluster_data["Cluster"] = cluster_labels.astype(str)

        fig = px.scatter(
            cluster_data,
            x="Sales",
            y="Gross Profit",
            color="Cluster",
            size="Units",
            hover_name="Product Name",
            title="Product Segmentation using K-Means"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111116",
            plot_bgcolor="#111116"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write(
            "Products are grouped using sales, cost, gross profit, units and gross margin."
        )

        st.dataframe(
            cluster_data,
            use_container_width=True
        )

        # ------------------------------------------
        # ISOLATION FOREST
        # ------------------------------------------

        st.markdown("### 2. Isolation Forest Anomaly Detection")

        if len(ml_data) >= 5:

            isolation_model = IsolationForest(
                contamination=0.05,
                random_state=42
            )

            anomaly_predictions = isolation_model.fit_predict(
                scaled_data
            )

            anomaly_data = products.loc[
                ml_data.index
            ].copy()

            anomaly_data["Anomaly Status"] = np.where(
                anomaly_predictions == -1,
                "Anomaly",
                "Normal"
            )

            anomaly_count = int(
                (anomaly_predictions == -1).sum()
            )

            normal_count = int(
                (anomaly_predictions == 1).sum()
            )

            a1, a2 = st.columns(2)

            a1.metric("Normal Products", normal_count)
            a2.metric("Anomalies Detected", anomaly_count)

            fig = px.scatter(
                anomaly_data,
                x="Sales",
                y="Gross Profit",
                color="Anomaly Status",
                hover_name="Product Name",
                title="Anomaly Detection Results"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#111116",
                plot_bgcolor="#111116"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                anomaly_data,
                use_container_width=True
            )

        else:

            st.info(
                "At least five valid products are required for anomaly detection."
            )


# ==================================================
# TAB 5 - PROFIT CONCENTRATION
# ==================================================

with tab5:

    st.subheader("Pareto Profit Concentration Analysis")

    pareto_data = products.sort_values(
        "Gross Profit",
        ascending=False
    ).copy()

    total_pareto_profit = pareto_data["Gross Profit"].sum()

    if (
        len(pareto_data) > 0
        and total_pareto_profit > 0
        and (pareto_data["Gross Profit"] >= 0).all()
    ):

        pareto_data["Profit Contribution (%)"] = (
            pareto_data["Gross Profit"]
            / total_pareto_profit
        ) * 100

        pareto_data["Cumulative Profit (%)"] = (
            pareto_data["Profit Contribution (%)"].cumsum()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=pareto_data["Product Name"],
                y=pareto_data["Gross Profit"],
                name="Gross Profit"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pareto_data["Product Name"],
                y=pareto_data["Cumulative Profit (%)"],
                name="Cumulative Profit %",
                yaxis="y2",
                mode="lines+markers"
            )
        )

        fig.update_layout(
            title="Product Profit Pareto Chart",
            template="plotly_dark",
            paper_bgcolor="#111116",
            plot_bgcolor="#111116",
            yaxis=dict(title="Gross Profit"),
            yaxis2=dict(
                title="Cumulative Profit (%)",
                overlaying="y",
                side="right",
                range=[0, 110]
            ),
            xaxis_tickangle=-45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        products_for_80 = int(
            (pareto_data["Cumulative Profit (%)"] < 80).sum() + 1
        )

        st.metric(
            "Products Needed to Reach Approximately 80% of Profit",
            products_for_80
        )

        st.dataframe(
            pareto_data,
            use_container_width=True
        )

    else:

        st.warning(
            "Pareto analysis requires positive total gross profit and no negative product profit values."
        )


# ==================================================
# TAB 6 - GEOGRAPHY
# ==================================================

with tab6:

    st.subheader("Geography and Regional Performance")

    possible_state_columns = [
        "State/Province",
        "State",
        "Province"
    ]

    state_column = next(
        (
            column for column in possible_state_columns
            if column in filtered.columns
        ),
        None
    )

    if state_column is not None:

        geography_data = (
            filtered.groupby(state_column, as_index=False)
            .agg({
                "Sales": "sum",
                "Gross Profit": "sum",
                "Units": "sum"
            })
        )

        geography_data = geography_data.sort_values(
            "Sales",
            ascending=False
        )

        fig = px.bar(
            geography_data.head(15),
            x=state_column,
            y="Sales",
            color="Gross Profit",
            title="Top 15 Regions by Sales",
            color_continuous_scale="Purples"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111116",
            plot_bgcolor="#111116"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            geography_data,
            use_container_width=True
        )

    else:

        st.info(
            "Your CSV does not contain a State/Province column. "
            "Upload a dataset with regional information to enable geography analysis."
        )

    st.markdown("### Data Quality Summary")

    quality_data = pd.DataFrame({
        "Metric": [
            "Total uploaded rows",
            "Rows after cleaning",
            "Duplicate rows",
            "Missing values after cleaning"
        ],
        "Value": [
            original_row_count,
            len(raw),
            duplicate_rows,
            int(raw.isna().sum().sum())
        ]
    })

    st.dataframe(
        quality_data,
        use_container_width=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    "<div style='text-align:center;color:#A5A5AF;'>"
    "Nassau Candy | Profit Intelligence Dashboard"
    "<br>Built with Python, Streamlit, Pandas, Plotly and Scikit-learn"
    "</div>",
    unsafe_allow_html=True
)
