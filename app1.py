import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.metrics import r2_score, mean_absolute_error

# Model Load karna
model = joblib.load("housing_price_gbmboost_pipeline.pkl")

st.set_page_config(
    page_title="Housing Price Predict Dashboard", page_icon="💰", layout="wide"
)

st.title("💰 Housing Price Predict & Valuation Dashboard")
st.write(
    "AI-Powered Real Estate Valuation Engine: Predict accurate property "
    "prices, analyze market trends, and audit portfolio performance instantly."
)

tab1, tab2 = st.tabs(["👤 Single Customer Audit", "📁 Bulk CSV Batch Audit"])

# ========== Helper Feature Engineering ==========
def engineer_features(df):
    df = df.copy()

    # Agar yeh Batch CSV hai, tabhi outlier filter chalayein
    if len(df) > 1:
        df = df[df["Gr Liv Area"] < 4000].reset_index(drop=True)

    df["MS SubClass"] = df["MS SubClass"].astype(str)

    df["House_Age"] = df["Yr Sold"] - df["Year Built"]
    df["Remod_Age"] = df["Yr Sold"] - df["Year Remod/Add"]
    df["Is_Remodeled"] = (df["Year Remod/Add"] != df["Year Built"]).astype(int)

    df["Total_SF"] = (
        df["1st Flr SF"].fillna(0)
        + df["2nd Flr SF"].fillna(0)
        + df["Total Bsmt SF"].fillna(0)
    )
    df["Area_x_Quality"] = df["Total_SF"] * df["Overall Qual"]

    df["Total_Baths"] = (
        df["Full Bath"].fillna(0)
        + (0.5 * df["Half Bath"].fillna(0))
        + df["Bsmt Full Bath"].fillna(0)
        + (0.5 * df["Bsmt Half Bath"].fillna(0))
    )

    df["Total_Porch_SF"] = (
        df["Wood Deck SF"].fillna(0)
        + df["Open Porch SF"].fillna(0)
        + df["Enclosed Porch"].fillna(0)
        + df["3Ssn Porch"].fillna(0)
        + df["Screen Porch"].fillna(0)
    )

    df["Has_Garage"] = (df["Garage Area"].fillna(0) > 0).astype(int)
    df["Has_Bsmt"] = (df["Total Bsmt SF"].fillna(0) > 0).astype(int)
    df["Has_Fireplace"] = (df["Fireplaces"].fillna(0) > 0).astype(int)

    return df


# ==================== TAB 1: SINGLE CUSTOMER AUDIT ====================
with tab1:
    st.sidebar.header("Customer Detail Input")
    
    # 1. NUMERICAL FEATURES
    st.sidebar.subheader("Numerical Inputs")
    ms_subclass = st.sidebar.number_input("MS SubClass", min_value=20, max_value=190, value=20, step=1) 
    lot_frontage = st.sidebar.number_input("Lot Frontage (Feet)", min_value=21.0, max_value=313.0, value=70.0, step=1.0)
    lot_area = st.sidebar.number_input("Lot Area (Sq Ft)", min_value=1300, max_value=215245, value=9500, step=100)
    overall_qual = st.sidebar.number_input("Overall Quality (1-10)", min_value=1, max_value=10, value=5, step=1)
    overall_cond = st.sidebar.number_input("Overall Condition (1-9)", min_value=1, max_value=9, value=5, step=1)
    year_built = st.sidebar.number_input("Year Built", min_value=1872, max_value=2010, value=2000, step=1)
    year_remod_add = st.sidebar.number_input("Year Remodeled", min_value=1950, max_value=2010, value=2000, step=1)
    garage_yr_blt = st.sidebar.number_input("Garage Year Built", min_value=1895, max_value=2207, value=2000, step=1)
    mas_vnr_area = st.sidebar.number_input("Mas Vnr Area (Sq Ft)", min_value=0.0, max_value=1600.0, value=0.0, step=10.0)
    bsmtfin_sf_1 = st.sidebar.number_input("BsmtFin SF 1", min_value=0.0, max_value=5644.0, value=0.0, step=10.0)
    bsmtfin_sf_2 = st.sidebar.number_input("BsmtFin SF 2", min_value=0.0, max_value=1526.0, value=0.0, step=10.0)
    bsmt_unf_sf = st.sidebar.number_input("Bsmt Unf SF", min_value=0.0, max_value=2336.0, value=0.0, step=10.0)
    total_bsmt_sf = st.sidebar.number_input("Total Bsmt SF", min_value=0.0, max_value=6110.0, value=500.0, step=10.0)
    first_flr_sf = st.sidebar.number_input("1st Flr SF", min_value=334, max_value=5095, value=1000, step=10)
    second_flr_sf = st.sidebar.number_input("2nd Flr SF", min_value=0, max_value=2065, value=0, step=10)
    low_qual_fin_sf = st.sidebar.number_input("Low Qual Fin SF", min_value=0, max_value=1064, value=0, step=10)
    gr_liv_area = st.sidebar.number_input("Gr Liv Area (Above Ground)", min_value=334, max_value=5642, value=1500, step=10)
    bsmt_full_bath = st.sidebar.number_input("Bsmt Full Bath", min_value=0.0, max_value=3.0, value=0.0, step=1.0)
    bsmt_half_bath = st.sidebar.number_input("Bsmt Half Bath", min_value=0.0, max_value=2.0, value=0.0, step=1.0)
    full_bath = st.sidebar.number_input("Full Bath", min_value=0, max_value=4, value=2, step=1)
    half_bath = st.sidebar.number_input("Half Bath", min_value=0, max_value=2, value=0, step=1)
    bedroom_abv_gr = st.sidebar.number_input("Bedrooms Above Ground", min_value=0, max_value=8, value=3, step=1)
    kitchen_abv_gr = st.sidebar.number_input("Kitchens Above Ground", min_value=0, max_value=3, value=1, step=1)
    tot_rms_abv_grd = st.sidebar.number_input("Total Rooms Above Ground", min_value=2, max_value=15, value=6, step=1)
    fireplaces = st.sidebar.number_input("Fireplaces", min_value=0, max_value=4, value=0, step=1)
    garage_cars = st.sidebar.number_input("Garage Cars", min_value=0.0, max_value=5.0, value=2.0, step=1.0)
    garage_area = st.sidebar.number_input("Garage Area (Sq Ft)", min_value=0.0, max_value=1488.0, value=400.0, step=10.0)
    wood_deck_sf = st.sidebar.number_input("Wood Deck SF", min_value=0, max_value=1424, value=0, step=10)
    open_porch_sf = st.sidebar.number_input("Open Porch SF", min_value=0, max_value=742, value=0, step=10)
    enclosed_porch = st.sidebar.number_input("Enclosed Porch", min_value=0, max_value=1012, value=0, step=10)
    three_ssn_porch = st.sidebar.number_input("3Ssn Porch", min_value=0, max_value=508, value=0, step=10)
    screen_porch = st.sidebar.number_input("Screen Porch", min_value=0, max_value=576, value=0, step=10)
    pool_area = st.sidebar.number_input("Pool Area", min_value=0, max_value=800, value=0, step=10)
    misc_val = st.sidebar.number_input("Misc Val", min_value=0, max_value=17000, value=0, step=100)
    mo_sold = st.sidebar.number_input("Month Sold (1-12)", min_value=1, max_value=12, value=6, step=1)
    yr_sold = st.sidebar.number_input("Year Sold", min_value=2006, max_value=2010, value=2008, step=1)

    st.sidebar.markdown("---")
    
    # 2. CATEGORICAL FEATURES
    st.sidebar.subheader("Categorical Inputs")
    ms_zoning = st.sidebar.selectbox("MS Zoning", ["RL", "RH", "FV", "RM", "C (all)", "I (all)", "A (agr)"])
    street = st.sidebar.selectbox("Street Type", ["Pave", "Grvl"])
    alley = st.sidebar.selectbox("Alley Access", ["None", "Pave", "Grvl"])
    lot_shape = st.sidebar.selectbox("Lot Shape", ["Reg", "IR1", "IR2", "IR3"])
    land_contour = st.sidebar.selectbox("Land Contour", ["Lvl", "Bnk", "HLS", "Low"])
    utilities = st.sidebar.selectbox("Utilities", ["AllPub", "NoSewr", "NoSeWa"])
    lot_config = st.sidebar.selectbox("Lot Config", ["Inside", "Corner", "CulDSac", "FR2", "FR3"])
    land_slope = st.sidebar.selectbox("Land Slope", ["Gtl", "Mod", "Sev"])
    neighborhood = st.sidebar.selectbox("Neighborhood", ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt", "Gilbert", "Sawyer", "NWAmes", "SawyerW", "BrkSide", "Mitchel", "Crawfor", "NoRidge", "Timber", "IDOTRR", "ClearCr", "StoneBr", "SWISU", "MeadowV", "Blmngtn", "BrDale", "Veenker", "NPkVill", "Blueste", "Greens", "NaN", "Landmrk"])
    condition_1 = st.sidebar.selectbox("Condition 1", ["Norm", "Feedr", "Artery", "RRAn", "PosN", "RRAe", "PosA", "RRNe", "RRNn"])
    condition_2 = st.sidebar.selectbox("Condition 2", ["Norm", "Feedr", "Artery", "PosN", "PosA", "RRAn", "RRNn", "RRAe"])
    bldg_type = st.sidebar.selectbox("Building Type", ["1Fam", "TwnhsE", "Twnhs", "Duplex", "2fmCon"])
    house_style = st.sidebar.selectbox("House Style", ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer", "2.5Unf", "1.5Unf", "2.5Fin"])
    roof_style = st.sidebar.selectbox("Roof Style", ["Gable", "Hip", "Gambrel", "Mansard", "Flat", "Shed"])
    roof_matl = st.sidebar.selectbox("Roof Material", ["CompShg", "WdShngl", "Metal", "WdShake", "Membran", "Tar&Grv", "Roll", "ClyTile"])
    exterior_1st = st.sidebar.selectbox("Exterior 1st", ["VinylSd", "HdBoard", "MetalSd", "Wd Sdng", "Plywood", "CemntBd", "BrkFace", "AsbShng", "Stucco", "WdShing", "BrkComm", "Stone", "CBlock", "AsphShn", "ImStucc", "PreCast"])
    exterior_2nd = st.sidebar.selectbox("Exterior 2nd", ["VinylSd", "MetalSd", "HdBoard", "Wd Shng", "Plywood", "Wd Sdng", "CmentBd", "BrkFace", "Stucco", "AsbShng", "ImStucc", "Brk Cmn", "Stone", "AsphShn", "CBlock", "Other", "PreCast"])
    mas_vnr_type = st.sidebar.selectbox("Masonry Veneer Type", ["None", "BrkFace", "Stone", "BrkCmn", "CBlock"])
    exter_qual = st.sidebar.selectbox("Exterior Quality", ["TA", "Gd", "Ex", "Fa"])
    exter_cond = st.sidebar.selectbox("Exterior Condition", ["TA", "Gd", "Fa", "Po", "Ex"])
    foundation = st.sidebar.selectbox("Foundation", ["PConc", "CBlock", "BrkTil", "Slab", "Stone", "Wood"])
    bsmt_qual = st.sidebar.selectbox("Basement Quality", ["None", "Gd", "TA", "Ex", "Fa", "Po"])
    bsmt_cond = st.sidebar.selectbox("Basement Condition", ["None", "TA", "Gd", "Fa", "Po", "Ex"])
    bsmt_exposure = st.sidebar.selectbox("Basement Exposure", ["None", "No", "Av", "Mn", "Gd"])
    bsmt_fin_type_1 = st.sidebar.selectbox("Basement Fin Type 1", ["None", "GLQ", "ALQ", "Unf", "Rec", "BLQ", "LwQ"])
    bsmt_fin_type_2 = st.sidebar.selectbox("Basement Fin Type 2", ["None", "Unf", "Rec", "LwQ", "BLQ", "GLQ", "ALQ"])
    heating = st.sidebar.selectbox("Heating Type", ["GasA", "GasW", "Grav", "Wall", "Floor", "OthW"])
    heating_qc = st.sidebar.selectbox("Heating Quality & Condition", ["Ex", "Gd", "TA", "Fa", "Po"])
    central_air = st.sidebar.selectbox("Central Air", ["Y", "N"])
    electrical = st.sidebar.selectbox("Electrical System", ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix", "None"])
    kitchen_qual = st.sidebar.selectbox("Kitchen Quality", ["TA", "Gd", "Ex", "Fa", "Po"])
    functional = st.sidebar.selectbox("Home Functionality", ["Typ", "Min2", "Min1", "Mod", "Maj1", "Maj2", "Sal", "Sev"])
    fireplace_qu = st.sidebar.selectbox("Fireplace Quality", ["None", "Gd", "TA", "Ex", "Fa", "Po"])
    garage_type = st.sidebar.selectbox("Garage Type", ["None", "Attchd", "Detchd", "BuiltIn", "Basement", "CarPort", "2Types"])
    garage_finish = st.sidebar.selectbox("Garage Finish", ["None", "Unf", "RFn", "Fin"])
    garage_qual = st.sidebar.selectbox("Garage Quality", ["None", "TA", "Fa", "Gd", "Po", "Ex"])
    garage_cond = st.sidebar.selectbox("Garage Condition", ["None", "TA", "Fa", "Gd", "Po", "Ex"])
    paved_drive = st.sidebar.selectbox("Paved Drive", ["Y", "P", "N"])
    pool_qc = st.sidebar.selectbox("Pool Quality", ["None", "Ex", "Gd", "TA", "Fa"])
    fence = st.sidebar.selectbox("Fence Quality", ["None", "MnPrv", "GdPrv", "GdWo", "MnWw"])
    misc_feature = st.sidebar.selectbox("Misc Feature", ["None", "Shed", "Gar2", "Othr", "TenC", "Elev"])
    sale_type = st.sidebar.selectbox("Sale Type", ["WD ", "New", "COD", "ConLD", "ConLI", "ConLw", "CWD", "Con", "Oth", "VWD"])
    sale_condition = st.sidebar.selectbox("Sale Condition", ["Normal", "Partial", "Abnorml", "Family", "Alloca", "AdjLand"])

    data = {
        "MS SubClass": [ms_subclass], "Lot Frontage": [lot_frontage], "Lot Area": [lot_area],
        "Overall Qual": [overall_qual], "Overall Cond": [overall_cond], "Year Built": [year_built],
        "Year Remod/Add": [year_remod_add], "Garage Yr Blt": [garage_yr_blt], "Mas Vnr Area": [mas_vnr_area],
        "BsmtFin SF 1": [bsmtfin_sf_1], "BsmtFin SF 2": [bsmtfin_sf_2], "Bsmt Unf SF": [bsmt_unf_sf],
        "Total Bsmt SF": [total_bsmt_sf], "1st Flr SF": [first_flr_sf], "2nd Flr SF": [second_flr_sf],
        "Low Qual Fin SF": [low_qual_fin_sf], "Gr Liv Area": [gr_liv_area], "Bsmt Full Bath": [bsmt_full_bath],
        "Bsmt Half Bath": [bsmt_half_bath], "Full Bath": [full_bath], "Half Bath": [half_bath],
        "Bedroom AbvGr": [bedroom_abv_gr], "Kitchen AbvGr": [kitchen_abv_gr], "TotRms AbvGrd": [tot_rms_abv_grd],
        "Fireplaces": [fireplaces], "Garage Cars": [garage_cars], "Garage Area": [garage_area],
        "Wood Deck SF": [wood_deck_sf], "Open Porch SF": [open_porch_sf], "Enclosed Porch": [enclosed_porch],
        "3Ssn Porch": [three_ssn_porch], "Screen Porch": [screen_porch], "Pool Area": [pool_area],
        "Misc Val": [misc_val], "Mo Sold": [mo_sold], "Yr Sold": [yr_sold],
        "MS Zoning": [ms_zoning], "Street": [street], "Alley": [None if alley == "None" else alley],
        "Lot Shape": [lot_shape], "Land Contour": [land_contour], "Utilities": [utilities],
        "Lot Config": [lot_config], "Land Slope": [land_slope], "Neighborhood": [neighborhood],
        "Condition 1": [condition_1], "Condition 2": [condition_2], "Bldg Type": [bldg_type],
        "House Style": [house_style], "Roof Style": [roof_style], "Roof Matl": [roof_matl],
        "Exterior 1st": [exterior_1st], "Exterior 2nd": [exterior_2nd],
        "Mas Vnr Type": [None if mas_vnr_type == "None" else mas_vnr_type],
        "Exter Qual": [exter_qual], "Exter Cond": [exter_cond], "Foundation": [foundation],
        "Bsmt Qual": [None if bsmt_qual == "None" else bsmt_qual],
        "Bsmt Cond": [None if bsmt_cond == "None" else bsmt_cond],
        "Bsmt Exposure": [None if bsmt_exposure == "None" else bsmt_exposure],
        "BsmtFin Type 1": [None if bsmt_fin_type_1 == "None" else bsmt_fin_type_1],
        "BsmtFin Type 2": [None if bsmt_fin_type_2 == "None" else bsmt_fin_type_2],
        "Heating": [heating], "Heating QC": [heating_qc], "Central Air": [central_air],
        "Electrical": [None if electrical == "None" else electrical], "Kitchen Qual": [kitchen_qual],
        "Functional": [functional], "Fireplace Qu": [None if fireplace_qu == "None" else fireplace_qu],
        "Garage Type": [None if garage_type == "None" else garage_type],
        "Garage Finish": [None if garage_finish == "None" else garage_finish],
        "Garage Qual": [None if garage_qual == "None" else garage_qual],
        "Garage Cond": [None if garage_cond == "None" else garage_cond],
        "Paved Drive": [paved_drive], "Pool QC": [None if pool_qc == "None" else pool_qc],
        "Fence": [None if fence == "None" else fence],
        "Misc Feature": [None if misc_feature == "None" else misc_feature],
        "Sale Type": [sale_type], "Sale Condition": [sale_condition]
    }

    input_df = pd.DataFrame(data)
    processed_df = engineer_features(input_df)

    if st.button("🚀 Predict Property Price", type="primary"):
        predicted_price = np.expm1(model.predict(processed_df)[0])

        st.subheader("📊 Valuation Result")
        st.metric(label="Estimated Market Value", value=f"${predicted_price:,.2f}")

        if predicted_price > 300000:
            st.success("🌟 **High-Value Property:** Premium asset classification based on market standards.")
            category = "High-Value Property"
        elif predicted_price < 150000:
            st.info("💡 **Affordable Segment:** Entry-level pricing bracket.")
            category = "Affordable Segment"
        else:
            st.success("✅ **Standard Residential Property:** Well-balanced valuation.")
            category = "Standard Property"

        # Client Feature: Downloadable Valuation Certificate (Text/Report Summary)
        report_text = f"""--- REAL ESTATE VALUATION CERTIFICATE ---
Property Details:
- Living Area: {gr_liv_area} Sq Ft
- Overall Quality: {overall_qual}/10
- Year Built: {year_built}
- Neighborhood: {neighborhood}

AI Valuation Results:
- Estimated Market Value: ${predicted_price:,.2f}
- Property Classification: {category}
------------------------------------------
Generated by AI-Powered Housing Price Engine.
"""
        st.download_button(
            label="📥 Download Valuation Certificate",
            data=report_text,
            file_name="Property_Valuation_Certificate.txt",
            mime="text/plain"
        )


# ==================== TAB 2: BULK CSV BATCH ====================
with tab2:
    st.header("📊 Enterprise Bulk Housing Portfolio Audit")
    
    uploaded_file = st.file_uploader(
        "Upload Portfolio CSV File (Unlimited Rows)", type=["csv"]
    )

    if uploaded_file is not None:
        raw_batch = pd.read_csv(uploaded_file)
        
        # Missing values handling
        numerical_cols = raw_batch.select_dtypes(include=['number']).columns
        raw_batch[numerical_cols] = raw_batch[numerical_cols].fillna(0)

        categorical_cols = raw_batch.select_dtypes(include=['object']).columns
        raw_batch[categorical_cols] = raw_batch[categorical_cols].fillna('None')
        
        st.info(f"📁 Loaded dataset containing **{len(raw_batch)}** rows.")

        processed_batch = engineer_features(raw_batch)
        
        # Model Prediction with Inverse Log Transformation
        raw_batch["Predicted_Price"] = np.expm1(model.predict(processed_batch)).round(2)

        # Accuracy & Error Metrics Calculation (If 'SalePrice' exists)
        has_actual_price = "SalePrice" in raw_batch.columns
        if has_actual_price:
            raw_batch["Difference ($)"] = (raw_batch["Predicted_Price"] - raw_batch["SalePrice"]).round(2)
            raw_batch["Error (%)"] = (abs(raw_batch["Difference ($)"] / raw_batch["SalePrice"]) * 100).round(2)

        # Category mapping function
        def categorize_price(price):
            if price > 300000:
                return "High-Value Property"
            elif price < 150000:
                return "Affordable Segment"
            else:
                return "Standard Property"

        raw_batch["Property_Category"] = raw_batch["Predicted_Price"].apply(categorize_price)

        total_properties = len(raw_batch)
        avg_price = raw_batch["Predicted_Price"].mean()
        total_portfolio_value = raw_batch["Predicted_Price"].sum()

        # Top KPI Metrics Row
        col1, col2, col3 = st.columns(3)
        col1.metric("🏠 Total Properties", total_properties)
        col2.metric("💰 Avg Estimated Price", f"${avg_price:,.2f}")
        col3.metric("📈 Total Portfolio Value", f"${total_portfolio_value:,.2f}")

        # Client Feature: Model Global Metrics (Agar actual SalePrice available ho)
        if has_actual_price:
            st.markdown("---")
            st.subheader("🎯 Portfolio Model Performance Analytics")
            r2 = r2_score(raw_batch["SalePrice"], raw_batch["Predicted_Price"])
            mae = mean_absolute_error(raw_batch["SalePrice"], raw_batch["Predicted_Price"])
            
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("📊 R² Score (Accuracy Metric)", f"{r2:.4f}")
            m_col2.metric("📉 Mean Absolute Error (MAE)", f"${mae:,.2f}")

        # Client Feature: Advanced Filters for Data Table
        st.markdown("---")
        st.subheader("🔍 Filter Portfolio Data")
        selected_categories = st.multiselect(
            "Filter by Property Category:",
            options=raw_batch["Property_Category"].unique(),
            default=list(raw_batch["Property_Category"].unique())
        )
        
        filtered_batch = raw_batch[raw_batch["Property_Category"].isin(selected_categories)]

        st.subheader("📋 Evaluated Portfolio Preview")
        st.dataframe(filtered_batch, use_container_width=True)

        # Client Feature: Interactive Charts
        st.markdown("---")
        st.subheader("📈 Visual Analytics & Market Insights")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.write("**Property Category Breakdown**")
            category_counts = raw_batch["Property_Category"].value_counts()
            st.bar_chart(category_counts)

        with chart_col2:
            if has_actual_price:
                st.write("**Actual Price vs Predicted Price Scatter Comparison**")
                # Scatter data view using line/scatter chart format
                chart_data = raw_batch[["SalePrice", "Predicted_Price"]]
                st.scatter_chart(chart_data)
            else:
                st.write("**Predicted Prices Distribution**")
                st.bar_chart(raw_batch["Predicted_Price"])

        # CSV Download Button
        st.markdown("---")
        csv_data = raw_batch.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Full Housing Valuation Report (CSV)",
            csv_data,
            "Housing_Price_Batch_Audit_Results.csv",
            "text/csv",
        )