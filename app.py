import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st
import altair as alt
import io

# ---------------------------
# Custom CSS for background + overlay + bold labels + chart styling
# ---------------------------
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1560185127-6ed189bf02f4");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(255,255,255,0.7); /* semi-transparent overlay */
    z-index: -1;
}
[data-testid="stHeader"], [data-testid="stSidebar"] {
    background: rgba(0,0,0,0);
}
/* Make title bigger */
h1 {
    font-size: 42px !important;
    font-weight: bold !important;
    text-align: center !important;
}
/* Make labels bold and larger */
label, .stNumberInput label, .stSelectbox label {
    font-weight: bold !important;
    font-size: 20px !important;
}
/* Chart titles and axis labels */
.vega-embed .vega-actions {display:none;}
.vega-embed .mark-text, .vega-embed .mark-label, .vega-embed .axis-title, .vega-embed .axis-label {
    font-size: 16px !important;
    font-weight: bold !important;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("C:/Users/Rajesh/real_estate/indian-real-estate-market-dataset-2023-25/indian_realestate_dataset_1000.csv")

# Features and target
X = df[["built_up_area_sqft","bedrooms","bathrooms","age_years","city"]]
y = df["price_inr"]

# Encode categorical city
X = pd.get_dummies(X, columns=["city"], drop_first=True)

# Scale numeric features for better predictions
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Prediction function
def predict_price(area, rooms, bathrooms, age, location):
    input_data = pd.DataFrame([[area, rooms, bathrooms, age, location]], 
                              columns=["built_up_area_sqft","bedrooms","bathrooms","age_years","city"])
    input_data = pd.get_dummies(input_data, columns=["city"])
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_data)
    return model.predict(input_scaled)[0]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🏠 Indian Real Estate Price Predictor")

# Typing inputs (number_input) for all except city
area = st.number_input("Built-up Area (sqft)", min_value=100, max_value=10000, step=50)
rooms = st.number_input("Bedrooms", min_value=1, max_value=10, step=1)
baths = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)
age = st.number_input("Age (years)", min_value=0, max_value=100, step=1)

# Dropdown for city names from dataset (unchanged)
city_list = sorted(df["city"].dropna().unique())
city = st.selectbox("City", city_list)

if st.button("Predict Price"):
    price = predict_price(area, rooms, baths, age, city)
    price_per_sqft = price / area if area > 0 else 0

    # Styled output
    st.markdown(f"<h2>✨ Estimated Price: <b>₹{round(price,2):,}</b></h2>", unsafe_allow_html=True)
    st.markdown(f"<h3>📐 Price per Sqft: <b>₹{round(price_per_sqft,2):,}</b></h3>", unsafe_allow_html=True)

    # ---------------------------
    # Bar chart of average prices per city with highlight
    # ---------------------------
    avg_prices = df.groupby("city")["price_inr"].mean().reset_index()
    chart = alt.Chart(avg_prices).mark_bar().encode(
        x=alt.X("city", sort="-y", title="City"),
        y=alt.Y("price_inr", title="Average Price (INR)"),
        color=alt.condition(
            alt.datum.city == city,
            alt.value("orange"),     # highlight selected city
            alt.value("steelblue")   # default color
        )
    ).properties(
        width=700,
        height=400,
        title="📊 Average Prices per City"
    )
    st.altair_chart(chart, use_container_width=True)

    # ---------------------------
    # Trend line: year-by-year average price in selected city with color gradient + moving average
    # ---------------------------
    city_data = df[df["city"] == city]
    if not city_data.empty:
        yearly_avg = city_data.groupby("age_years")["price_inr"].mean().reset_index()
        yearly_avg["change"] = yearly_avg["price_inr"].diff().fillna(0)
        yearly_avg["trend_color"] = yearly_avg["change"].apply(lambda x: "green" if x > 0 else ("red" if x < 0 else "gray"))
        yearly_avg["moving_avg"] = yearly_avg["price_inr"].rolling(window=3, min_periods=1).mean()

        trend_line = alt.Chart(yearly_avg).mark_line(point=True).encode(
            x=alt.X("age_years", title="Property Age (years)"),
            y=alt.Y("price_inr", title="Average Price (INR)"),
            color=alt.Color("trend_color:N", scale=None),
            tooltip=["age_years","price_inr","change"]
        )

        moving_avg_line = alt.Chart(yearly_avg).mark_line(strokeDash=[5,5], color="blue").encode(
            x="age_years",
            y="moving_avg",
            tooltip=["age_years","moving_avg"]
        )

        trend = (trend_line + moving_avg_line).properties(
            width=700,
            height=400,
            title=f"📈 Year-by-Year Price Trend in {city} (with Moving Average)"
        )
        st.altair_chart(trend, use_container_width=True)

    # ---------------------------
    # Download report as CSV
    # ---------------------------
    report = pd.DataFrame({
        "Area (sqft)": [area],
        "Bedrooms": [rooms],
        "Bathrooms": [baths],
        "Age (years)": [age],
        "City": [city],
        "Predicted Price (INR)": [round(price,2)],
        "Price per Sqft (INR)": [round(price_per_sqft,2)]
    })

    csv_buffer = io.StringIO()
    report.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Prediction Report",
        data=csv_buffer.getvalue(),
        file_name="real_estate_prediction.csv",
        mime="text/csv"
    )
