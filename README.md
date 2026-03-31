#  Customer Churn Prediction & Action System

##  End-to-End Workflow

This system is designed to not only predict customer churn but also provide actionable business insights to reduce churn risk.

### 1. User Interface (Streamlit Dashboard)
The frontend is built using Streamlit, enabling business users to interact with the system easily.
- Users upload customer data via CSV
- Dashboard displays key metrics such as total customers and high-risk segments
- Provides a clear and interactive view of predictions

### 2. AI Prediction Engine (Machine Learning)
The backend uses machine learning models to evaluate churn risk.
- Generates churn probability (0-100%) instead of binary output
- Supports multiple models (MLP, Random Forest, XGBoost)
- Loads trained models from the `/models` directory for real-time predictions

### 3. Business Logic Layer (Action System)
Transforms predictions into actionable insights:
- **Risk Segmentation:** Low (<30%), Medium (30-70%), High (>70%)
- **Explainability:** Identifies top factors influencing churn (e.g., high charges, low tenure)
- **Action Recommendations:** Provides business strategies based on risk level

**Example:**
- High Risk + High Charges -> Offer discount plan
- High Risk + Poor Support -> Initiate personalized customer support call

### 4. Project Architecture
- `/notebooks` -> EDA and model experimentation
- `/models` -> Trained models (.pkl, .pth)
- `/src` -> Training and preprocessing scripts
- `/app/main.py` -> Streamlit dashboard

### 5. How to Use
1. Run the app: `streamlit run app/main.py`
2. Upload customer dataset
3. Click "Predict Churn Risk"
4. View predictions, risk levels, and recommended actions

###  Impact
- Identifies high-risk customers proactively
- Enables data-driven retention strategies
- Reduces manual analysis effort and improves decision-making speed
