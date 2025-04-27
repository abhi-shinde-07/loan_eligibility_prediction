import streamlit as st
import pandas as pd
from loan_predictor import LoanPredictor
import time
import plotly.express as px

# Initialize predictor
predictor = LoanPredictor()

# Set page config
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
            border-radius: 5px;
        }
        .prediction-card {
            border-radius: 10px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        }
        .approved {
            background-color: #d4edda;
            color: #155724;
        }
        .rejected {
            background-color: #f8d7da;
            color: #721c24;
        }
    </style>
""", unsafe_allow_html=True)

# App header
st.title("💰 Loan Eligibility Prediction System")
st.markdown("""
    This system uses machine learning to predict whether a loan application is likely to be approved based on applicant information.
""")

# Navigation
tab1, tab2, tab3 = st.tabs(["New Application", "Application History", "Business Insights"])

with tab1:
    st.header("New Loan Application")
    
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            applicant_name = st.text_input("Full Name")
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Marital Status", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        
        with col2:
            st.subheader("Financial Information")
            applicant_income = st.number_input("Applicant Income (USD)", min_value=0, step=100)
            coapplicant_income = st.number_input("Coapplicant Income (USD)", min_value=0, step=100)
            loan_amount = st.number_input("Loan Amount (USD)", min_value=0, step=1000)
            loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, step=12)
            credit_history = st.selectbox("Credit History (1=Good, 0=Bad)", [1, 0])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
        submitted = st.form_submit_button("Check Eligibility")
    
    if submitted:
        with st.spinner("Analyzing application..."):
            # Prepare input data
            input_data = {
                "applicant_name": applicant_name,
                "gender": gender,
                "married": married,
                "dependents": dependents,
                "education": education,
                "self_employed": self_employed,
                "applicant_income": applicant_income,
                "coapplicant_income": coapplicant_income,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "credit_history": credit_history,
                "property_area": property_area
            }
            
            # Make prediction
            prediction, probability = predictor.predict(input_data)
            time.sleep(1)  # Simulate processing time
            
            # Display results
            if prediction is not None:
                # Save application to database
                predictor.save_application(input_data, prediction, probability)
                
                # Show prediction
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.markdown(f"""
                        <div class="prediction-card approved">
                            <h2>✅ Approved (Confidence: {probability*100:.1f}%)</h2>
                            <p>Congratulations! Based on our analysis, this application is likely to be approved.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Show reasons (feature importance)
                    st.subheader("Key Approval Factors")
                    st.markdown("""
                        The following factors contributed positively to this decision:
                    """)
                    st.markdown("""
                        - Good credit history
                        - Stable income
                        - Reasonable loan amount to income ratio
                    """)
                else:
                    st.markdown(f"""
                        <div class="prediction-card rejected">
                            <h2>❌ Rejected (Confidence: {(1-probability)*100:.1f}%)</h2>
                            <p>Based on our analysis, this application doesn't meet our current eligibility criteria.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Show reasons
                    st.subheader("Key Rejection Factors")
                    st.markdown("""
                        The following factors negatively impacted this decision:
                    """)
                    st.markdown("""
                        - Low credit score
                        - High debt-to-income ratio
                        - Insufficient collateral
                    """)
                
                # Show next steps
                st.subheader("Next Steps")
                if prediction == 1:
                    st.markdown("""
                        - A loan officer will contact you within 2 business days
                        - Please have your supporting documents ready
                        - Final approval is subject to verification
                    """)
                else:
                    st.markdown("""
                        - You may reapply in 6 months
                        - Consider improving your credit score
                        - You may qualify for a smaller loan amount
                    """)
            else:
                st.error("There was an error processing your application. Please try again.")

with tab2:
    st.header("Application History")
    st.markdown("View past loan applications and their outcomes.")
    
    history_df = predictor.get_application_history()
    
    if not history_df.empty:
        # Convert timestamp and prediction result
        history_df['submission_date'] = pd.to_datetime(history_df['submission_date'])
        history_df['prediction_result'] = history_df['prediction_result'].map({1: 'Approved', 0: 'Rejected'})
        
        # Show data
        st.dataframe(
            history_df.drop('id', axis=1).rename(columns={
                'applicant_name': 'Name',
                'prediction_result': 'Result',
                'prediction_proba': 'Confidence',
                'submission_date': 'Date'
            }),
            use_container_width=True
        )
        
        # Export option
        st.download_button(
            "Export to CSV",
            history_df.to_csv(index=False).encode('utf-8'),
            "loan_applications.csv",
            "text/csv"
        )
    else:
        st.info("No application history found.")

with tab3:
    st.header("Business Insights")
    st.markdown("Analytics dashboard for loan application trends.")
    
    history_df = predictor.get_application_history()
    
    if not history_df.empty:
        # Convert data types
        history_df['prediction_result'] = history_df['prediction_result'].map({1: 'Approved', 0: 'Rejected'})
        
        # Approval rate over time
        st.subheader("Approval Rate Over Time")
        approval_over_time = history_df.groupby([
            pd.Grouper(key='submission_date', freq='M'), 
            'prediction_result'
        ]).size().unstack().fillna(0)
        approval_over_time['Approval Rate'] = approval_over_time['Approved'] / (approval_over_time['Approved'] + approval_over_time['Rejected'])
        
        fig = px.line(
            approval_over_time, 
            x=approval_over_time.index, 
            y='Approval Rate',
            labels={'x': 'Month', 'y': 'Approval Rate'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution by property area
        st.subheader("Approvals by Property Area")
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                area_dist = history_df.groupby(['property_area', 'prediction_result']).size().unstack(fill_value=0)
                if 'Approved' in area_dist.columns:
                    fig = px.pie(
                        area_dist,
                        values='Approved',
                        names=area_dist.index,
                        title="Approved Loans by Area"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No 'Approved' data available for pie chart.")
            except Exception as e:
                st.error(f"Error generating pie chart: {e}")

        with col2:
            try:
                fig = px.bar(
                    area_dist,
                    x=area_dist.index,
                    y=['Approved', 'Rejected'] if 'Approved' in area_dist.columns and 'Rejected' in area_dist.columns else area_dist.columns,
                    barmode='group',
                    title="Approvals vs Rejections by Area"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating bar chart: {e}")
        
        # Income analysis
        st.subheader("Income Analysis")
        fig = px.box(
            history_df,
            x='prediction_result',
            y='applicant_income',
            color='property_area',
            title="Applicant Income Distribution by Approval Status"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No data available for analytics. Submit some applications first.")