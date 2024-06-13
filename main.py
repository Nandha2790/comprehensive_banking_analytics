import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_option_menu import option_menu
from joblib import load
from sklearn.preprocessing import StandardScaler


st.set_page_config(layout='wide')

st.title("Comprehensive Banking Analytics")
selected = option_menu( 
                       menu_title=None,
                       options= ["Loan_Approval"],
                       orientation='horizontal',
                       )

if selected == "Loan_Approval":
    loan_applicants = st.file_uploader("Upload file as .CSV",)
    
    def labeling(late_payment):
        
        '''
        we have late payments in range between 0 to 25, we will lable with below categorized value
        1. low risk: 0 to 2
        2. medium risk: 3 to 10
        3. high risk: 11 to 25
        
        '''
        if late_payment<=2:
            return 'low_risk'
        elif 3 <= late_payment <=10 :
            return 'medium_risk'
        else:
            return 'high_risk'
        
    def loan(x,y):
        if ( x in ['Standard','Good'] ) and (y in ['low_risk','medium_risk']):
            return 'Approved'
        else:
            return 'Rejected' 

    def loan_approval(file_name):
        df = pd.read_csv(file_name) # data reading

        #descrite data in the dataset
        num_cols_disc = [ 'ID','Customer_ID','Age','Num_Bank_Accounts','Num_Credit_Card','Num_of_Loan','Delay_from_due_date','Num_of_Delayed_Payment',
                        'Num_Credit_Inquiries','Credit_History_Age']
        
        for i in num_cols_disc:
            df[i] = df[i].astype(int)   # converting datatype to round numbers(int)
        
        df1 = df.copy()
        
        cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Interest_Rate', 'Changed_Credit_Limit', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 
                'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance'] # for log tranformation same like model training
        
        for i in cols:
            df1[f"{i}_log"] = np.log(df1[i])
        
        cols_x = ['Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment','Num_Credit_Inquiries',
                'Credit_Mix', 'Credit_History_Age', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Annual_Income_log', 'Monthly_Inhand_Salary_log',
                'Interest_Rate_log', 'Changed_Credit_Limit_log', 'Outstanding_Debt_log', 'Credit_Utilization_Ratio_log', 'Total_EMI_per_month_log',
                'Amount_invested_monthly_log', 'Monthly_Balance_log'] # choosing feature ordening same like model training
        
        x = df1[cols_x] 
        
        x1 = x.copy()
        
        #encoding the categorigal columns
        
        Credit_Mix_map = { 'Bad':0, 'Standard':1, 'Good': 2  }
        Payment_of_Min_Amount_map = { 'No':0, 'NM':1, 'Yes':2 }
        Payment_Behaviour_map = { 'Low_spent_Small_value_payments': 0 , 'Low_spent_Medium_value_payments':1, 'Low_spent_Large_value_payments':2,
                                'High_spent_Small_value_payments':3, 'High_spent_Medium_value_payments':4, 'High_spent_Large_value_payments':5
                                }
        
        x1['Credit_Mix'] = x1['Credit_Mix'].map(Credit_Mix_map)
        
        x1['Payment_of_Min_Amount'] = x1['Payment_of_Min_Amount'].map(Payment_of_Min_Amount_map)
        
        x1['Payment_Behaviour'] = x1['Payment_Behaviour'].map(Payment_Behaviour_map)

        #feature scaling
        scaler = StandardScaler()
        scaled_class = scaler.fit_transform(x1)
        x_scaled = pd.DataFrame(scaled_class, columns=x1.columns)
        
        #loading the trained model
        model = load('classification.joblib')

        #define the feature names for output dataset
        cols_dataset = [ 'Customer_ID', 'Name', 'Age', 'Occupation', 'Annual_Income', 'Monthly_Inhand_Salary', 'Total_EMI_per_month','Num_of_Delayed_Payment' ,'Interest_Rate']

        #output data created with selected features
        df_predicted = df[cols_dataset]

        # model prediction 
        df_predicted.loc[:, 'Credit_Score_Predicted'] = model.predict(x_scaled)
        
        #Risk Assesment
        df_predicted['Risk_Assessment'] = df_predicted['Num_of_Delayed_Payment'].apply(labeling)

        # mapping the loan status based on the predicted value
        df_predicted['Loan_Status'] = df_predicted.apply( lambda row: loan(row['Credit_Score_Predicted'], row['Risk_Assessment']),axis=1)
         
        return df_predicted
    
    if loan_applicants:
        df = loan_approval(loan_applicants)
        st.dataframe(df,use_container_width=True)
        
        df_approved = df[ df['Loan_Status'] == 'Approved'].reset_index(drop=True)
        df_rejected = df[ df['Loan_Status'] == 'Rejected'].reset_index(drop=True)
        
        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode("utf-8")
        
        csv_approved = convert_df(df_approved)
        csv_rejected = convert_df(df_rejected)
        
        st.download_button( "Download approved loan applicants as CSV file ", csv_approved, "Loan_Approved.csv",mime='text/csv')
        st.download_button( "Download rejected loan applicants as CSV file ", csv_rejected, "Loan_Rejected.csv",mime='text/csv')
        
        st.header("Creditworthiness of loan applicants:",divider='rainbow')
        
        col1,col2,col3 = st.columns([4.2,10,10])
        
        total = df.shape[0]
        appr = df[ df['Loan_Status'] == 'Approved']['Loan_Status'].value_counts()[0]
        rej = df[ df['Loan_Status'] == 'Rejected']['Loan_Status'].value_counts()[0]
        col1.container(border=True).metric('Total applicants', total,)
        col2.container(border=True).metric('Approved', appr, f"{(appr/total)*100:.2f}" )
        col3.container(border=True).metric('Rejected', rej, f"{(-rej/total)*100:.2f}" )
        
        
        
        col1,col2 = st.columns([10,7])
        fig_bar = px.bar( df['Occupation'].value_counts().reset_index() ,
                     x='Occupation', 
                     y = 'count' , 
                     text_auto=True,
                     title="Applicants Occupation counts"
                     )
        fig_bar.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        
        col1.container(border=True).plotly_chart(fig_bar,use_container_width=True)

        fig_bar1 = px.bar( df['Credit_Score_Predicted'].value_counts().reset_index() ,
                     x='Credit_Score_Predicted', 
                     y = 'count' , 
                     text_auto=True,
                     title=f"Applicants Credit Score Predicted summary "
                     )
        fig_bar1.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        col2.container(border=True).plotly_chart(fig_bar1,use_container_width=True)
        
        df1_occupation =  df.groupby('Occupation').agg({'Annual_Income': ['mean', 'min', 'max']}).reset_index()
        
        st.subheader("Applicants Salary details Summarry:")
        st.table(df1_occupation)
        
        
        
        
