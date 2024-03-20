import gradio as gr
import pandas as pd
import numpy as np
import joblib, os


script_dir = os.path.dirname(os.path.abspath(__file__))

pipeline_path = os.path.join(script_dir, 'toolkit', 'pipeline.joblib')
model_path = os.path.join(script_dir, 'toolkit', 'Random Forest Classifier.joblib')

# Load transformation pipeline and model
pipeline = joblib.load(pipeline_path)
model = joblib.load(model_path)



# Create a function that applies the ML pipeline and makes predictions
def predict(SeniorCitizen,Partner,Dependents, tenure,
            InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,
            StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,
            MonthlyCharges,TotalCharges):

    # Create a dataframe with the input data
     input_df = pd.DataFrame({
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]

 })

# Selecting categorical and numerical columns separately
     cat_cols = [col for col in input_df.columns if input_df[col].dtype == 'object']
     num_cols = [col for col in input_df.columns if input_df[col].dtype != 'object']

     X_processed = pipeline.transform(input_df)

     # Extracting feature names for numerical columns
     num_feature_names = num_cols

     # Extracting feature names for categorical columns after one-hot encoding
     cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
     cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)

     # Concatenating numerical and categorical feature names
     feature_names = num_feature_names + list(cat_feature_names)

     # Convert X_processed to DataFrame
     final_df = pd.DataFrame(X_processed, columns=feature_names)

     # Extract the first three columns
     first_three_columns = final_df.iloc[:, :3]

     # Extract the remaining columns except the first three
     remaining_columns = final_df.iloc[:, 3:]

     # Concatenate the remaining columns with the first three columns shifted to the end
     final_df = pd.concat([remaining_columns, first_three_columns], axis=1)

     # Make predictions using the model
     predictions = model.predict(final_df)

     # prediction_label = "This customer is likely to Churn" if predictions.item() == "Yes" else "This customer is Not likely churn"
     prediction_label = {"Prediction: CHURN ": float(predictions[0]), "Prediction: STAY": 1-float(predictions[0])}

     return prediction_label




input_interface = []

with gr.Blocks(theme=gr.themes.Soft()) as app:

    Title = gr.Label('Customer Churn Prediction App')

    with gr.Row():
        Title

    with gr.Row():
        gr.Markdown("This app predicts likelihood of a customer to leave or stay with the company")

    with gr.Row():
        with gr.Column():
            input_interface_column_1 = [
                gr.components.Radio(['Yes', 'No'], label="Are you a Seniorcitizen?"),
                gr.components.Radio(['Yes', 'No'], label='Do you have Partner?'),
                gr.components.Radio(['No', 'Yes'], label='Do you have any Dependents?'),
                gr.components.Slider(label='Lenghth of tenure in months', minimum=0, maximum=73),
                gr.components.Radio(['DSL', 'Fiber optic', 'No'], label='Do you have InternetService'),
                gr.components.Radio(['No', 'Yes'], label='Do you have OnlineSecurity?'),
                gr.components.Radio(['No', 'Yes'], label='Do you have OnlineBackup?')
            ]

        with gr.Column():
            input_interface_column_2 = [
                gr.components.Radio(['No', 'Yes'], label='Do you have DeviceProtection?'),
                gr.components.Radio(['No', 'Yes'], label='Do you have TechSupport?'),
                gr.components.Radio(['No', 'Yes'], label='Do you have StreamingTV?'),
                gr.components.Radio(['No', 'Yes'], label='Do you have StreamingMovies?'),
                gr.components.Radio(['Month-to-month', 'One year', 'Two year'], label='which Contract do you use?'),
                gr.components.Radio(['Yes', 'No'], label='Do you prefer PaperlessBilling?'),
                gr.components.Radio(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], label='Which PaymentMethod do you prefer?'),
                gr.components.Slider(label="Enter monthly charges", minimum=18.40, maximum=118.65),
                gr.components.Slider(label="Enter total charges", maximum=9000)
            ]

    with gr.Row():
        input_interface.extend(input_interface_column_1)
        input_interface.extend(input_interface_column_2)

    with gr.Row():
        predict_btn = gr.Button('Predict')
        output_interface = gr.Label(label="churn")
    
    with gr.Accordion("Open for information on inputs"):
        gr.Markdown("""This app receives the following as inputs and processes them to return the prediction on whether a customer, given the inputs, will churn or not.
                    
                    - SeniorCitizen: Whether a customer is a senior citizen or not
                    - Partner: Whether the customer has a partner or not (Yes, No)
                    - Dependents: Whether the customer has dependents or not (Yes, No)
                    - Tenure: Number of months the customer has stayed with the company
                    - InternetService: Customer's internet service provider (DSL, Fiber Optic, No)
                    - OnlineSecurity: Whether the customer has online security or not (Yes, No, No Internet)
                    - OnlineBackup: Whether the customer has online backup or not (Yes, No, No Internet)
                    - DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)
                    - TechSupport: Whether the customer has tech support or not (Yes, No, No internet)
                    - StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service)
                    - StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No Internet service)
                    - Contract: The contract term of the customer (Month-to-Month, One year, Two year)
                    - PaperlessBilling: Whether the customer has paperless billing or not (Yes, No)
                    - Payment Method: The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))
                    - MonthlyCharges: The amount charged to the customer monthly
                    - TotalCharges: The total amount charged to the customer
                    """)
        
    predict_btn.click(fn=predict, inputs=input_interface, outputs=output_interface)
        
app.launch(share=True)