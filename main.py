import streamlit as st
from streamlit_option_menu import option_menu
import chardet
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.regression import (
    setup as reg_setup, compare_models as reg_compare, 
    create_model as reg_create, pull as reg_pull, 
    ensemble_model as reg_ensemble, tune_model as reg_tune, 
    predict_model as reg_predict, save_model as reg_save
)
import tempfile

# Initialize session state
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None

# Initialize other session states if not already set
for key in ["setup1", "setup2", "compare1", "compare2", "create1", "create2", "user_input", "predict1", "predict2", "ensemble1", "ensemble2", "tune1", "tune2",
            "predict1_ensemble", "predict2_ensemble","tune1_predict","tune2_predict","tune1_ensemble","tune2_ensemble","tune1_ensemble_prredict","tune2_ensemble_prredict",
            "save1","save2","save3","save4","save1FileName","save2FileName","save3FileName","save4FileName"]:
    if key not in st.session_state:
        st.session_state[key] = None

class Regression:
    def __init__(self, data):
        self.dataset = data

    def layout(self):
        col1, col2 = st.columns([2, 1])
        with col2:
            # Display checkboxes for model options
            create_model = st.checkbox("Create Model")
            ensemble_model = st.checkbox("Ensemble Model")
            tune_model = st.checkbox("Tune Model")
            tune_ensemble = st.checkbox("Tune Ensemble Model")
            save_model = st.checkbox("Save Model")  # Add option to save the model

            # Execute functions based on checkbox selection
            if create_model:
                with col1:
                    self.create_model()
            if ensemble_model:
                with col1:
                    self.ensemble_model()
            if tune_model:
                with col1:
                    self.tune_model()
            if tune_ensemble:
                with col1:
                    self.tune_ensemble()

    def create_model(self):
        # Only setup the model if it hasn't been created already
        if st.session_state.setup1 is None:
            st.markdown("#### Setup Your Environment by Selecting the Target Variable")
            target_variable = st.selectbox("Select the target for regression", self.dataset.columns)
            if st.checkbox(f"Confirm selection of '{target_variable}' as target"):
                # Setup environment
                st.session_state.setup1 = reg_setup(self.dataset, target=target_variable, preprocess=True)
                st.session_state.setup2 = reg_pull()
                st.markdown("#### Setup Complete:")
                st.dataframe(st.session_state.setup2)

                # Compare models
                st.markdown("#### Comparing Models:")
                st.session_state.compare1 = reg_compare()
                st.session_state.compare2 = reg_pull()
                st.dataframe(st.session_state.compare2)
        else:
            st.markdown("#### Model Already Set Up")
            st.dataframe(st.session_state.setup2)
            st.markdown("#### Model Comparison Results")
            st.dataframe(st.session_state.compare2)

        # If no model has been created yet, allow user to create one
        if st.session_state.create1 is None:
            regressors = [
                'lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 'ransac', 'tr', 'huber',
                'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 'mlp', 'xgboost', 'lightgbm', 'catboost'
            ]
            st.session_state.user_input = st.selectbox("Select a model to create", regressors)
            if st.checkbox("Proceed to create"):
                st.session_state.create1 = reg_create(st.session_state.user_input)
                st.session_state.create2 = reg_pull()
                st.markdown(f"#### Model '{st.session_state.user_input}' Created:")
                st.dataframe(st.session_state.create2)

                st.markdown(f"#### Predictions performed for {st.session_state.user_input}:")
                st.session_state.predict1 = reg_predict(st.session_state.create1)
                st.session_state.predict2 = reg_pull()
                st.dataframe(st.session_state.predict2)
                
                # Save model after predictions
                saved_model,name = self.save_model(st.session_state.create1)
                # Provide download button for the saved model
                self.display_download_button(saved_model,name)

        else:
            st.markdown(f"#### Created Model - {st.session_state.user_input}")
            st.dataframe(st.session_state.create2)
            st.markdown(f"#### Predictions Done - {st.session_state.user_input}")
            st.dataframe(st.session_state.predict2)
            st.markdown("#### Model is already saved")
            
            # Provide download button for the already saved model
            self.display_download_button(st.session_state.save1,st.session_state.save1FileName)

    def save_model(self, created_model):
        """Save the created model and return the saved model data."""
        if created_model is None:
            st.warning("Please create a model before saving.")
            return None
        
        # Save the model directly to session_state (avoiding complex libraries)
        if created_model == st.session_state.create1:
            st.session_state.save1FileName=st.text_input("Enter Your Desired File Name for created normal model")
            if st.session_state.save1FileName:
                st.session_state.save1 = reg_save(created_model)  # Save model to session_state
                return st.session_state.save1,st.session_state.save1FileName
        elif created_model == st.session_state.ensemble1:
            st.session_state.save2FileName=st.text_input("Enter Your Desired File Name for created ensemble model")
            if st.session_state.save2FileName:
                st.session_state.save2 = reg_save(created_model)
                return st.session_state.save2,st.session_state.save2FileName
        elif created_model == st.session_state.tune1:
            st.session_state.save2FileName=st.text_input("Enter Your Desired File Name for created tuned model")
            if st.session_state.save3FileName:
                st.session_state.save3 = reg_save(created_model)
                return st.session_state.save3,st.session_state.save3FileName
        elif created_model == st.session_state.tune1_ensemble:
            st.session_state.save2FileName=st.text_input("Enter Your Desired File Name for created ensembled tuned model")
            if st.session_state.save4FileName:
                st.session_state.save4 = reg_save(created_model)
                return st.session_state.save4,st.session_state.save4FileName
        return None,None

    def display_download_button(self, saved_model,given_file_name):
        """Display the download button for an already saved model."""
        if saved_model is not None:
            if st.button(f"Download {file_name}.pkl", key="download_existing_model_button"):
                st.download_button(
                    label="Download Model",
                    data=saved_model,
                    file_name=f"{given_file_name}.pkl",
                    mime="application/octet-stream",
                    key="download_existing_model_button"
                )

    def ensemble_model(self):
        if st.session_state.create1 is None:
            st.write("First create a model before trying to ensemble it.")
        else:
            if st.session_state.ensemble1 is None:
                st.session_state.ensemble1 = reg_ensemble(st.session_state.create1)
                st.markdown(f"#### Ensembling the model - {st.session_state.user_input}")
                st.session_state.ensemble2 = reg_pull()
                st.dataframe(st.session_state.ensemble2)
                st.markdown("#### Predictions from Ensemble Model")
                st.session_state.predict1_ensemble = reg_predict(st.session_state.ensemble1)
                st.session_state.predict2_ensemble = reg_pull()
                st.dataframe(st.session_state.predict2_ensemble)
                st.markdown("#### Saving the model")
                saved_model,name = self.save_model(st.session_state.ensemble1)
                # Provide download button for the saved model
                self.display_download_button(saved_model,name)
            else:
                st.markdown(f"#### Ensembled Model - {st.session_state.user_input}")
                st.dataframe(st.session_state.ensemble2)
                st.markdown(f"#### Predictions On Ensembled Model - {st.session_state.user_input}")
                st.dataframe(st.session_state.predict2_ensemble)
                st.markdown("#### Model is already saved")
                
                # Provide download button for the already saved ensemble model
                self.display_download_button(st.session_state.save2,st.session_state.save2FileName)

    def tune_model(self):
        if st.session_state.create1 is None:
            st.info("First create Model to tune it")
        else:
            if st.session_state.tune1 == None:
                st.session_state.tune1 = reg_tune(st.session_state.create1)
                st.session_state.tune2 = reg_pull()
                st.info("After Tuning the model the results are like")
                st.dataframe(st.session_state.tune2)
                st.info("Predictions for tuned model are like this")
                st.session_state.tune1_predict = reg_predict(st.session_state.tune1)
                st.session_state.tune2_predict = reg_pull()
                st.dataframe(st.session_state.tune2_predict)
                st.markdown("#### Saving the model")
                saved_model,name = self.save_model(st.session_state.tune1)
                # Provide download button for the saved model
                self.display_download_button(saved_model,name)
            else:
                st.markdown(f"#### Tuned Model - {st.session_state.user_input}")
                st.dataframe(st.session_state.tune2)
                st.markdown(f"#### Predicted Model - {st.session_state.user_input}")
                st.dataframe(st.session_state.tune2_predict)
                st.markdown("#### Model is already saved")
                
                # Provide download button for the already saved tuned model
                self.display_download_button(st.session_state.save3,st.session_state.save3FileName)

    def tune_ensemble(self):
        if st.session_state.ensemble1 == None:
            st.info("First Create Ensembled Model To Tune It")
        else:
            if st.session_state.tune1_ensemble == None:
                st.session_state.tune1_ensemble = reg_tune(st.session_state.ensemble1)
                st.session_state.tune2_ensemble = reg_pull()
                st.info("After tuning the ensembled model the results are like this")
                st.dataframe(st.session_state.tune2_ensemble)
                st.info("After predictions, the predictions are like this")
                st.session_state.tune1_ensemble_prredict = reg_predict(st.session_state.tune1_ensemble)
                st.session_state.tune2_ensemble_prredict = reg_pull()
                st.dataframe(st.session_state.tune2_ensemble_prredict)
                st.markdown("#### Saving the model")
                saved_model,name = self.save_model(st.session_state.tune1_ensemble)
                # Provide download button for the saved model
                self.display_download_button(saved_model,name)
            else:
                st.markdown(f"#### Already Ensemble Model is tuned - {st.session_state.user_input}")
                st.dataframe(st.session_state.tune2_ensemble)
                st.markdown(f"#### Already Predictions are done - {st.session_state.user_input}")
                st.dataframe(st.session_state.tune2_ensemble_prredict)
                st.markdown("#### Model is already saved")
                
                # Provide download button for the already saved tuned ensemble model
                self.display_download_button(st.session_state.save4,st.session_state.save4FileName)

with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    df = None

    if uploaded_file is not None:
        # Detect encoding using chardet
        raw_data = uploaded_file.read(1024)
        detected_encoding = chardet.detect(raw_data).get('encoding', 'utf-8')
        st.write(f"Detected Encoding: {detected_encoding}")
        uploaded_file.seek(0)

        # Attempt to load the file with detected encoding, falling back if necessary
        try:
            df = pd.read_csv(uploaded_file, encoding=detected_encoding)
            st.write("File loaded successfully!")
        except (UnicodeDecodeError, pd.errors.ParserError):
            st.warning(f"Failed to load file with encoding '{detected_encoding}'. Trying 'ISO-8859-1'.")
            try:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                st.write("File loaded successfully with ISO-8859-1 encoding!")
            except Exception as e:
                st.error("Unable to load file. Please check the file format and encoding.")
                st.error(f"Error details: {e}")

        # Check if DataFrame is empty after loading
        if df is not None and df.empty:
            st.warning("The uploaded file is empty or only contains headers. Please upload a valid CSV file.")
            df = None  # Reset if empty

    # Option menu for Machine Learning choices
    ml_menu = option_menu(
        "Machine Learning",
        options=["Regression", "Time-series", "Classification", "Clustering", "Anomaly-Detection", "Profiling Dashboard", "Predictions"],
        default_index=0,
        menu_icon="robot",
    )

# Main layout based on menu selections
st.title("Data Science App")

if df is not None:
    if ml_menu == "Regression":
        st.session_state.dataset = df
        regression_object = Regression(st.session_state.dataset)
        regression_object.layout()
        
    elif ml_menu == "Profiling Dashboard":
        st.subheader("Profiling Dashboard")
        try:
            profile = ProfileReport(df, explorative=True)
            st_profile_report(profile)
        except Exception as e:
            st.error("Error generating the profiling report.")
            st.error(f"Error details: {e}")
            
    elif ml_menu == "Predictions":
        st.subheader("Predictions")
        st.write("Prediction functionality is under development.")
else:
    st.write("Please upload a CSV file to start.")
