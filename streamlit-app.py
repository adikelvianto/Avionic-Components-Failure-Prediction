import pandas as pd 
import numpy as np
import pickle
import streamlit as st

# VOR Data Read
df_VOR = pd.read_csv('VOR Train Test.csv')
df_VOR = df_VOR.drop('ORDER',axis=1)

Features_VOR = df_VOR.iloc [:,0:-1].values      # Feature Matrix / Independent Variables
Targets_VOR, Values_VOR = pd.factorize(df_VOR["FAULTY PARTS"])      # Target Matrix / Dependent Variable

# MMC Data Read
df_MMC = pd.read_csv('Multimode Train Test.csv')
df_MMC = df_MMC.drop('ORDER',axis=1)

Features_MMC = df_MMC.iloc [:,0:-1].values      # Feature Matrix / Independent Variables
Targets_MMC, Values_MMC = pd.factorize(df_MMC["FAULTY PARTS"])      # Target Matrix / Dependent Variable

#Prediction function for VOR
def modelnb_VOR(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,19)
    loaded_modelnb_VOR = pickle.load(open("modelnb_VOR.pkl","rb"))
    resultnb_VOR_num = loaded_modelnb_VOR.predict(to_predict)
    resultnb_VOR = Values_VOR[resultnb_VOR_num]
    return resultnb_VOR [0]

def modelknn_VOR(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,19)
    loaded_modelknn_VOR = pickle.load(open("modelknn_VOR.pkl","rb"))
    resultknn_VOR_num = loaded_modelknn_VOR.predict(to_predict)
    resultknn_VOR = Values_VOR[resultknn_VOR_num]
    return resultknn_VOR [0]

def modellr_VOR(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,19)
    loaded_modellr_VOR = pickle.load(open("modellr_VOR.pkl","rb"))
    resultlr_VOR_num = loaded_modellr_VOR.predict(to_predict)
    resultlr_VOR = Values_VOR[resultlr_VOR_num]
    return resultlr_VOR [0]

#Prediction function for MMC
def modellr_MMC(to_predict_list_MMC):
    to_predict_MMC = np.array(to_predict_list_MMC).reshape(1,23)
    loaded_modellr_MMC = pickle.load(open("modellr_mmr.pkl","rb"))
    resultlr_MMC_num = loaded_modellr_MMC.predict(to_predict_MMC)
    resultlr_MMC = Values_MMC[resultlr_MMC_num]
    return resultlr_MMC [0]

def modelsvc_MMC(to_predict_list_MMC):
    to_predict_MMC = np.array(to_predict_list_MMC).reshape(1,23)
    loaded_modelsvc_MMC = pickle.load(open("modelsvc_mmr.pkl","rb"))
    resultsvc_MMC_num = loaded_modelsvc_MMC.predict(to_predict_MMC)
    resultsvc_MMC = Values_MMC[resultsvc_MMC_num]
    return resultsvc_MMC [0]

def modelnb_MMC(to_predict_list_MMC):
    to_predict_MMC = np.array(to_predict_list_MMC).reshape(1,23)
    loaded_modelnb_MMC = pickle.load(open("modelnb_mmr.pkl","rb"))
    resultnb_MMC_num = loaded_modelnb_MMC.predict(to_predict_MMC)
    resultnb_MMC = Values_MMC[resultnb_MMC_num]
    return resultnb_MMC [0]

# Styling

st.markdown(
"""
<style>
    /* Center-align content within columns */
    .centered-header {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 20px;
        padding-bottom: 50px;
        height: 20px; /* Adjust height as needed */
    }

    /* Font color for dark mode */
    body.darkMode .centered-header {
        color: white;
    }

    /* Font color for light mode */
    body.lightMode .centered-header {
        color: black;
    }
</style>
""",
    unsafe_allow_html=True
)

st.markdown(
"""
<style>
    /* Center-align content within columns */
    .centered-content {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 15px;
        padding-bottom: 35px;
        height: 20px; /* Adjust height as needed */
    }

    /* Font color for dark mode */
    body.darkMode .centered-header {
        color: white;
    }

    /* Font color for light mode */
    body.lightMode .centered-header {
        color: black;
    }
</style>
""",
    unsafe_allow_html=True
)

VOR_testing_list = df_VOR.columns[:-1].tolist()
MMC_testing_list = df_MMC.columns[:-1].tolist()


with st.sidebar:
    selected_model = st.selectbox("Select component to predict: ", ["VOR", "Multimode Control Panel"])


st.markdown('<h3 style="text-align: center; color: #50dafb;">{}</h3>'.format(selected_model), unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; padding-bottom: 70px;">Component Failure Prediction</h3>', unsafe_allow_html=True)

st.markdown('<h4 style="text-align: center; padding-bottom: 50px; color: rgba(121, 137, 216, 0.959);">Set Test</h4>', unsafe_allow_html=True)

# Function to create VOR form input
def create_VOR_form(VOR_testing_list):
    flat_mapped_values_VOR = []

    for i in range (len(VOR_testing_list)):
        col1, col2, col3 = st.columns([4, 1, 1])

        with col1:
            st.write(VOR_testing_list[i])

        with col2:
            toggle_value = st.toggle(f'toggle_VOR_{i+1}', label_visibility="hidden", key=f'toggle_VOR_{i+1}')
            flat_mapped_values_VOR.append(0 if toggle_value else 1)

        with col3:
            if toggle_value:
                st.markdown('<p style="color: red; font-size:16px;"><strong>Fail</strong></p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: green; font-size: 16px;"><strong>Pass</strong></p>', unsafe_allow_html=True)

    flat_mapped_values_VOR = [value for value in flat_mapped_values_VOR]

    print(flat_mapped_values_VOR)

    return flat_mapped_values_VOR



# Function to create MMC form input
def create_MMC_form(MMC_testing_list):
    flat_mapped_values_MMC = []
    
    for i in range (len(MMC_testing_list)):
        col1, col2, col3 = st.columns([4, 1, 1])

        with col1:
            st.write(MMC_testing_list[i])

        with col2:
            toggle_value_MMC = st.toggle(f'toggle_MMC_{i+1}', label_visibility="hidden", key=f'toggle_MMC_{i+1}')
            flat_mapped_values_MMC.append(0 if toggle_value_MMC else 1)

        with col3:
            if toggle_value_MMC:
                st.markdown('<p style="color: red; font-size:16px;"><strong>Fail</strong></p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: green; font-size: 16px;"><strong>Pass</strong></p>', unsafe_allow_html=True)

    flat_mapped_values_MMC = [value for value in flat_mapped_values_MMC]

    return flat_mapped_values_MMC


            
if selected_model == "VOR":
    VOR_testing_list = create_VOR_form(VOR_testing_list)

    def reset_toggle_VOR():
        for i in range(len(MMC_testing_list)):
            st.session_state[f'toggle_VOR_{i+1}'] = False

    # Button in the center
    col1, col2, col3, col4 = st.columns([1.7, 0.7, 0.7, 1.5])
    with col1:
        pass
    with col2:
        center_button=st.button(f"Predict")
    with col3:
        st.button('Reset', on_click=reset_toggle_VOR)
    with col4:
        pass

    st.divider()

    if center_button:
        st.markdown('<h4 style="text-align: center; padding-bottom: 50px; color: rgba(101, 137, 216, 0.959);">Result</h4>', unsafe_allow_html=True)
        header1, header2 = st.columns([2, 2])
        with header1:
            st.markdown('<div class="centered-header"><strong>Prediction Algorithm</strong></div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-content">Gaussian Naive Bayes</div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-content">Logistic Regression</div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-content">K-Nearest Neighbors</div>', unsafe_allow_html=True)

        with header2:
            st.markdown('<div class="centered-header"><strong>Suspect Fail Part</strong></div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-content">{}</div>'.format(modelnb_VOR(VOR_testing_list)), unsafe_allow_html=True)
            st.markdown('<div class="centered-content">{}</div>'.format(modellr_VOR(VOR_testing_list)), unsafe_allow_html=True)
            st.markdown('<div class="centered-content">{}</div>'.format(modelknn_VOR(VOR_testing_list)), unsafe_allow_html=True)

else:
    MMC_testing_list = create_MMC_form(MMC_testing_list)
    
    def reset_toggle_MMC():
        for i in range(len(MMC_testing_list)):
            st.session_state[f'toggle_MMC_{i+1}'] = False
            
     # Button in the center
    col1, col2, col3, col4 = st.columns([1.7, 0.7, 0.7, 1.5])
    with col1:
        pass
    with col2:
        center_button=st.button(f"Predict")
    with col3:
        st.button('Reset', on_click=reset_toggle_MMC)
    with col4:
        pass

    st.divider()

    if center_button:
        st.markdown('<h4 style="text-align: center; padding-bottom: 50px; color: rgba(101, 137, 216, 0.959);">Result</h4>', unsafe_allow_html=True)
        header1, header2 = st.columns([2, 2])
        with header1:
            st.markdown('<div class="centered-header"><strong>Prediction Algorithm</strong></div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-content">Gaussian Naive Bayes</div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-content">Support Vector Machine</div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-content">Logistic Regression</div>', unsafe_allow_html=True)

        with header2:
            st.markdown('<div class="centered-header"><strong>Suspect Fail Part</strong></div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-content">{}</div>'.format(modelnb_MMC(MMC_testing_list)), unsafe_allow_html=True)
            st.markdown('<div class="centered-content">{}</div>'.format(modellr_MMC(MMC_testing_list)), unsafe_allow_html=True)
            st.markdown('<div class="centered-content">{}</div>'.format(modelsvc_MMC(MMC_testing_list)), unsafe_allow_html=True)








