# Avionic Components Failure Prediction Prototype Website

## Overview

This project is an internship assignment conducted at PT. GMF Aeroasia Tbk over a span of 4 months. The objective was to develop a prototype website for predicting the failure of avionic components, specifically VHF Omnidirectional Range (VOR) and Multimode Control Panel components. The application utilizes machine learning methods implemented with scikit-learn library, and the web interface is built using HTML, CSS, and Flask.

## Project Highlights

- **Datasets Creation:**
  - Collected maintenance information from "Internal Component Refurbishment" documents for both VOR (19 sub-tests) and Multimode Control Panel (23 sub-tests) components.
  - Created two datasets based on the collected information:
    - [VOR Dataset](https://github.com/adikelvianto/Avionic-Components-Failure-Prediction/blob/main/VOR%20Train%20Test.csv)
    - [Multimode Control Panel Dataset](https://github.com/adikelvianto/Avionic-Components-Failure-Prediction/blob/main/Multimode%20Train%20Test.csv).
  - Splitted the dataset into training, validation, and test sets.

- **Machine Learning Model Selection:**
  - Utilized "Nested Cross Validation" to determine the best hyperparameters for each machine learning algorithm.
  - Considered algorithms such as Decision Tree, Random Forest, Gradient Boosting, Gaussian Naive Bayes, KNN, and MLP.

- **Algorithm Deployment:**
  - Selected the top 3 performing algorithms for each component based on accuracy on the test set.
  - Implemented the selected algorithms in the Flask API for deployment.

- **Website Development:**
  - Structured the website with HTML, consisting of 5 pages.
  - Applied CSS for design, including a bit of JavaScript for interactive features like toggle buttons.

- **Deployment Platforms:**
  - Originally deployed on Heroku. However, due to policy changes that discontinued free hosting support, the project was subsequently redeployed in December 2023 on [PythonAnywhere](https://www.pythonanywhere.com/) for hosting. Additionally, the Streamlit app was built and hosted on [Streamlit Cloud](https://streamlit.io/cloud).
  - The application, including the Streamlit app, can be accessed at:
    - [Avionic Components Failure Prediction - PythonAnywhere](https://adikelvianto00.pythonanywhere.com/)
    - [Avionic Components Failure Prediction - Streamlit](https://avionic-components-failure-prediction.streamlit.app/).

## Getting Started

To run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/adikelvianto/Avionic-Components-Failure-Prediction.git
   cd Avionic-Components-Failure-Prediction
2. Install dependencies:
   ```bash
    pip install -r requirements.txt

3. Run Flask application:
   ```bash
    python app.py
    ```
    **or**

4. Run Streamlit application:  
    ```bash
    streamlit run streamlit-app.py