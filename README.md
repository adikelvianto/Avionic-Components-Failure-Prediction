# Avionic Components Failure Prediction Prortype Website

This is my **internship** project at **PT. GMF Aeroasia Tbk** for about **4 months**. In this project, I created a **prototype website** to provide **failure prediction** of VHF Omnidirectional Range (VOR) and Multimode Control Panel components, by using **scikit-learn** library, **HTML, CSS, and Flask**.

## Project Tasks

- Collected information regarding maintenance from "Internal Component Refurbishment" documents both for VOR component (19 sub-tests) and Multimode Control Panel component (23 sub-tests) to **create 2 datasets**.
- **Split dataset** into train, validation, and test datasets and utilized "**Nested Cross Validation**" method to get **best hyperparameter** for each machine learning algorithm used (Decision Tree, Random Forest, Gradient Boosting, Gaussian Naive Bayes, KNN, MLP).
- **Picked 3** best performing algorithms for both components that give **best accuracy** on the test set **to be deployed** on the website.
- Created the structure of the website using HTML with a total of **5 pages**, designed using CSS, and a bit of JavaScript to allow value stored in each toggle button changed when the user gives a click.
- **Deployed** the website by using **Flask API** and the help of **Heroku** platform as a **server**.

Unfortunately, due to Heroku's new policy that does not support free hosting, the URL of the deployed website is not provided
