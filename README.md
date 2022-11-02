This is my **internship** project at **PT. GMF Aeroasia Tbk**. for about **4 months**. In this project, I created a **prototype website** to give **failure prediction** of VHF Omnidirectional Range (VOR) and Multimode Control Panel components, by using **scikit-learn** library, **HTML, CSS and Flask.**

What I've done on this project are: 
* Collected information regarding maintenance from "Internal Component Refurbishment" documents both for VOR component (19 sub test) and Multimode Control Panel component (23 sub test) to **create 2 datasets**.
* **Split dataset** into train, validation and test dataset and did "**Nested Cross Validation**" method to get **best hyperparameter** for each machine learning algorithm used (Decision Tree, Random Forest, Gradient Boosting, Gaussian Naive Bayes, KNN, MLP).
* **Picked 3** best perform algorithm for both component that give **best accuracy** on test set **to be deployed** on website. 
* Created structure of the website using HTML with total of **5 pages**, designed using CSS and a bit of JavaScript to allow value stored in each toggle button changed when user give a click. 
* **Deployed** the website by using **Flask API** and help of **Heroku** platform as a **server**. 

Result of the deployed website can be found [**here**](https://avionic-failure-prediction.herokuapp.com/)
