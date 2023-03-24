# Heart Disease Prediction
This project aims to predict the likelihood of heart disease in patients with either diabetes or skin cancer using machine learning algorithms. 
The project utilizes demographic and medical data to identify factors that increase the risk of heart disease in these patients, with the ultimate goal of improving patient care through early intervention and treatment.

## Table of contents
- [Introduction](#Introduction)
- [Research Question](#Research-Question)
- [Data Exploration](#Data-Exploration)
- [Data Cleaning](#Data-Cleaning)
- [Data Visualization](#Data-Visualization)
- [Data Preprocessing](#Data-Preprocessing)
- [Building Classifiers](#Building-Classifiers)
- [Results and Evaluation](#Results-and-Evaluation)
- [Findings and Insights](#Findings-and-Insights)

### Introduction
Cardiovascular disease is the main cause of mortality in both the United States and the rest of the globe. Heart disease is the leading cause of death in the United States, accounting for 655,000 deaths annually. Nearly half of individuals in the United States suffer from cardiovascular disease. It affects both genders. In fact, one in three female deaths is caused by cardiovascular disease. It impacts individuals of various ages, races, and socioeconomic statuses. High levels of blood pressure and cholesterol, smoking and alcohol consumption, an unhealthy diet full of fat, sodium, and sugar, obesity, a family long history of heart illnesses, and chronic diseases are heart disease leading factors. Chronic disease is deemed to be a key driver that accelerates the development of cardiovascular disease. Among these chronic diseases are diabetes and various types of cancer such as skin cancer.
The leading cause of mortality in persons with diabetes is cardiovascular disease (CVD), which accounts for two-thirds of all fatalities in people with type 2 diabetes. Furthermore, those with diabetes are twice as likely as those who do not have diabetes to get heart disease or a stroke than those who do not.
The National Cancer Institute reports that certain cancer therapies can harm the heart and circulatory system. Chemotherapy and radiation can induce or exacerbate high blood pressure, abnormal heart rhythms, and heart failure. This is due to the fact that cancer therapies can impact several organs, including the heart.

### Research Question
The purpose of this research is to build a classification model for diabetic and skin cancer patients that helps in precisely determining if a patient has heart disease or not. The research question derived from this objective is:
- Does a chronic disease such as skin cancer and diabetes cause heart disease?

### Data Exploration
Dataset Source: [Kaggle - Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)

The dataset consists of 400k adults from a survey which was conducted by the Centers for Disease Control and Prevention (CDC) in 2020. The dataset contains 18 variables. The attributes are mentioned in the table below:

|Attribute|Explanation|
|---------|-----------|
|Heart Disease|Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI)|
|BMI|Body Mass Index|
|Smoking|Have you smoked at least 100 cigarettes in your entire life?|
|Alcohol Drinking|Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)|
|Stroke|Ever had a stroke?|
|Physical Health|Includes physical illness and injury, for how many days during the past 30|
|Mental Health| How many days during the past 30 days was your mental health not good?|
|Difficulty Walking|Do you have serious difficulty walking or climbing stairs?|
|Sex|Male or Female|
|Age-Category|Fourteen-level age category|
|Race|Imputed race/ethnicity value|
|Diabetic|Ever diagnosed with diabetes|
|Physical Activity|Adults who reported doing physical activity or exercise during the past 30 days other than their regular job|
|General Health|Health status|
|Sleep Time|On average, how many hours of sleep do you get in 24 hours?|
|Asthma|Ever diagnosed with Asthma|
|Kidney Disease|Not including kidney stones, bladder infection, or incontinence, were you ever told you had kidney disease?|
|Skin Cancer|Ever Diagnosed with Skin Cancer|

Attribute Heart Disease is the target variable in the dataset. There are 9 Boolean attributes, 5 string attributes, and 4 decimal attributes in the dataset.

Statistical Summary:

![Statistical Summary](https://github.com/SatyamedhasP/Heart_Disease_Prediction/blob/main/Project%20Images/Statistical%20summary.png)

### Data Cleaning
There were no missing values in the dataset. Hence, there was no need to replace the values, add any values using the mean of the attributes or drop instances from the dataset.

### Data Visualization
Here are few examples of data visualization to get a better understanding of the data in the heart disease dataset.

1. Heart Disease as per age catgeory:

![Heart Disease as per age catgeory](https://github.com/SatyamedhasP/Heart_Disease_Prediction/blob/main/Project%20Images/Heart%20disease%20vs%20age%20category.png)

We can interpret that the age category 80 or older and 70-74 have the highest number of heart diseases. Age groups from 65-69 and 75-79 almost have the same number of heart diseases. The lowest heart diseases are found in the categories 18-24 and 25-29

2. Patients having skin cancer:

![Patients having skin cancer](https://github.com/SatyamedhasP/Heart_Disease_Prediction/blob/main/Project%20Images/Heart%20disease%20vs%20skin%20cancer.png)

The above distribution indicates the number of skin cancer patients who were diagnosed with heart disease compared to patients who did not have skin cancer.

3. Heatmap:

![Heatmap](https://github.com/SatyamedhasP/Heart_Disease_Prediction/blob/main/Project%20Images/Heatmap.png)

Note: The remaining visualizations can be found in the project notebook.

### Data Preprocessing

1. Encoding Categorical Data:
- To build classifiers, it was necessary to process the categorical data into a numerical format. We use the ‘get_dummies’ method from the Pandas library to achieve this.
- The get dummies method works on the Boolean attributes and converts them to a numerical format such as ‘0’ or ‘1’ and so on. 
- The attributes which needed conversion were Smoking, Alcohol Drinking, Difficulty Walking, Sex, Physical Activity, Skin Cancer, and the target variable heart disease. 
- The first step in the process was to call the method on the desired attribute and drop the extra column generated by the get dummies method. 
- In the second step, we drop the initial column which consisted of the categorical information. Finally, we concatenate the data frame and the newly created numerical column for the attribute.

2. Ordinal Encoder:
- The age category column consisted of ages in multiple ranges. For Ex: 60-65, 18-24, etc. To convert this data into a numerical format we used Ordinal Encoder which helped to convert the ranges into numbers. 
- Initially, we import Ordinal Encoder from the scikit-learn library. We then reshape the column by converting it into a NumPy array. 
- Once we have reshaped the array we then fit and transform the data of the array using an Ordinal Encoder. 
- The newly created column is then concatenated to the original data frame.

3. Standard Scaling:
- A few attributes were in the numerical format but had various ranges. 
- To get them in the same range we used Standard Scaling to scale the data in the desired range. 
- The first step was to import Standard Scaler from the scikit-learn library. 
- We then fit and transform the data by passing in the desired columns of the dataset using Standard Scaler.

4. Dataset Split:
- Keeping the research questions in mind, we defined 2 datasets from the original dataset. 
- In the first dataset, we selected all the instances with skin cancer patients. 
- And the second dataset consisted of only diabetes patients. 
- In the datasets defined, we did not want other diseases such as Asthma, Kidney Disease, etc and hence we decided to drop the attributes with respect to our research questions. 
- For the skin cancer, the diabetes attribute was dropped and vice-versa for the diabetes dataset.

### Building Classifiers
- The problem at hand being a classification problem, decided to work with multiple classification machine learning algorithms. 
- The prediction algorithms used for the skin cancer patient’s dataset and diabetic patients dataset were Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Naïve Bayes, and Artificial Neural Networks. 
- To move ahead with building machine learning prediction models, the datasets were first split into training and testing datasets for the skin cancer dataset and diabetes dataset. 
- 70% of the data was set to training and 30% was defined for testing. 
- As this is a classification problem, trying to predict heart disease, several classifiers were built. 
- One common function was defined to provide the scores of the model, confusion matrix, and the classification report. 
- Passing the model and the test data as the argument in the function would provide all the desired outputs.

### Results and Evaluation
1. Evaluation Metrics:
The models were evaluated based on common essential classification performance metrics which include recall (REC), precision (PRE), f1-score, accuracy, and receiver operating characteristic curve (ROC) and area under the ROC curve (AUC).

2. Results:
- Patients with Skin Cancer

<div style="display:flex;flex-direction:row">
    <img src="https://github.com/SatyamedhasP/Heart_Disease_Prediction/blob/main/Project%20Images/Skin%20Cancer%20-%20model%20copmarison.png" width="33%" />
    <img src="https://github.com/SatyamedhasP/Heart_Disease_Prediction/blob/main/Project%20Images/Skin%20cancer%20-%20ROC.png" width="33%" />
    <img src="https://github.com/SatyamedhasP/Heart_Disease_Prediction/blob/main/Project%20Images/Skin%20cancer%20CM.png" width="33%" />
</div>

    - As per the model comparison graph, the Logistic Regression algorithm has the highest accuracy. Artificial Neural Networks algorithm has the second-highest accuracy. The logistic Regression algorithm also provides the highest precision as compared to others.
    - We can interpret from the ROC curve that the highest AUC score is shown by the Logistic Regression algorithm with a score of 0.75. The ANN model presents a very low AUC score of 0.50 as compared to the Logistic Regression AUC score of 0.75. This was interesting to observe as the accuracy of the ANN model and the Logistic Regression model were almost the same.
    - To enhance the model, used Grid Search CV. The accuracy increased to 83.42% from 83.4%.

- Patients with Diabetes

<div style="display:flex;flex-direction:row">
    <img src="https://github.com/SatyamedhasP/Heart_Disease_Prediction/blob/main/Project%20Images/Diabetes%20model%20comparison.png" width="33%" />
    <img src="https://github.com/SatyamedhasP/Heart_Disease_Prediction/blob/main/Project%20Images/ROC%20curve.png" width="33%" />
    <img src="https://github.com/SatyamedhasP/Heart_Disease_Prediction/blob/main/Project%20Images/Diabete%20CM.png" width="33%" />
</div>

- For the diabetic dataset, the highest accuracy was achieved by the Logistic Regression algorithm as well. The ANN algorithm worked with almost the same accuracy when used with Dropout and Early stopping.
- The highest precision was calculated for the ANN model.
- From the ROC curve, the highest AUC score can be seen for the Logistic Regression Algorithm with an AUC score of 0.72.
- Similar to the skin cancer dataset, the ANN model for the diabetic dataset presents a very low AUC score of 0.51 as compared to the Logistic Regression AUC score of 0.72
- To enhance the model, used Grid Search CV. The accuracy increased to 79.24% from 79.2%

### Findings and Insights
Even though the best-built models might not be optimal in terms of accuracy, there are significant insights that can be drawn from this project which are as follow:

1. Increasing the number of instances might help in getting higher-accuracy classification models.
2. Incorporating new empirically or theoretically evident features in particular those correlated to heart diseases such as family history and unhealthy diet may greatly affect the accuracy level of the models.
3. In evaluating the models, it is vital to not rely one performance metric. We see how both logistic regression and ANN classifiers have very similar accuracy scores and they greatly differ in AUC scores.
