# Heart Disease Prediction
This project aims to predict the likelihood of heart disease in patients with either diabetes or skin cancer using machine learning algorithms. 
The project utilizes demographic and medical data to identify factors that increase the risk of heart disease in these patients, with the ultimate goal of improving patient care through early intervention and treatment.

## Table of contents
- [Introduction](#Introduction)
- [Research Question](#Research Question)
- [Data Exploration](#Data Exploration)
- [Data Cleaning](#Data Cleaning)
- [Data Visualization](#Data Visualization)
- [Data Preprocessing](#Data Preprocessing)
- [Results and Evaluation](#Results and Evaluation)
- [Findings and Insights](#Findings and Insights)

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

### Data Cleaning
There were no missing values in the dataset. Hence, there was no need to replace the values, add any values using the mean of the attributes or drop instances from the dataset.

### Data Visualization
