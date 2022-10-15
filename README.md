# Reduce-Hiring-Cost
Develop AI model to predicted which employee who might leave the company on employee dataset that created by IBM Data Science Team. This model might be used to reduced training and hiring cost for an employee in the company. We applied logistic regression, random forest classifier, and keras tensor flow to predicted employee attrition

# 1. Understand the problem statement and business case

Hiring and retaining employees are extremely complex task that require capital, time, and skills.According to toggl.com/blog/cost-of-hiring-an-employee, small business owners spend 40% of their working hours on task that do not generate any income such as hiring. 
There are many studies out there that have also looked into the cost of hiring an employee. Here as some of the results :
- As stated in a study by the National Association of Colleges and Employers, hiring an employee in a company with 0-500 people costs an average of $7,645.
- Another study by the Society for Human Resource Management states that the average cost to hire an employee is $4,129, with around 42 days to fill a position.
- According to Glassdoor, the average company in the United States spends about $4,000 to hire a new employee, taking up to 52 days to fill a position.

In this project, we are going to develop a model that could predict which employees that are more likely to quit. By predicting which employees that are more likely to quit, we can prevent them to quit by giving them more allowance, pay raise, etc and it can reduces cost to hiring and training new employees.

# 2. Import Libraries and Datasets

We used employee information dataset that are created by IBM Data Science Team from kaggle. The dataset summarizes the information of 1470 employees that are stay and already quit the company with 35 informations of the employees such as age, attrition, department, and education. The following is a display of the first two rows of dataset.

| Age  | Attrition | BusinessTravel | DailyRate | Department | DistanceFromHome | Education  | EducationField | EmployeeCount | EmployeeNumber | EnvironmentSatisfaction | Gender | HourlyRate | JobInvolvement | JobLevel | JobRole | JobSatisfaction | MaritalStatus | MonthlyIncome | MonthlyRate | NumCompaniesWorked | Over18 | OverTime | PercentSalaryHike | PerformanceRating  | RelationshipSatisfied | StandardHours | StockOptionLevel | TotalWorkingYears | TrainingTimesLastYear | WorkLifeBalance | YearsAtCompany | YearsInCurrentRole | YearsSinceLastPromotion | YearsWithCurrManager |
| ------------- | ------------- | ------------ | ------------- |------------- |-------------  |------------- | ------------- |------------- |------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------ | ------------- |------------- |-------------  |------------- | ------------- |------------- |------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 41  | Yes  |Travel_Rarely  |1102  |Sales  |1   | 2  | Life Sciences  |1  |1   |2  |Female   | 94  | 3  |2  |Sales Executive  |4  |Single  | 5993 | 19479 |8  |Y   |Yes  |11   | 3  | 1  | 80  |0  |8  |0  | 1 | 6 | 4  | 0  | 5  |
|49 | No  | Travel_Frequently  | 279  | Research & Development  | 8 |1  | Life Sciences | 1  | 2  | 3  | Male  | 61  | 2 | 2  | Research Scientist | 2  | Married  | 5130  | 24907 | 1  | Y  | No | 23  | 4  | 4 | 80  | 1  | 10  | 3  | 3  | 10 | 7  | 1  | 7  |

There are categorical data in some of the columns in the dataset, the following is explanation of each columns :

- Education 
(1 = 'Below College'
2 = 'College'
3 = 'Bachelor'
4 = 'Master'
5 = 'Doctor')

- EnvironmentSatisfaction 
(1 = 'Low'
2 = 'Medium'
3 = 'High'
4 = 'Very High')

- JobInvolvement
(1 = 'Low'
2 = 'Medium'
3 = 'High'
4 = 'Very High')

- JobSatisfaction
(1 = 'Low'
2 = 'Medium'
3 = 'High'
4 = 'Very High')

- PerformanceRating
(1 = 'Low'
2 = 'Good'
3 = 'Excellent'
4 = 'Outstanding')

- RelationshipSatisfaction
(1 = 'Low'
2 = 'Medium'
3 = 'High'
4 = 'Very High')

- WorkLifeBalance
(1 = 'Bad'
2 = 'Good'
3 = 'Better'
4 = 'Best')

# 3. Data Visualization
Before visualizing the data, we do the following steps :
- Checking if there any missing or null values in the dataset, fortunately we dont have any missing values
- Replace 'Attrition', 'OverTime', 'Over18' columns with integer before performing any visualization ( 1 = Yes and 0 = No)
- Make two new datasets that consist of employee that stay and employee that already quit

## Data Distribution Plot
We check data distribution for each feature using descriptive statistics, histogram, and boxplot.

![Data Vis 1](https://user-images.githubusercontent.com/107464383/195823551-a39a1296-4bbf-4824-885c-d412330e0e71.PNG)
![Data Vis 2](https://user-images.githubusercontent.com/107464383/195823656-24caa083-e777-4532-ae6e-7cb235af7e29.PNG)
![Data Vis 3](https://user-images.githubusercontent.com/107464383/195823720-322de19f-83dd-40f5-a684-b0bbc51c71ed.PNG)

From the plot above, we know that several features such as 'MonthlyIncome' and 'TotalWorkingYears' are tail heavy. It makes sense to drop 'EmployeeCount' , 'Standardhours' and 'Over18' since they do not change from one employee to the other. We should drop 'EmployeeNumber' as well

![Data Vis 4](https://user-images.githubusercontent.com/107464383/195826158-507f2c8c-5c21-4ba4-82cc-86320d37bc77.PNG)
![Data Vis 5](https://user-images.githubusercontent.com/107464383/195826205-120fc658-6d64-4f10-b3f3-367183a790c3.PNG)

- Single employees tend to leave compared to married and divorced
- Sales Representitives tend to leave compared to any other job  
- Less experienced (low job level) tend to leave the company

![Data Vis 6](https://user-images.githubusercontent.com/107464383/195827027-cb3ec9fa-c054-46d2-ab4e-a782b9c6f9d0.PNG)
![Data Vis 7](https://user-images.githubusercontent.com/107464383/195827051-ca4d643f-5718-4e25-b8fa-959f805f276e.PNG)

- The difference between salary of male and female employees is not significant
- Sales representative is jobrole with lowest monthly salary
- Manager is jobrole with highest monthly salary

##Data Correlation

![Data Vis 8](https://user-images.githubusercontent.com/107464383/195828418-1caa7235-6dbf-4203-a425-645103e11dac.png)

- Job level is strongly correlated with total working hours
- Monthly income is strongly correlated with Job level
- Monthly income is strongly correlated with total working hours
- Age is stongly correlated with monthly income

# 4. CREATE TESTING AND TRAINING DATASET & PERFORM DATA CLEANING

## Remove unnecesary features
Based on the results of data understanding and exploratory data analysis, we decided not to use 'EmployeeCount' , 'Standardhours' and 'Over18' as input to the machine learning clustering model since they do not change from one employee to the other. We decided not to use 'EmployeeID' as well since 'EmployeeID' has a unique value for each employee.

## One Hot Encoding
In this dataset, we have columns with categorical data such as BusinessTravel, Department, EducationField, Gender, JobRole, and MaritalStatus. Computer can't process the data with categorical variable, so we should convert the data from categorical data into numerical data. One Hot Encoding is an encoding method. This method represenrepresents data of type category as a binary vector with integer values, 0 and 1, where all elements will be 0 except for one element that has a value of 1, namely the element that has the value of the category.

Before One Hot Encoding :
![Cat Variable](https://user-images.githubusercontent.com/107464383/195836643-f1a63415-2bfa-40a9-8bc2-9b6542a0a63c.PNG)

After One Hot Encoding :
![Cat Variable OHE](https://user-images.githubusercontent.com/107464383/195837037-e61c02c1-404d-434a-a71f-17b5da582c05.PNG)

## Data Normalization
Before we put the data into our model, we should normalize the data first.Normalization gives equal weights/importance to each variable so that no single variable steers model performance in one direction just because they are bigger numbers. We use minmax scaler to normalize the data.

Before Data Normalization :

![Before Normalization](https://user-images.githubusercontent.com/107464383/195841439-4d0b19ef-ad9a-4804-ad63-80451d138579.PNG)

After Data Normalization :

![After Normalization](https://user-images.githubusercontent.com/107464383/195841562-d9d886e4-2552-409d-8c7e-dd41398e82a6.PNG)

# Train and Evaluate a Logistic Regression Classifier

## Train Logistic Regression Classifier
We split the dataset into X Train, X Test, Y Train, and Y Test. X is consist of all columns in dataset except for attrition columns because attrition columns is data that we want to predict. Y is attrition column. We used 80% of our data into training dataset and 20% of our data into testing dataset.

The following is the result of our model using logistic regression classifier :
![Y pred 1](https://user-images.githubusercontent.com/107464383/195975027-58433b99-bea1-4b87-947e-99bb803479d3.PNG)

## Evaluate Logistic Regression Classifier
The main evaluation metric that we are used are confusion matrix and F1 score. The F-score, also called the F1-score, is a measure of a model's accuracy on a dataset. We can say F1-score is model accuracy. Confusion matrix is performance measurement for machine learning classification problem where output can be two or more classes. It is a table with 4 different combinations of predicted and actual values.

### Logistic Regression Classifier Confusion Matrix

![Logistic Regression CM](https://user-images.githubusercontent.com/107464383/195975063-2550ba4a-bf43-44a3-874a-aad7a5db935f.PNG)

Based on matrix above, we correctly classify around 2.500 employees who will stay and 8 employees who will quit. We misclassify 38 employees who will quit and 2 employees who will stay.

### Logistic Regression Classifier F1 Score

![Logistic Regression F1 Score](https://user-images.githubusercontent.com/107464383/195977593-c3629b02-4a64-44bb-b080-a1b1432a95ec.PNG)

Table above shows that our logistic regression model have F1 Score of 0.82, it means accuracy of our logistic regression model is 82%

# Train and Evaluate a Random Forest Regression Classifier

## Train Random Forest Regression Classifier
We split the dataset into X Train, X Test, Y Train, and Y Test. X is consist of all columns in dataset except for attrition columns because attrition columns is data that we want to predict. Y is attrition column. We used 80% of our data into training dataset and 20% of our data into testing dataset.

The following is the result of our model using logistic regression classifier :

![RandomForest Y Pred](https://user-images.githubusercontent.com/107464383/195977635-b03d930d-21c3-4dab-a5b1-4dc3be8d2913.PNG)

## Evaluate Random Forest Regression Classifier

### Random Forest Regression Classifier Confusion Matrix

![RandomForest CM](https://user-images.githubusercontent.com/107464383/195977754-66f2b648-9618-4bc3-b661-6ab2afa40c8b.PNG)

Based on matrix above, we correctly classify around 2.400 employee who will stay and 5 employee who will quit. We misclassify 41 employee who will quit and 3 employee who will stay.

### Random Forest Regression Classifier F1 Score

![RandomForest F1](https://user-images.githubusercontent.com/107464383/195977872-21de9c1b-88b1-495d-a0db-4ec1ed334022.PNG)

Table above shows that our logistic regression model have F1 Score of 0.80, it means accuracy of our logistic regression model is 80%

# Train and Evaluate Deep Learning Model

## Train Deep Learning Model

For our deep learning model, we used keras tensorflow. We used 4 layer to train our data with total 527.001 parameter

![Model Parameter](https://user-images.githubusercontent.com/107464383/195978298-62e5c6f1-cad6-4de6-82b3-1317abc4e3b0.PNG)

After we set our parameter, we compile our model with adam for our optimizer and accuracy for our metrics. We fit our model with 100 epochs and 50 batch size, the following is the result for our model :

![Epoch](https://user-images.githubusercontent.com/107464383/195978390-fbf325d9-e9e4-434c-8d68-f816af5eeff1.PNG)

## Evaluate Deep Learning Model

### Deep Learning Model Confusion Matrix

![Deep Learning Model CM](https://user-images.githubusercontent.com/107464383/195978652-08727fe7-9d8a-48c4-b08a-a627443af691.PNG)

Based on matrix above, we correctly classify around 2.400 employee who will stay and 14 employee who will quit. We misclassify 12 employee who will quit and 32 employee who will stay.

### Deep Learning Model F1 Score

![F1 Score Deep Learning](https://user-images.githubusercontent.com/107464383/195978783-1e45fef6-3449-4cb2-ba03-226da967869f.PNG)

Table above shows that our logistic regression model have F1 Score of 0.83, it means accuracy of our logistic regression model is 83%

#Conclusions

To predict which employee that might leave the company, we used three algorithm for our model. The following three algorithm are :
- Logistic regression classifier with accuracy of 82%
- Random forest regression classifier with accuracy of 80%
- Deep learning model with accuracy of 83%

We chose model with highest accuracy which is deep learning model with accuracy of 83%. Our model correctly predicted 14 employee who might leave the company. If we used the right approachment for those employee and changes their mind to leave the company, we can reduce $107.030 for our hiring and training cost according National Association of College and Employees







