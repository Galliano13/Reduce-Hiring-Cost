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
We split the dataset into X Train, X Test, Y Train, and Y Test. X is consist of all columns in dataset except for attrition columns because attrition columns is data that we want to predict. Y is attrition column. We used 75% of our data into training dataset and 25% of our data into testing dataset.

The following is the result of our model using logistic regression classifier :
![Y pred 1](https://user-images.githubusercontent.com/107464383/195885368-1a74c8c9-ad0f-405f-9c0c-fc98400f9ffc.PNG)

## Evaluate Logistic Regression Classifier
The main evaluation metric that we are used are confusion matrix and F1 score. 
