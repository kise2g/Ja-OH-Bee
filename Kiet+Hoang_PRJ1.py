
# coding: utf-8

# You may use this notebook for your project or you may develop your project on your own machine. Either way, be sure to submit all your code to Vocareum via this notebook or upload any code used for your project as a part of the sumbission.
# 
# If you intend to use this notebook for your report (pdf) submission; be sure to look into markdown text for any discussion you need: [Jupyter Documentation](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html)

# # Partially comprehensive guide on your Future aka (Ja Oh B)
# Dataset used:
# 
# 1. Adult-UCI Machine Learning Repository
#     - **Age**- Integer
#     - **workclass** - Categorical
#     - **education** - Categorical
#     - **education-num** - Integer
#     - **marital-status** - Categorical
#     - **sex** - Binary
#     - **income** - Targeting variables - Binary
# 2. Employee Productivity and Satisfaction HR Data
#     - **Age** - age of employee
#     - **Gender** - gender of employee
#     - **Projects Completed** - no of projects completed out of 25
#     - **Satisfaction Rate** - rate out of 100
#     - **Feedback Score** - score out of 5
# 3. Employee Turnover
#     - **stag** - Experience
#     - **event** - Employee Turnover
#     - **profession** - Employee Profession

# # Importing Libraries
#    

# In[1]:


#!pip install scipy #delete and reinstall scipy 
import numpy as np
import pandas as pd
import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression 
import seaborn as sns 
import matplotlib.pyplot as plt
import scipy.stats as stats


# # This code was ran with local machine.
# On UCI repository, the dataset was under Stata format, therefore, I performed this code, and imported the dataset in CSV to Jupyter Notebook.

# In[ ]:



file_path = ''

# Define column headers based on the apparent structure of the file
column_headers = [
    'Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-num', 
    'Marital-status', 'Occupation', 'Relationship', 'Race', 'Sex', 
    'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Native-country', 'Income'
]

# Load the data into a dataFrame with the define headers
data = pd.read_csv(file_path, header=None, names=column_headers)

# Define the path to save the new CSV
new_csv_path = ''
data.to_csv(new_csv_path, index=False)


# # Transforming data with a set of dummies variables 
# For each category in your categorical variable, create a new binary variable. This variable takes the value 1 if the observation falls into that category and 0 otherwise.

# In[3]:


adult_df = pd.read_csv('adult_dataset (1).csv')
adult_df.head(10)

missing_data = adult_df.isnull().sum()
#Converting categorical variable - Using a fun way like one-hot encoding
#defining categorical column
categorical_columns = [col for col in adult_df.columns if adult_df[col].dtype == 'object' and col != ['Income','native-country']]
adult_df['income'].replace({'<=50K':0,'>50K':1},inplace=True)
adult_df_encoded = pd.get_dummies(adult_df, columns=categorical_columns, drop_first=True)

#Spliting the data into two set of data, in-sample and out-of-sample data
train_data, test_data = train_test_split(adult_df_encoded, test_size=0.2, random_state=42)



# # Adult.csv overview

# In[4]:


#Distribution of gender
sns.countplot(x = 'sex',data=adult_df,palette='pastel')


# In[5]:


plt.figure(figsize=(10,6),)
sns.countplot(x = 'race',data=adult_df,palette='bright')
plt.xticks(rotation=0)


# In[6]:


#Comprehensive distribution of individuals on whether their income exceeded 50K
plt.figure(figsize=(20,10))

g = sns.catplot(x='race',hue='sex', col = 'income', data=adult_df,
                kind='count', palette='dark',height=8)
g.set_axis_labels("Race", "Target")
g.set_titles("{col_name}")


# 

# # Regression Analysis
# 
# - Correlation between **explanatory variables** and the **predicted outcome**. This type of correlation helps to understand the strength of the relationship between two variables.
# 
#  
# $P(y=1 | x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \ldots + \beta_n x_n)}}$
# 
# 

# In[7]:


a=['age','capital-loss','capital-gain','hours-per-week','fnlwgt']
for i in a:
    #print(a)
    print(i,':',stats.pointbiserialr(adult_df['income'],adult_df[i])[0])


# In[8]:


X_train = train_data.drop('income_1', axis=1)
y_train = train_data['income_1']
X_test = test_data.drop('income_1', axis=1)
y_test = test_data['income_1']

logistic_regression = LogisticRegression(max_iter=1000, solver='liblinear')
logistic_regression.fit(X_train, y_train)


# In[10]:


coefficients = logistic_regression.coef_[0]
feature_names = X_train.columns
intercept = logistic_regression.intercept_[0]

print(coefficients)
print(intercept)


# In[11]:


y_pred = logistic_regression.predict(X_test)

accuracy = logistic_regression.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)

accuracy, mse  


# # Key Takeaways
# 
# - *Accuracy*: 79.95%
# - *Mean Squared Error (MSE)*: 0.2005
#     - For this dataset, Xgboost, Random Forest Classification will yield higher accuracy than Logistic Regression. However, these methods require OOP which is out of my capability.

# # Turnover.csv dataset

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
turnover_df = pd.read_csv('turnover.csv',encoding='latin1')
turnover_df.head(5)
#turnover_df['profession'].unique()

turnover_df.groupby('industry')
turnover_df


# # Turnover.csv overview

# In[14]:


import seaborn as sns
#group industries
# Create a countplot to show turnover frequency by industry
plt.figure(figsize=(12, 8))
sns.countplot(data= turnover_df, x='industry', hue='event', palette='Set1')
plt.title('Frequency of Employee Turnover by Industry')
plt.xlabel('Industry')
plt.ylabel('Frequency')
plt.xticks(rotation=60, ha='right')
plt.legend(title='Turnover', labels=['No Turnover', 'Turnover'])
plt.tight_layout()

# Show the plot
plt.show()


# In[15]:


#Data Visualization
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

turnover_events = turnover_df[turnover_df['event'] == 1]
counts = turnover_events['gender'].value_counts()

plt.figure(figsize=(12, 8))
plt.pie(counts, labels=['Female','Male'], autopct='%1.1f%%', startangle=90)
plt.title('Employee Turnover by Gender')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()                   


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a distribution plot of 'age' with a hue based on 'event'
plot = sns.displot(data=turnover_df, x='age', hue='event', kind='hist', bins=20, kde=True, aspect=1.5,legend=True)
plot.set(title='Distribution of Age by Turnover Event')
plt.show()


# # Key Takeaways
# - People have the tendency to job hopping during their mid 20s until early 30s. Genders are also significant, with over 3/4 of Female ever left their jobs.
# - One of a very interesting Behavioral Economics Experiment was Niederle & Vesterlund (2007). The study examine that " Do women shy away from competition? Do men compete too much?
#     - Ability Difference? not really. Its actually because significant gender gap in decision to take risk
#         - 35% of women vs 73% of men choose to take risk.

# In[17]:


hr_df = pd.read_csv('hr_dashboard_data.csv')
hr_df.head(5)


# # Hr_dashboard_data.csv overview

# In[18]:


num_column =hr_df.select_dtypes(include = ['int','float'])


# In[19]:


num_bins = 30 

# Create subplots for each numerical column
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 15))

# Flatten the axes array for easy indexing
axes = axes.flatten()

# Iterate through numerical columns
for i, col in enumerate(num_column.columns):
    # Create histogram with specified bins
    sns.histplot(data=hr_df, x=col, bins=num_bins, palette='dark', kde=True, ax=axes[i])
    
    axes[i].set_title(f'Histogram for {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Obs')


plt.tight_layout()
plt.show()


# In[20]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=hr_df, x='Salary', y='Age', hue='Position', palette='dark')
#Fitting the plot with higher-order polynomial regression to capture non-linear, 
sns.regplot(data=hr_df, x='Salary', y='Age', scatter=False, color='blue',order=10)
#Fitting the plot with log-linear regression
sns.regplot(data=hr_df, x='Salary', y='Age', scatter=False, color='red', logx=True)
sns.regplot(data=hr_df, x='Salary', y='Age', scatter=False, color='pink',order=1)
# Adding lines to divide the plot into four quadrants
plt.axhline(y=hr_df['Age'].median(), color='black', linestyle='--', linewidth=1)
plt.axvline(x=hr_df['Salary'].median(), color='black', linestyle='--', linewidth=1)


plt.title('Scatter Plot between Salary, Age, and Gender')
plt.xlabel('Salary')
plt.ylabel('Age')
plt.legend(title='Position')
plt.show()


# # Key Takeaways
# - The histogram above was comprehensive, but it hardly indicates any trends and high variance across observation.
# - In term of Position.
#     - The graph indicated that Intern has among the lowest salary while senior developer, managing level position will yield more salary.
#     - There is a strong positive correlation between *Age* and *Salary*.
