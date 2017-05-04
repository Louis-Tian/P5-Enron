
# coding: utf-8

# # Summary Report
# In this section, I will directly answer the questions that is required for completing this project. The detailed documentation is also included after this section.
# 
# #### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]
# 
# The learning problem at hand is to identifying POIs based on a set of predefined features about individuals. This is precisely what machine learning is trying to achieve, learning from existing data and try to make accurate predictions from the unseen. This POI identification problem is a typical binary classification problem. Some common algorithm used in dealing with this type of problem include:
# * Logistic Regression
# * Decision Tree (including Random Forest, Adaboosting Decision Tree)
# * Support Vector Machine
# * Naive Bayes
# 
# The dataset given includes 145 rows and 20 features (excluding names). It's generated base on two separate sources of information, a financial dataset and the famous Enron Email Corpus dataset. It is relatively easy to see how those data might be helpful. For example, it would be reasonable to assume that POIs tends to be within a social circle and hence must communicate to each other quite often. 
# 
# Other than the "TOTAL" row (which I simply drop it from the dataset), I didn't find any obvious outlier. The dataset did however has a lot of missing value (NaNs). I handled this missing value by filling the missing value with either 0 or featuren median depends on whether the features is a financial or email feature. The rationale of which is quite lengthy and can be find in the `understand the dataset` section
# 
# #### What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]
# My final list of features includes the following: 'salary', 'bonus', 'pct_msg_with_poi', 'total_stock_value', 'expenses', 'exercised_stock_options', 'deferred_income', 'short_term_incomes', and 'long_term_incentive'. 
# 
# I first manually picked six features which I believe will be important based on my reading about the scandal. Then I used a K-Best algorithm generate another list of top six features. 
#     - KBest Feature Score - 
#     exercised_stock_options    25.380102
#     total_stock_value          24.752531
#     bonus                      21.327894
#     salary                     18.861776
#     deferred_income            11.732698
#     long_term_incentive        10.222905
# 
# The final list is the combination of the both.
# 
# I engineered two features:
# 
# * __`short_term_incomes`__ I intuitive believe this could be important, as I expect POIs to receive a lot of short term income, regardless in what form.
# 
# * __`pct_msg_with_poi`__ I created this by combining the four original email features. The idea is to measures how closely someone is within the POI 'circle'. If someone is within the POI circle, then he/she must both send and receive a larger proportion of email from POI than those, not in the circle.
# 
# I preprocessed the data by using StandardScaler when using the following algorithm
# 1. Support Vector Classifier (scaling is required as SVM is not scale invariant).
# 2. Logistic Regression (scaling is used because I also used PCA, which is not scaled invariant)
# 
# #### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]
# I have tried following algorithms:
#     * Simple Decision Tree
#     * Random Forest 
#     * Adaboosting Decision Tree
#     * Logit Regression
#     * Support Vector Machine
#     * Naive Bayes
#     
# I selected Random Forest to be the final algorithm because it performed significantly better than other algorithm and the performance is very consistent across training and validation dataset. 
# 
# #### What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]
# 
# I used `GridSearchCV` to systematically tune the parameters on all of the algorithms (except Naive Bayes which don't have any parameter to be tuned).
# 
# Using Probably Approximately Correct Learning's terminology, machine learning can be fundamentally decomposed into two subtasks, learning the concept model class and learning the best concept within the class.
# * The first step is about forming a hypothesis about concept class, a family of possible models. The different algorithm corresponding to different concept class and hence model selection is about finding the best concept class. 
# * Given any concept class, we also need to find the best concept for the class, that is the role of parameter tuning.
# 
# Any machine learning must perform both steps to achieve any kind of learning.
# 
# 
# #### What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]
# A good machine learning algorithm must generalise. Meaning it must performance equally well on data that it haven't seen before. Validation is the step of testing the algorithm on an unseen dataset to ensure it generalises. The classic mistake is to train and validate an overfitted model based on the same set of data. 
# 
# Following common practise, I will split the data into two set, training and validation. The training data set is used in conjunction with cross-validation to find the best parameter and hyperparameter. Once the model is training and tuned, then it is validated using the validation dataset. I compare the performance of the model using the two different datasets to assess how well the model generalise. 
# 
# #### Give at least two evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]
# 
# The following is the performance metrics of my final random forest algorithm on the validation dataset.
# 
#                  precision    recall  f1-score   support
# 
#           False      1.000     0.795     0.886        39
#            True      0.385     1.000     0.556         5
# 
#     avg / total      0.930     0.818     0.848        44
#     
# The recall on positive label is 1.0, meaning all the classifier identified all of the positive labels in the validation set. (correctly identified all of the POI in the test dataset)
# 
# The precision of 0.385 on positive means the among all of the predicted positive, around 38.5% of them are correct (true postive). So by combining the two, when the classifier predicts someone to be POI, we can say we are 38.5% confident he/she is a POI.

# ---
# # Details of Process

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# # Understand the Question
# 
# #### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. 
# 
# The learning problem at hand is identify person of interest based on the given dataset. This exactly what machine learning trying to achieve, learning from existing data and try to make accurate prediction for unseem. The POI idenfication is a binary classification problem. Some typical algorithm used in dealing with this type of problem include:
# 
# * Logistic Regression
# * Decision Tree (including Random Forest, Adaboosting Decision Tree)
# * Support Vector Machine
# * Naive Bayes

# # Understanding the Dataset

# In[2]:

with open('./final_project_dataset.pkl', 'r') as fd:
    df = pd.DataFrame.from_dict(pickle.load(fd), orient='index')            .replace(['NaN'], np.nan)            .drop(['email_address'], axis='columns')

labels = df['poi'].astype(bool)
features = df.drop('poi', axis=1).astype(np.float32)

df = pd.concat([labels, features], axis=1)

df = df.drop('TOTAL') # drop the total row


# #### Number of data points and features

# In[3]:

email_features = ['to_messages', 'from_messages', 'from_this_person_to_poi', 'from_poi_to_this_person', 'shared_receipt_with_poi']
financial_features = list(set(df.columns) - set(email_features))

print '# of rows: {}'.format(df.shape[0])
print '''
# of features: {}
    - # of email features: {}
    - # of financial features: {}
'''.format(df.shape[1], len(email_features), len(financial_features))


# The dataset contains 146 data points and 20 features (excluding names).

# #### Allocation across classes (POI/non-POI)

# In[4]:

print df['poi'].value_counts()


# This dataset is highly unbalance. Only 18 examples are labelled as POI, vs 128 Non-POI.

# #### Handling Missing Value

# In[5]:

df[financial_features].info()


# In[6]:

df[email_features].info()


# In[7]:

poi_with_email_features = ((df['poi'] == True) & (~df['to_messages'].isnull())).sum() 
poi_without_email_features = ((df['poi'] == True) & (df['to_messages'].isnull())).sum() 
total_with_email_features = (~df['to_messages'].isnull()).sum() 
total_without_email_features = (df['to_messages'].isnull()).sum()

print '# of poi with email features: {}'.format(poi_with_email_features)
print '# of poi without email features: {}'.format(poi_without_email_features)

print 'Proportion of poi from individual with email features: {:.3f}'.format(poi_with_email_features / float(total_with_email_features))
print 'Proportion of poi from individual without email features: {:.3f}'.format(poi_without_email_features / float(total_without_email_features))


# This dataset is created augument the financial information with the email information extracted from Enron Email Corpus dataset. However, the financial information includes not only the Enron's employee but also some external Debtor and non-employee Directors whose email information is not included in Enron Email Corpus dataset. 
# 
# This introduces two kinds of missing value in this dataset. Those NaNs in financial information easily replace by zero. Those in emails features represents lack of information and is more difficult to deal with. 
# 
# As discussed in *Dataset and Questions* lecture, this could introduce some serious bias in our estimator. Out of the 18 POIs in the dataset, 14 has email features. This could leads our estimator to mistake missing email features as a good indicator for predicting POI. 
# 
# However, given the size of our dataset, we would like to preserve as much data as possible. As show in the calculation above, 16.3% of individuals with email features available in the dataset are POI, and only 6.8% of those without. Given the significant difference, there is a good chance of it introducing large bias is small. And for the purpose of this project, I think the most simpliest (yet reasonable) way to dealing with missing value in email features is filling the missing value with features' medium. 

# In[8]:

df[financial_features] = df[financial_features].fillna(0)

for feature in email_features:
    median = df.loc[~df[feature].isnull(), feature].median(axis='index')
    df[feature] = df[feature].fillna(value=median)


# # Outlier Investigation

# In[9]:

df[email_features].describe().T


# In[10]:

df[financial_features].describe().T


# Everything seems to be within reasonable range.

# # Feature Engineering and Selection

# In[11]:

# classsification label
y = pd.to_numeric(df['poi'])
X = df.drop(['poi'], axis='columns')

# New feature !!!
df['short_term_incomes'] = df[['salary', 'bonus', 'director_fees', 'expenses', 'other', 'exercised_stock_options']].sum(axis=1)
df['pct_msg_with_poi'] = (df['from_poi_to_this_person'] + df['from_this_person_to_poi']) / (df['to_messages'] + df['from_messages'])

# manual selected features
selected_features = ['expenses', 'bonus', 'exercised_stock_options', 'total_stock_value', 'pct_msg_with_poi', 'short_term_incomes']

# K-Best features
Kbest = SelectKBest(k=6).fit(X, y) # all features 
X_kbest = Kbest.transform(X)
kbest_features = list(pd.Series(index = X.columns, data = Kbest.scores_).sort_values(ascending=False)[0: 6].index)
print '- KBest Feature Score - '
print pd.Series(index = X.columns, data = Kbest.scores_).sort_values(ascending=False)[0: 6]


# Final Features
list_of_features = list(set(selected_features) | set(kbest_features)) # combine features
X = df.drop(['poi'], axis='columns')[list_of_features]

print '\n-- Final Features --\n{}'.format(list_of_features)


# #### Features enginerring
# __ `short_term_incomes`__
# 
# I intuitive believe this could be important, as I expect POIs to recieve a lot of short term income, regardless in what form.
# 
# __`pct_msg_with_poi`__
# 
# I created one addtional feature `pct_msg_with_poi` by combining the four original email features. The idea is to measures how closely someone is within the POI 'circle'. If some one is within the POI circle then he/she must both send and recieve a larger proportion of email from POI than those not in the circle.
# 
# I have manually pick 5 features that I believe would be most useful in predicting the POIs based on my reading of Wikipeida page on the Enron scandal. The size of the features are best on 
# 
# #### Manual Feature Selection
# __`exercised_stock_options` and `total_total_stock_value`__
# 
# The Enron executives uses deceiving accounting practise to hide debt and losses in order to keep stock price afloat, from which they recieve large benefit from execise stock options and sell stock on hand.
# 
# __`bonus` and `expenses`__
# 
# Enron is also critised for its `bonus` chasing cooperate culture, where the employee would chase for unprofitable or even loss making deal in order to recieve sizable cash bonus. And the same time, the CEO Skilling believed that if 'employees were constantly worried about cost, it would hinder original thinking', hence created extravagant expending culture as well. Both directly contributes to the fall of Enron.
# 
# __`pct_msg_with_poi`__ and __`short_term_incomes`__
# 
# Of course, I also included this feature, after all I made it for a reason.
# 
# #### K-Best
# I have also selected top 6 features using the K-Best algorithm. As it turns out, four out of four features selected using K-Best are the same as my manual select list.
# 
# ### Final List of Features
# I combined my manual selected features with those from the K-Best to create my final list of features.

# # Train Test Split and Validation

# A good machine learning algorithm must generalise. Meaning it must performance equally well on data that it haven't seem before. Validation is the step of testing the algorithm on an unseem dataset to ensure it actaully generalise.
# 
# Following common practise, I will split the data into two set. 

# In[12]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print 'trainning set:', X_train.shape
print 'test set:', X_test.shape


# # Evaluation Metrics

# Because the goal of the project to have both precision and recall above 0.3, f1, which can be considered as an average of both precision and recall, is considered as the best candidate. for our evaluation metrics. 

# In[13]:

def report(estimator):
    estimator.fit(X_train, y_train)
    best = estimator.best_estimator_
        
    print 'Best estimator:'
    print '-' * 60
    print best
    
    print '\n\nPerformance on Training Set:'
    print '-' * 60
    print classification_report(digits=3, y_true=y_train, y_pred=best.predict(X_train))
    
    print '\n\nPerformance on Validation Set:'
    print '-' * 60
    print classification_report(digits=3, y_true=y_test, y_pred=best.predict(X_test))
    
    print '\n\nPerformance on Entire DataSet:'
    print '-' * 60
    print classification_report(digits=3, y_true=y, y_pred=best.predict(X))


# # About Tuning Parameters and Model Selection
# In the following sections, I am going to explore various binary classification algorithms and use `GridSearchCV` to systematically tune the parameters.
# 
# Using Probably Approximately Correct Learning's terminology, machine learning can be fundamentally decompoased into two subtasks, learning the concept model class and learning the best concept within the class.
# * The first step is about forming hypothesis about concept class, a family of possible models. The different algoirthm corresponding to different concpet class and hence model selection is about find the best concept class. 
# * Given any concept class, we also need to find the best concept with the class, that is the role of parameter tuning.
# 
# Any machine learning must performan both steps to achieve any sort of learning. Failed at doing any of two, one will left with a bad model that doesn't generalise. 

# # Desicison Tree Classifier

# In[14]:

decisionTree = GridSearchCV(
    estimator=DecisionTreeClassifier(),            
    param_grid = {
        'max_depth': range(1, 5),
        'min_impurity_split': [0.01, 0.1, 0.3],
        'class_weight': ['balanced', { 0: 1, 1: 2 }, { 0: 1, 1: 4 }, { 0: 1, 1: 8 }]
    },
    scoring='f1',
)
    
report(decisionTree)


# The decision tree classifier reports very different performance score under training and validation data set. This model is subject to overfitting.

# # Random Forest 

# In[15]:

random_forest = GridSearchCV(
    estimator=RandomForestClassifier(random_state=0, n_estimators=10),            
    param_grid = {
        'max_depth': range(1, 5),
        'min_impurity_split': [0.01, 0.1, 0.3],
        'class_weight': ['balanced', { 0: 1, 1: 2 }, { 0: 1, 1: 4 }, { 0: 1, 1: 8 }]
    },
    scoring='f1'
)
    
report(random_forest)


# This random forest achieves an 100% recall on both test set and validation set. Furthermore, the difference in precision scores using test set and validation set are within 0.03. This model generalise pretty well, and achieved a precision and recall significantly higher than the minimium required 0.3.

# # ADA Boosting

# In[16]:

adaboosting = GridSearchCV(
    estimator=AdaBoostClassifier(random_state=0, base_estimator=DecisionTreeClassifier(), n_estimators=10),            
    param_grid = {
        'base_estimator__min_impurity_split': [0.01, 0.1, 0.3],
        'base_estimator__max_depth': range(1, 5),
        'base_estimator__class_weight': ['balanced', { 0: 1, 1: 2 }, { 0: 1, 1: 4 }, {0: 1, 1: 8}], 
        'learning_rate': [1, 2, 4, 6],
    },
    scoring='f1',
)

report(adaboosting)


# After tuning the premeters, the model is able to achieve 0.4 for both presion and recall. However, as we can see, those score are quite different than the one reported using training data set. It is likely to surfer from overfitting as well.

# # Logistic Regression

# In[17]:

logit = Pipeline(steps=[
    ('scale', StandardScaler()),
    ('pca', PCA(random_state=0)),
    ('logit', LogisticRegression()),
])

logit = GridSearchCV(
    estimator=logit,            
    param_grid = {
        'pca__n_components': range(1, 6),
        'logit__C': [0.1, 0.5, 1, 2, 4, 10, 50],
        'logit__tol': [0.0001, 0.1, 1, 10, 100],
        'logit__class_weight': [{0: 1, 1: 30}]
    },
    scoring='f1',
)

report(logit)


# Logitis regression requires independent variables to be independent of each other, which is clearly not the case in our features. Hence, I have transfered data using PCA prior fitting the model. And because PCA is not scale invariant, I have applied additional scaling before passing the data to PCA. I had to force a rather large class weight to postive labels, as my previous attempts with low class weight results zero recall for positive on validation data set, due to the dataset unbalances.

# # Naive Bayes

# In[18]:

nb = GridSearchCV(
    estimator=GaussianNB(),            
    param_grid = {},
    scoring='f1',
)

report(nb)


# The Naive Bayes classifier is performance is pretty bad when comparing to other classifier I have tried. Especially it returns a zero on the valudation set. The model simply doesn't generalise. 
# 
# And different than the logit regression, there is no parameter in Naive Bayes we can tune to improve it performance and counter the effect of the unbalance in the dataset. 

# # SVM

# In[19]:

svm = Pipeline(steps=[
    ('scale', StandardScaler()),
    ('svm', SVC(random_state=0)),
])

svm = GridSearchCV(
    estimator=svm,            
    param_grid = {
        'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'svm__C': [0.01, 0.1, 1, 10],
        'svm__degree': range(1, 2, 3),
        'svm__coef0': [0, 1, 5, 10],
        'svm__gamma': ['auto', 0.1, 1],
        'svm__class_weight': ['balanced', {0: 1, 1: 8}, {0: 1, 1: 16}],
        'svm__tol': [0.001, 0.1, 1, 5]
    },
    scoring='f1', 
)

report(svm)


# Support vector machine is commonly used for classification problem. 
# 
# Support Vector Machine algorithms are not scale invariant and known to be effective in high dimensional spaces [reference](http://scikit-learn.org/stable/modules/svm.html). Hence, I have used StandardScaler as the preprocessor to transform the data before fitting.
# 
# However, as it turns out the performance of the fitted model is rather poor on validation set. And the model doesn't generalise very well. 

# In[20]:

svm = Pipeline(steps=[
    ('scale', StandardScaler()),
    ('pca', PCA()),
    ('svm', SVC(random_state=0)),
])

svm = GridSearchCV(
    estimator=svm,            
    param_grid = {
        'pca__n_components': range(1, 5),
        'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'svm__C': [0.01, 0.1, 1, 10],
        'svm__degree': range(1, 2, 3),
        'svm__coef0': [0, 1, 5, 10],
        'svm__gamma': ['auto', 0.1, 1],
        'svm__class_weight': ['balanced', {0: 1, 1: 8}, {0: 1, 1: 16}],
        'svm__tol': [0.001, 0.1, 1, 5]
    },
    scoring='f1', 
)

report(svm)


# The performance of the estimator improved signifantly after PCA added to the pipeline. The models seems to generalise quite well. However the precision is still to low.

# # Model Selection

# With an unbeatable 100% recall, 0.385 precision, and consistent performance across test and validation dataset, the random forest classifier is undoubtly the best estimators among all explored here. 

# # Performance without engineered feautres

# In[21]:

_random_forest = GridSearchCV(
    estimator=RandomForestClassifier(random_state=0, n_estimators=10),            
    param_grid = {
        'max_depth': range(1, 5),
        'min_impurity_split': [0.01, 0.1, 0.3],
        'class_weight': ['balanced', { 0: 1, 1: 2 }, { 0: 1, 1: 4 }, { 0: 1, 1: 8 }]
    },
    scoring='f1',
)

X_train = X_train.drop(['short_term_incomes', 'pct_msg_with_poi'], axis='columns')
X_test = X_test.drop(['short_term_incomes', 'pct_msg_with_poi'], axis='columns')
X = X.drop(['short_term_incomes', 'pct_msg_with_poi'], axis='columns')

_random_forest.fit(X_train, y_train)
best = _random_forest.best_estimator_

print 'Best estimator:'
print '-' * 60
print best

print '\n\nPerformance on Training Set:'
print '-' * 60
print classification_report(digits=3, y_true=y_train, y_pred=best.predict(X_train))

print '\n\nPerformance on Validation Set:'
print '-' * 60
print classification_report(digits=3, y_true=y_test, y_pred=best.predict(X_test))

print '\n\nPerformance on Entire DataSet:'
print '-' * 60
print classification_report(digits=3, y_true=y, y_pred=best.predict(X))


# After removing the engineered feature. the performance of our estimator decreased dramatically. The recall decreased from 100% to 20%.

# # Export Result for Project Submission

# In[22]:

with open('./my_classifier.pkl', 'wb') as fd:
    pickle.dump(obj=random_forest, file=fd)
    
with open('./my_dataset.pkl', 'wb') as fd:
    pickle.dump(df.to_dict(orient='index'), file=fd)
    
with open('./my_feature_list.pkl', 'wb') as fd:
    pickle.dump(obj=['poi'] + list_of_features, file=fd) 


# In[23]:

get_ipython().magic(u'run tester.py')

