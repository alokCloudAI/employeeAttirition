from traceback import print_tb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


from sklearn import metrics
from sklearn.metrics import confusion_matrix



# Importing data from CSV file
df_data = pd.read_csv("/Users/alokkumar/Desktop/employeeAttirition/HR-Employee-Attrition.csv", sep=',')

# Data Analysis
df_data.head()
# df.shape()
df_data.describe(percentiles=[0.01,0.05,0.10,0.25,0.50,0.75,0.85,0.88,0.9,0.99])
df_data.info()
# df_data.isnull.sum()
# df_data.isnull.values.any()

data= df_data.drop(['Attrition'],axis=1)
target = df_data[['Attrition']]

target['Attrition'] = np.where(target['Attrition'] == 'Yes',1,0)
target.head()

# Data Cleaning
data = data.drop(['EmployeeNumber'],axis=1)
num = data.select_dtypes(include='integer')
char = data.select_dtypes(include='object')

num.head()
char.head()

def unique_levels(data):
    data = data.value_counts().count()
    return data

data_value_counts = pd.DataFrame(num.apply(lambda data: unique_levels(data)))

data_value_counts.columns = ['fetaure_levels']
data_value_counts

cat_slice = data_value_counts.loc[data_value_counts['feature_levels']<=20]
cat_list = cat_slice.index
cat = num.loc[:,cat_list]

num_slice = data_value_counts.loc[data_value_counts['feature_levels']>20]
num_list = num_slice.index
num = num.loc[:,num_list]

char = pd.concat([char,cat],axis=1,join="inner")

varselector = VarianceThreshold(threshold=0) #Less than 5 and more than 35 can remove
varselector.fit_transform(num)
num_cols = varselector.get_support(indices=True)
num_cols

char = char.drop(['EmployeeCount','StandardHours'],axis=1)

num.head()
char.head()

# outliner removal

def outlier_cap(data):
      data=data.clip(lower=data.quantile(0.01))
      data=data.clip(upper=data.quantile(0.99))
      return data

num = num.apply(lambda data : outlier_cap(data))
num.describe(percentiles=[0.01,0.05,0.10,0.25,0.50,0.75,0.85,0.88,0.9,0.99])

# Data Visualization

data_combined = pd.concat([data,target],axis=1,join='inner')

sns.set(rc = {'figure.figsize':(30,20)})
sns.heatmap(data_combined.corr(),annot=True)


def correlation(dataset,threshold):
      col_corr = set()
      corr_matrix = dataset.corr()
      for i in range(len(corr_matrix.columns)):
        for j in range(i):
         if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
      return col_corr
  
# Greater than 0.75 correlation can be removed

corr_features = correlation(data_combined,0.75)
print(corr_features)

num = num.drop(['MonthlyIncome','TotalWorkingYears'],axis=1)
char = char.drop(['PerformanceRating','YearsInCurrentRole','YearsWithCurrManager'],axis=1)

sns.set(rc = {'figure.figsize':(7,5)})


# data Transfroming

discrete=KBinsDiscretizer(n_bins=10,encode='ordinal', strategy='quantile')
num_binned=pd.DataFrame(discrete.fit_transform(num),index=num.index, columns=num.columns).add_suffix('_Rank')

num_bin_combined=pd.concat([num_binned,target],axis=1,join='inner')

from numpy import mean

for col in (num_binned.columns):
    plt.figure()
    sns.lineplot(x=col,y=num_bin_combined['Attrition'].mean(),data=num_bin_combined,color='red')
    sns.barplot(x=col,y='Attrition',data=num_bin_combined,estimator=mean )
plt.show()

num.head()

num_bin_combined.head()
num_bin_combined.groupby(by='Age_Rank')['Attrition'].sum()

cond_list = [(num_binned['Age_Rank'].isin([0.0,3.0])),
             (num_binned['Age_Rank'].isin([1.0,2.0])),
             (num_binned['Age_Rank']>=4.0)]
choice_list = ['High Attrition Age','Mid Attrition Age','Low Attrition Age']
char['Age_Category'] = np.select(cond_list,choice_list)

num_bin_combined.groupby(by='DailyRate_Rank')['Attrition'].sum()

cond_list = [(num_binned['DailyRate_Rank']==1.0),
             (num_binned['DailyRate_Rank'].isin([2.0,4.0,5.0,9.0])),
             (num_binned['DailyRate_Rank'].isin([0.0,3.0,6.0,7.0,8.0]))]
choice_list = ['High Attrition DailyRate','Mid Attrition DailyRate','Low Attrition DailyRate']
char['DailyRate_Category'] = np.select(cond_list,choice_list)


num_bin_combined.groupby(by='DistanceFromHome_Rank')['Attrition'].sum()


cond_list = [(num_binned['DistanceFromHome_Rank'].isin([6.0,8.0])),
             (num_binned['DistanceFromHome_Rank'].isin([0.0,1.0,5.0,7.0])),
             (num_binned['DistanceFromHome_Rank'].isin([2.0,3.0,4.0]))]
choice_list = ['High Attrition DistanceFromHome','Mid Attrition DistanceFromHome','Low Attrition DistanceFromHome']
char['DistanceFromHome_Category'] = np.select(cond_list,choice_list)

num_bin_combined.groupby(by='HourlyRate_Rank')['Attrition'].sum()

cond_list = [(num_binned['HourlyRate_Rank'].isin([2.0,3.0,5.0])),
             (num_binned['HourlyRate_Rank'].isin([0.0,7.0,9.0])),
             (num_binned['HourlyRate_Rank'].isin([1.0,4.0,6.0,8.0]))]
choice_list = ['High Attrition HourlyRate','Mid Attrition HourlyRate','Low Attrition HourlyRate']
char['HourlyRate_Category'] = np.select(cond_list,choice_list)

num_bin_combined.groupby(by='MonthlyRate_Rank')['Attrition'].sum()

cond_list = [(num_binned['MonthlyRate_Rank'].isin([3.0])),
             (num_binned['MonthlyRate_Rank'].isin([0.0,5.0,6.0,8.0,9.0])),
             (num_binned['MonthlyRate_Rank'].isin([1.0,2.0,4.0,7.0]))]
choice_list = ['High Attrition MonthlyRate','Mid Attrition MonthlyRate','Low Attrition MonthlyRate']
char['MonthlyRate_Category'] = np.select(cond_list,choice_list)

num_bin_combined.groupby(by='YearsAtCompany_Rank')['Attrition'].sum()


cond_list = [(num_binned['YearsAtCompany_Rank'].isin([1.0])),
             (num_binned['YearsAtCompany_Rank'].isin([2.0,3.0,4.0])),
             (num_binned['YearsAtCompany_Rank'].isin([0.0,5.0,6.0,7.0,8.0]))]
choice_list = ['High Attrition YearsAtCompany','Mid Attrition YearsAtCompany','Low Attrition YearsAtCompany']
char['YearsAtCompany_Category'] = np.select(cond_list,choice_list)


char_combined=pd.concat([char,target],axis=1,join='inner')

from numpy import mean

for col in (char.columns):
    plt.figure()
    sns.barplot(x=col, y="Attrition",data=char_combined,estimator=mean)
plt.show()

char = char.drop(['Over18','Gender'],axis=1)

char_combined.groupby(by='PercentSalaryHike')['Attrition'].sum()


cond_list = [(char_combined['PercentSalaryHike'].isin([11,12,13])),
             (char_combined['PercentSalaryHike'].isin([14,15])),
             (char_combined['PercentSalaryHike'].isin([16,17,18,19,20,21,22,23,24,25]))]
choice_list = ['High Attrition PercentSalaryHike','Mid Attrition PercentSalaryHike','Low Attrition PercentSalaryHike']
char['PercentSalaryHike_Category'] = np.select(cond_list,choice_list)

char_combined.groupby(by='YearsSinceLastPromotion')['Attrition'].sum()

cond_list = [(char_combined['YearsSinceLastPromotion'].isin([0])),
             (char_combined['YearsSinceLastPromotion'].isin([1,2,7])),
             (char_combined['YearsSinceLastPromotion'].isin([3,4,5,6,8,9,10,11,12,13,14,15]))]
choice_list = ['High Attrition YearsSinceLastPromotion','Mid Attrition YearsSinceLastPromotion','Low Attrition YearsSinceLastPromotion']
char['YearsSinceLastPromotion_Category'] = np.select(cond_list,choice_list)

char = char.drop(['PercentSalaryHike','YearsSinceLastPromotion'],axis=1)

data_all = char

data_all.head()

# Data Encoding

data_dum = pd.get_dummies(data_all, drop_first = True)
data_dum.shape
data_dum.head()

# Feature selections


selector = SelectKBest(chi2,k=40)
selector.fit_transform(data_dum,target)
cols = selector.get_support(indices=True)
data_select_features = data_dum.iloc[:,cols]
data_select_features.head()

# Train validation split

X_train,X_val,y_train,y_val=train_test_split(data_dum,target,test_size=0.3,random_state=1)

# Model Selection


lr = LogisticRegression(random_state=4)
lr.fit(X_train,y_train)

dtc = DecisionTreeClassifier(criterion='gini',random_state=4)

param_dist = {'max_depth': [3, 5, 6, 7], 'min_samples_split': [50, 100, 150, 200, 250]}
gscv_dtc = GridSearchCV(dtc, cv=10, param_grid=param_dist, n_jobs=-1)
gscv_dtc.fit(X_train,y_train)

gscv_dtc.best_params_

dtc=DecisionTreeClassifier(criterion='gini',random_state=2,max_depth=5,min_samples_split=50) # random_state =4 
dtc.fit(X_train,y_train)

knc = KNeighborsClassifier()

param_dist = {'n_neighbors': list(range(1, 31))}
gscv_knc = GridSearchCV(knc, cv=10, param_grid=param_dist, n_jobs=-1)
gscv_knc.fit(X_train,y_train)

gscv_knc.best_params_

knc=KNeighborsClassifier(n_neighbors=10)
knc.fit(X_train,y_train)
svc = SVC(kernel='rbf',random_state=4)

param_dist = {'C': [1, 10, 100, 1000], 'degree': [1, 2, 3]}
gscv_svc = GridSearchCV(svc, cv=10, param_grid=param_dist, n_jobs=-1)
gscv_svc.fit(X_train,y_train)

gscv_svc.best_params_

svc=SVC(C=10,kernel='rbf',degree=1,random_state=4)
svc.fit(X_train,y_train)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion='gini',random_state=4)

param_dist = {'max_depth': [3, 5, 6, 7], 'min_samples_split': [50, 100, 150, 200, 250]}
gscv_rfc = GridSearchCV(rfc, cv=10, param_grid=param_dist, n_jobs=-1)
gscv_rfc.fit(X_train,y_train)

gscv_rfc.best_params_

rfc=RandomForestClassifier(criterion='gini',random_state=4,max_depth=6,min_samples_split=50)
rfc.fit(X_train,y_train)

feature_importances_rfc=pd.DataFrame(rfc.feature_importances_,
                                     index=X_train.columns,
                                     columns=['importance']).sort_values('importance',ascending=False)
feature_importances_rfc

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(criterion='mse',random_state=4)

param_dist = {'max_depth': [3, 5, 6, 7], 'min_samples_split': [50, 100, 150, 200, 250] }
gscv_gbc = GridSearchCV(gbc, cv = 10, param_grid=param_dist,n_jobs = -1)
gscv_gbc.fit(X_train,y_train)

gscv_gbc.best_params_

gbc=GradientBoostingClassifier(criterion='mse',random_state=4,max_depth=6,min_samples_split=250)
gbc.fit(X_train,y_train)

feature_importances_gbc=pd.DataFrame(gbc.feature_importances_,
                                     index=X_train.columns,
                                     columns=['importance']).sort_values('importance',ascending=False)
feature_importances_gbc

base_learners = [
                 ('rfc', RandomForestClassifier(criterion='gini',random_state=4,max_depth=6,min_samples_split=50)),
                 ('gbc', GradientBoostingClassifier(criterion='mse',random_state=4,max_depth=5,min_samples_split=50))  
                ] 

from sklearn.ensemble import StackingClassifier
sc = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
sc.fit(X_train, y_train)

# Feature importance

def Model_importances(model,data):
  feature_names = data.columns
  importances = model.feature_importances_
  std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
  model_importances = pd.Series(importances,index=feature_names)
  fig, axes = plt.subplots(figsize=(15,15))
  model_importances.plot.bar(yerr=std, ax=axes)
  axes.set_title("Feature Importances")
  axes.set_ylabel("Mean decrease in impurity")
  fig.tight_layout()

Model_importances(rfc,data_dum) 

y_pred_lr=lr.predict(X_val)
y_pred_dtc=dtc.predict(X_val)
y_pred_knc=knc.predict(X_val)
y_pred_svc=svc.predict(X_val)
y_pred_rfc=rfc.predict(X_val)
y_pred_gnb=gnb.predict(X_val)
y_pred_gbc=gbc.predict(X_val)
y_pred_sc=sc.predict(X_val)
  
print('Logistic Regression')
print("Accuracy:",metrics.accuracy_score(y_val, y_pred_lr))
print("Precision",metrics.precision_score(y_val,y_pred_lr))
print("Recall",metrics.recall_score(y_val,y_pred_lr))
print("f1_score",metrics.f1_score(y_val,y_pred_lr))

metrics.plot_confusion_matrix(lr,X_val,y_val)

print('Decision Tree Classifier')
print("Accuracy:",metrics.accuracy_score(y_val, y_pred_dtc))
print("Precision",metrics.precision_score(y_val,y_pred_dtc))
print("Recall",metrics.recall_score(y_val,y_pred_dtc))
print("f1_score",metrics.f1_score(y_val,y_pred_dtc))

metrics.plot_confusion_matrix(dtc,X_val,y_val)

print('KNeighbors Classifier')
print("Accuracy:",metrics.accuracy_score(y_val, y_pred_knc))
print("Precision",metrics.precision_score(y_val,y_pred_knc))
print("Recall",metrics.recall_score(y_val,y_pred_knc))
print("f1_score",metrics.f1_score(y_val,y_pred_knc))

metrics.plot_confusion_matrix(knc,X_val,y_val)


print('Support Vector Classifier')
print("Accuracy:",metrics.accuracy_score(y_val, y_pred_svc))
print("Precision",metrics.precision_score(y_val,y_pred_svc))
print("Recall",metrics.recall_score(y_val,y_pred_svc))
print("f1_score",metrics.f1_score(y_val,y_pred_svc))

metrics.plot_confusion_matrix(svc,X_val,y_val)


print('Random Forest Classifier')
print("Accuracy:",metrics.accuracy_score(y_val, y_pred_rfc))
print("Precision",metrics.precision_score(y_val,y_pred_rfc))
print("Recall",metrics.recall_score(y_val,y_pred_rfc))
print("f1_score",metrics.f1_score(y_val,y_pred_rfc))

metrics.plot_confusion_matrix(rfc,X_val,y_val)

print('GaussianNB')
print("Accuracy:",metrics.accuracy_score(y_val, y_pred_gnb))
print("Precision",metrics.precision_score(y_val,y_pred_gnb))
print("Recall",metrics.recall_score(y_val,y_pred_gnb))
print("f1_score",metrics.f1_score(y_val,y_pred_gnb))

metrics.plot_confusion_matrix(gnb,X_val,y_val)


print('Gradient Boosting Classifier')
print("Accuracy:",metrics.accuracy_score(y_val, y_pred_gbc))
print("Precision",metrics.precision_score(y_val,y_pred_gbc))
print("Recall",metrics.recall_score(y_val,y_pred_gbc))
print("f1_score",metrics.f1_score(y_val,y_pred_gbc))

metrics.plot_confusion_matrix(gbc,X_val,y_val)

print('Stacking Classifier')
print("Accuracy:",metrics.accuracy_score(y_val, y_pred_sc))
print("Precision",metrics.precision_score(y_val,y_pred_sc))
print("Recall",metrics.recall_score(y_val,y_pred_sc))
print("f1_score",metrics.f1_score(y_val,y_pred_sc))

metrics.plot_confusion_matrix(sc,X_val,y_val)


# Deployment 

p_clf = lr
pfi_dump = pickle.dumps(p_clf)
pfi_load = pickle.loads(pfi_dump)