import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=Warning)
from xgboost import XGBClassifier
from lightgbm import  LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,AdaBoostClassifier
from sklearn.model_selection import cross_validate,train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,accuracy_score,plot_roc_curve
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from helpers.eda import *
from helpers.data_prep import *

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

df = pd.read_csv("diabetes.csv")

check_df(df)

cat_cols,num_cols,cat_but_car = grab_col_names(df)

for col in num_cols:
    print(check_outlier(df,col))

replace_with_thresholds(df,"Insulin")

for col in num_cols:
    print(check_outlier(df, col))

## Heatmap ##

corr = df.corr()
print(corr)
sns.heatmap(corr,
         xticklabels=corr.columns,
         yticklabels=corr.columns,
         annot = True,
         cmap = "Blues")
plt.show()


## Numerical to Categorical ##

bins = [0,100,126, int(df["Glucose"].max())]
mylabels = ['No_Diabetes', 'Pre_Diabetes', 'Diabetes']

df["Glucose_Cat"] = pd.cut(df["Glucose"], bins, labels=mylabels)
df.head()


## Numerical to Categorical ##
BMI_bins = [0,18,25,30, int(df["BMI"].max())]
BMI_labels = ['Underweight', 'Healthyweight', 'Overweight', "Obese"]

df["BMI_Cat"] = pd.cut(df["BMI"], BMI_bins, labels=BMI_labels)
df.head()

## Numerical to Categorical ##

Age_bins = [0,22,40,int(df["Age"].max())]
Age_labels = ['Young_Adults', 'Middle_Aged_Adults', 'Old']

df['AGE_Cat'] = pd.cut(df["Age"],Age_bins,labels = Age_labels)


## Numerical to Categorical ##

Pregnancy_bins = [0,1,int(df["Pregnancies"].max())]
Pregnancy_labels = ['Not_pregnant_before','Pregnant_Before']
df['Pregnancy_Cat'] = pd.cut(df["Pregnancies"],Pregnancy_bins,labels = Pregnancy_labels)

cat_cols,num_cols,cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if "Outcome" not in col]
df= one_hot_encoder(df,cat_cols,drop_first=True)

## Feature Extraction ##

df.loc[(df['BloodPressure']<=80),'NEW_BLOOD_PRESSURE_CAT'] = "Normal"
df.loc[(df['BloodPressure']>80) & (df['BloodPressure']<=120),'NEW_BLOOD_PRESSURE_CAT'] = "High"
df.loc[(df['BloodPressure']>120),'NEW_BLOOD_PRESSURE_CAT'] = "Dangerous"

cat_cols,num_cols,cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if "Outcome" not in col]

df= one_hot_encoder(df,cat_cols,drop_first=True)

df.head()

X = df.drop(["Outcome"],axis = 1)
y = df["Outcome"]


######################################################
# Modeling
######################################################

classifiers = [('LR', LogisticRegression()),
               ('KNN', KNeighborsClassifier()),
               ("SVC", SVC()),
               ("CART", DecisionTreeClassifier()),
               ("RF", RandomForestClassifier()),
               ('Adaboost', AdaBoostClassifier()),
               ('GBM', GradientBoostingClassifier()),
               ('XGBoost', XGBClassifier()),
               ('LightGBM', LGBMClassifier()),
               # ('CatBoost', CatBoostClassifier(verbose=False))
               ]


for name, classifier in classifiers:
    cv_results = cross_validate(classifier, X, y, cv=3, scoring=["roc_auc"])
    print(f"AUC: {round(cv_results['test_roc_auc'].mean(),4)} ({name}) ")



######################################################
# Automated Hyperparameter Optimization
######################################################


knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 30),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500, 1000]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 20],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 0.8, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500, 1500],
                   "colsample_bytree": [0.5, 0.7, 1]}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]


best_models = {}


for name, classifier, params in classifiers:
    print(f"########## {name} ##########")
    cv_results = cross_validate(classifier, X, y, cv=3, scoring=["roc_auc"])
    print(f"AUC (Before): {round(cv_results['test_roc_auc'].mean(),4)}")


    gs_best = GridSearchCV(classifier, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)
    final_model = classifier.set_params(**gs_best.best_params_)

    cv_results = cross_validate(final_model, X, y, cv=3, scoring=["roc_auc"])
    print(f"AUC (After): {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

######################################################
# Stacking & Ensemble Learning
######################################################

voting_clf = VotingClassifier(
    estimators=[('XGBoost', best_models["XGBoost"]),
                ('RF', best_models["RF"]),
                ('LightGBM', best_models["LightGBM"])],
    voting='soft')

voting_clf.fit(X, y)

cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean() # 0.7682
cv_results['test_f1'].mean() # 0.6396
cv_results['test_roc_auc'].mean() # 0.8308

######################################################
# Prediction for a New Observation
######################################################

random_user = X.sample(1, random_state=45)

voting_clf.predict(random_user)