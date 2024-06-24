import warnings
import os
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import  train_test_split, GridSearchCV, cross_validate, validation_curve
import graphviz
from sklearn.tree import export_graphviz
from skompiler import skompile

import sqlalchemy as sa



os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

pd.set_option('display.max_columns', None)
warnings.simplefilter(action= 'ignore', category= Warning )

#Exploratory Data Analysis



#Data Preprocessing & Feature Engineering


#Modeling using CART

df = pd.read_csv("datasets/diabetes.csv")

y= df["Outcome"]
X = df.drop(["Outcome"], axis = 1)

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

#Confusion matrix için y_pred
y_pred = cart_model.predict(X)

#AUC için prob:
y_prob = cart_model.predict_proba(X)[:,1]

#Confusion Matrix
print(classification_report(y, y_pred))

#Holdout Yöntemi ile Başarı Değerlendirme

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=17)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)


#train hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

#Test Hatası
y_pred = cart_model.predict(X_test)
y_prob= cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)


###############################
#CV ile başarı değerlendirme
###############################

cart_model  = DecisionTreeClassifier(random_state= 17).fit(X, y)

cv_results = cross_validate(cart_model,
                            X, y,
                            cv=10,
                            scoring = ["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


###############################
#Hyperparameter Optimization with GriSearchCV
###############################

cart_model.get_params() #mevcut parametreleri getparams ile getirdik

cart_params = {'max_depth': range(1,11),
               "min_samples_split": range(2,20)}

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs = -1,
                              verbose = True).fit(X, y )

cart_best_grid.best_params_
cart_best_grid.best_score_

random= X.sample(1, random_state=45)

cart_best_grid.predict(random)

#############################
#Final Model
##############################

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)
cart_final.get_params()

cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)

cv_results['test_accuracy'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()


###################################
#Feature Importance
###################################

cart_final.feature_importances_ #değişkenlerin önem düzeylerini verir. Ancak şuan bunlar anlaaybileceğimiz bir formatta değğil. Öyle bir işlem yapmalıyız ki değişkein hangi önem skorua sahip olduğunu görelim

##########################################
# Visualizing the Desicion Tree
##########################################


def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled = True, out_file = None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)

tree_graph(model= cart_final, col_names= X.columns, file_name= "cart_final.png")
cart_final.get_params()


###############################
# Extracting Desicion Rules
###############################

tree_rules = export_text(cart_final, feature_names= list(X.columns))
print(tree_rules)

print(skompile(cart_final.predict).to('python/code'))  #Bunlar görsel tekniklerle elde ettiğimiz
# karar aağacımızın fonksiyonlaştırılabilecek olan kara kurallarıdır

#print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))

print(skompile(cart_final.predict).to('excel'))

########################################
#Prediction using Python Codes
########################################

def predict_with_rules (x):
    return (((((0 if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else 0 if x[6] <=
    0.5005000084638596 else 0) if x[5] <= 45.39999961853027 else 1 if x[2] <=
    99.0 else 0) if x[7] <= 28.5 else (1 if x[5] <= 9.649999618530273 else
    0) if x[5] <= 26.350000381469727 else (1 if x[1] <= 28.5 else 0) if x[1
    ] <= 99.5 else 0 if x[6] <= 0.5609999895095825 else 1) if x[1] <= 127.5
     else (((0 if x[5] <= 28.149999618530273 else 1) if x[4] <= 132.5 else
    0) if x[1] <= 145.5 else 0 if x[7] <= 25.5 else 1 if x[7] <= 61.0 else
    0) if x[5] <= 29.949999809265137 else ((1 if x[2] <= 61.0 else 0) if x[
    7] <= 30.5 else 1 if x[6] <= 0.4294999986886978 else 1) if x[1] <=
    157.5 else (1 if x[6] <= 0.3004999905824661 else 1) if x[4] <= 629.5 else 0
    )

X.columns

x= x= [12, 13, 20, 23, 4, 55, 12, 7]

predict_with_rules(x)

##########################################
# Saving and Loading Model
##########################################

joblib.dump(cart_final, "cart_final.pkl") #Geliştirilen modeli kaydettik. Gelen pkl dosyası artık bizim modelimiz

cart_model_from_disc = joblib.load("cart_final.pkl") #Kurduğum modeli kaydettikten sonra yükledim ve farklı bir isimlendirme yaptım



print(cart_model_from_disc.predict(pd.DataFrame(x).T))






