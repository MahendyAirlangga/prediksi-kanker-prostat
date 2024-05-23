import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Membaca data
dataframe = pd.read_excel('prostate.xlsx')

# Memilih kolom-kolom yang digunakan
data = dataframe[['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45', 'lpsa', 'train']]

print("Data Awal".center(75, "="))
print(data)
print("============================================================")

# Penanganan Data Missing Value
print()
print("-----Penanganan Missing Value dengan Menghapus-----")
data1 = data.dropna()
print(data1.isna().sum())
print()

# Deteksi Outlier
print()
print("-----Deteksi Outlier-----")
outliers = []
def detect_outlier(data1):
    threshold = 3
    mean = np.mean(data1)
    std = np.std(data1)

    for x in data1:
        z_score = (x - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(x)
    return outliers

# Penanganan Outlier pada data
def handle_outliers(data1):
    median = np.median(data1)
    median_as_int = int(median)  # Cast the median value to int
    data1 = np.where(data1 > median, median_as_int, data1)
    return data1

# z-score
data = stats.zscore(data, axis= 1)
print("Hasil z-score =")
print(data)

# Grouping variabel
print("Grouping Variabel".center(75, "="))
X = data1.iloc[:, 0:9].values
y = data1.iloc[:, 9].values
print("Data Variabel".center(75, "="))
print(X)
print("Data Kelas".center(75, "="))
print(y)
print("============================================================")

# Pembagian training dan testing
print("SPLITTING DATA 20-80".center(75,"="))
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Pemodelan Decision Tree
print("MODEL DECISION TREE".center(75, "="))
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)

Y_pred_dt = decision_tree.predict(X_test)

accuracy_dt = accuracy_score(y_test, Y_pred_dt)
precision_dt = precision_score(y_test, Y_pred_dt)
recall_dt = recall_score(y_test, Y_pred_dt)
f1_score_dt = f1_score(y_test, Y_pred_dt)

print('CLASSIFICATION REPORT DECISION TREE'.center(75, '='))
print(classification_report(y_test, Y_pred_dt, zero_division=1))

cm_dt = confusion_matrix(y_test, Y_pred_dt)
print('Confusion matrix for Decision Tree\n',cm_dt)

print('Akurasi Decision Tree : ', accuracy_dt * 100, "%")
print('Precision Decision Tree : ' + str(precision_dt))
print('Recall Decision Tree : ' + str(recall_dt))
print('F1-Score Decision Tree : ' + str(f1_score_dt))
print("============================================================")

# Menampilkan the decision tree
plt.figure(figsize=(12, 8))
feature_names = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45', 'lpsa']
plot_tree(decision_tree, filled=True, feature_names=feature_names)
plt.show()
print("============================================================")
print()

# Pemodelan Random Forest
print("MODEL RANDOM FOREST".center(75, "="))
random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(X_train, y_train)

# Prediksi menggunakan model Random Forest
Y_pred_rf = random_forest.predict(X_test)
# Evaluasi model Random Forest
accuracy_rf = accuracy_score(y_test, Y_pred_rf)
precision_rf = precision_score(y_test, Y_pred_rf)
recall_rf = recall_score(y_test, Y_pred_rf)
f1_score_rf = f1_score(y_test, Y_pred_rf)

print('CLASSIFICATION REPORT RANDOM FOREST'.center(75,'='))
print(classification_report(y_test, Y_pred_rf))

cm_rf = confusion_matrix(y_test, Y_pred_rf)
print('Confusion matrix for Random Forest\n',cm_rf)

print('Akurasi Random Forest : ', accuracy_rf * 100, "%")
print('Precision Random Forest : ' + str(precision_rf))
print('Recall Random Forest : ' + str(recall_rf))
print('F1-Score Random Forest : ' + str(f1_score_rf))
print("============================================================")

#menampilkan model random forest
estimator = random_forest.estimators_[0]  # Memilih pohon keputusan pertama

# Menampilkan pohon keputusan dari model Random Forest
plt.figure(figsize=(12, 8))
feature_names = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45', 'lpsa']
plot_tree(estimator, filled=True, feature_names=feature_names)
plt.show()
print("============================================================")
print()

# Pemodelan Naive Bayes
print("MODEL NAIVE BAYES".center(75, "="))
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

Y_pred_nb = naive_bayes.predict(X_test)
accuracy_nb = accuracy_score(y_test, Y_pred_nb)
precision_nb = precision_score(y_test, Y_pred_nb)
recall_nb = recall_score(y_test, Y_pred_nb)
f1_score_nb = f1_score(y_test, Y_pred_nb)

print('CLASSIFICATION REPORT NAIVE BAYES'.center(75,'='))
print(classification_report(y_test, Y_pred_nb, zero_division=1))

print('Akurasi Naive Bayes : ', accuracy_nb * 100, "%")
print('Precision Naive Bayes : ' + str(precision_nb))
print('Recall Naive Bayes : ' + str(recall_nb))
print('F1-Score Naive Bayes : ' + str(f1_score_nb))
print("============================================================")

#perhitungan confusion matrix
cm_nb = confusion_matrix(y_test, Y_pred_nb)
print('Confusion matrix for Naive Bayes\n',cm_nb)
f, ax = plt.subplots(figsize=(8,5))

#show naive bayes
sns.heatmap(confusion_matrix(y_test, Y_pred_nb), annot=True, fmt=".0f", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#COBA INPUT
print("CONTOH INPUT".center(75, '='))
lcavol = float(input("lcavol = "))
lweight = float(input("lweight = "))
age = int(input("age = "))
lbph = float(input("lbph = "))
svi = int(input("svi = "))
lcp = float(input("lcp = "))
gleason = int(input("gleason = "))
pgg45 = int(input("pgg45 = "))
lpsa = float(input("lpsa = "))

Train = [lcavol,lweight,age,lbph,svi,lcp,gleason,
         pgg45,lpsa]
print(Train)

test = pd.DataFrame(Train).T

predtest1 = decision_tree.predict(test)
predtest2 = random_forest.predict(test)
predtest3 = naive_bayes.predict(test)

if predtest1==1:
    print("Pasien Positive (decision tree)")
else:
    print("Pasien Negative (decision tree)")

if predtest2==1:
    print("Pasien Positive (random forest)")
else:
    print("Pasien Negative (random forest)")

if predtest3==1:
    print("Pasien Positive (naive bayes)")
else:
    print("Pasien Negative (naive bayes)")
