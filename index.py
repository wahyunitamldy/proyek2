import numpy as np
import pandas as pd
from statistics import mean
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Membaca Data
# data = pd.read_excel("kidneydisease.xlsx")
data = pd.read_csv('new_brain_stroke.csv')
print(data.head())

# Mengambil beberapa atribut / variabel
data1 = data.loc[:, ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']]
print(data1.head())

# label_encoder =LabelEncoder()
# data1['gender'] = label_encoder.fit_transform(data1['gender'])
# data1['ever_married'] = label_encoder.fit_transform(data1['ever_married'])

# MENDETEKSI DATA MISSING
print("Deteksi Missing Value")
print(data1.isna().sum())

# Penanganan Data Missing Value
## MENGHAPUS DATA MISSING VALUE
print("Penanganan Missing Value")
data_cleaned = data1.dropna()
print("Data tanpa missing value")
print(data_cleaned)

# Penanganan Data Missing Value
## MENGGANTI DATA MISSING VALUE DENGAN MEAN
print("Penanganan Missing Value 2")
data1['gender'].fillna(data1['gender'].mode()[0], inplace=True)
data1['age'].fillna(data1['age'].mean(), inplace=True)
data1['hypertension'].fillna(data1['hypertension'].mean(), inplace=True)
data1['heart_disease'].fillna(data1['heart_disease'].mean(), inplace=True)
data1['ever_married'].fillna(data1['ever_married'].mode()[0], inplace=True)
data1['work_type'].fillna(data1['work_type'].mode()[0], inplace=True)
data1['Residence_type'].fillna(data1['Residence_type'].mode()[0], inplace=True)
data1['avg_glucose_level'].fillna(data1['avg_glucose_level'].mean(), inplace=True)
data1['bmi'].fillna(data1['bmi'].mean(), inplace=True)
data1['smoking_status'].fillna(data1['smoking_status'].mode()[0], inplace=True)
print("Missing data pada gender =", data1['gender'].isna().sum())
print("Missing data pada age =", data1['age'].isna().sum())
print("Missing data pada hypertension =", data1['hypertension'].isna().sum())
print("Missing data pada heart_disease =", data1['heart_disease'].isna().sum())
print("Missing data pada ever_married =", data1['ever_married'].isna().sum())
print("Missing data pada work_type =", data1['work_type'].isna().sum())
print("Missing data pada Residence_type =", data1['Residence_type'].isna().sum())
print("Missing data pada avg_glucose_level =", data1['avg_glucose_level'].isna().sum())
print("Missing data pada bmi =", data1['bmi'].isna().sum())
print("Missing data pada smoking_status =", data1['smoking_status'].isna().sum())

# Menampilkan nilai mean setelah penanganan missing value
mean_gender = data1['gender'].value_counts()
mean_age = data1['age'].mean()
mean_hypertension = data1['hypertension'].mean()
mean_heart_disease = data1['heart_disease'].mean()
mean_ever_married = data1['ever_married'].value_counts()
mean_work_type = data1['work_type'].value_counts()
mean_Residence_type = data1['Residence_type'].value_counts()
mean_avg_glucose_level = data1['avg_glucose_level'].mean()
mean_bmi = data1['bmi'].mean()
mean_smoking_status = data1['smoking_status'].value_counts()

print("Mean untuk 'gender':", mean_gender)
print("Mean untuk 'age':", mean_age)
print("Mean untuk 'hypertension':", mean_hypertension)
print("Mean untuk 'heart_disease':", mean_heart_disease)
print("Mean untuk 'ever_married':", mean_ever_married)
print("Mean untuk 'work_type':", mean_work_type)
print("Mean untuk 'Residence_type':", mean_Residence_type)
print("Mean untuk 'avg_glucose_level':", mean_avg_glucose_level)
print("Mean untuk 'bmi':", mean_bmi)
print("Mean untuk 'smoking_status':", mean_smoking_status)

# Mendeteksi Outlier
print("Deteksi Outlier")
outliers = []

# Menampilkan nilai unik dalam kolom 'gender'
unique_values_gender = data1['gender'].unique()
print("Nilai unik dalam kolom gender : ", unique_values_gender)


def detect_outlier(data):
    # outliers = [] 
    threshold = 3
    mean_value = data.mean()
    std_dev = data.std()

    for x in data:
        z_score = (x - mean_value) / std_dev
        if np.abs(z_score) > threshold:
            outliers.append(x)
    return outliers

# Mencetak Outlier
outlier1 = detect_outlier(data1['gender'])
print("Outlier kolom gender : ", outlier1)
print("Banyak outlier gender : ", len(outlier1))
print()

outlier2 = detect_outlier(data1['age'])
print("Outlier kolom age : ", outlier2)
print("Banyak outlier age : ", len(outlier2))

outlier3 = detect_outlier(data1['hypertension'])
print("Outlier kolom hypertension : ", outlier3)
print("Banyak outlier hypertension : ", len(outlier3))
print()

outlier4 = detect_outlier(data1['heart_disease'])
print("Outlier kolom heart_disease : ", outlier3)
print("Banyak outlier heart_disease : ", len(outlier3))
print()

outlier5 = detect_outlier(data1['ever_married'])
print("Outlier kolom ever_married : ", outlier3)
print("Banyak outlier ever_married : ", len(outlier3))
print()

outlier6 = detect_outlier(data1['work_type'])
print("Outlier kolom work_type : ", outlier3)
print("Banyak outlier work_type : ", len(outlier3))
print()

outlier7 = detect_outlier(data1['Residence_type'])
print("Outlier kolom Residence_type : ", outlier3)
print("Banyak outlier Residence_type : ", len(outlier3))
print()

outlier8 = detect_outlier(data1['avg_glucose_level'])
print("Outlier kolom avg_glucose_level : ", outlier3)
print("Banyak outlier avg_glucose_level : ", len(outlier3))
print()

outlier9 = detect_outlier(data1['bmi'])
print("Outlier kolom bmi : ", outlier3)
print("Banyak outlier bmi : ", len(outlier3))
print()

outlier10 = detect_outlier(data1['smoking_status'])
print("Outlier kolom smoking_status : ", outlier3)
print("Banyak outlier smoking_status : ", len(outlier3))
print()


# Penanganan Outlier
variabel = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
variabel = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

for var in variabel:
    outlier_datapoints = detect_outlier(data1[var])
    print("Outlier ", var, " = ", outlier_datapoints)

# Penanganan Outlier untuk Mengganti outlier dengan nilai rata-rata (mean)
for var in variabel:
    outlier_datapoints = detect_outlier(data1[var])
    rata = mean(data1[var])
    data1[var] = data1[var].replace(outlier_datapoints, rata)

# Menampilkan data setelah penanganan outlier
print("Data setelah penanganan outlier:")
print(data1)

# modelkn0n
# Load dataset
X = data1[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
y = data1['stroke']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)

# Predictions on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#  model KNN
knn_model = KNeighborsClassifier(n_neighbors=3) 

# Latih model KNN menggunakan data latih80
knn_model.fit(X_train, y_train)

# Fungsi untuk menerima input dari pengguna dan melakukan prediksi
def predict_knn():
    print("Masukkan nilai untuk setiap fitur:")
    gender = float(input("GENDER: "))
    age = float(input("AGE: "))
    hypertension = float(input("HYPERTENSION: "))
    heart_disease = float(input("HEART DISEASE: "))
    ever_married = float(input("EVER MARRIED: "))
    work_type = float(input("WORK TYPE: "))
    Residence_type = float(input("RESIDENCE TYPE: "))
    avg_glucose_level = float(input("GLUCOSE LEVEL: "))
    bmi = float(input("BMI: "))
    smoking_status = float(input("SMOKING STATUS: "))

    # Buat array fitur dari input pengguna
    user_input = [[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]]

    # Lakukan prediksi menggunakan model KNN
    prediction = knn_model.predict(user_input)

    # Tampilkan hasil prediksi
    print("Prediksi Stroke:", prediction[0])

# Panggil fungsi untuk meminta input dan melakukan prediksi
predict_knn()