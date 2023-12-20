import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.text import Text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from PIL import Image
from sklearn.preprocessing import LabelEncoder  # Tambahkan baris ini

st.title("Aplikasi Machine Learning Prediksi Resiko Stroke")
# Ouvre l'image à partir du fichier
image = Image.open('img/stroke.png')
st.image(image)



# Load the data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Supprimer les valeurs manquantes de l'atribue 'bmi' et les remplacer par la moyenne
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# Supprimer la colonne ID
df.drop(columns=["id"], inplace=True)


# Show the data
st.subheader('Data Information')
st.dataframe(df)

# Label Encoding for categorical variables
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])
df['Residence_type'] = label_encoder.fit_transform(df['Residence_type'])
df['work_type'] = label_encoder.fit_transform(df['work_type'])
df['ever_married'] = label_encoder.fit_transform(df['ever_married'])
df['smoking_status'] = label_encoder.fit_transform(df['smoking_status'])


# Split data into features (X) and labels (y)
X = df.drop('stroke', axis=1)
y = df['stroke']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

# Plot 1: Stroke vs Age
sns.kdeplot(data=df, x='age', hue='stroke', ax=axes[0])
axes[0].set_title('Stroke vs Age')
axes[0].legend(['No stroke', 'Stroke'])

# Plot 2: Heart Disease Distribution
sns.countplot(x=df['heart_disease'], ax=axes[1])
axes[1].set_title('Heart Disease Distribution')
axes[1].set_xticklabels(['No heart disease', 'Heart disease'])

# Adjust layout
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)

# Evaluate the model
acc = accuracy_score (y_test, y_pred)
prec = precision_score (y_test, y_pred)
recall = recall_score (y_test, y_pred)

# Show the results
st.subheader('Model Evaluasi')
st.write('Akurasi: ', acc)
st.write('Presisi: ', prec)
st.write('Ingatan: ', recall)

# Get input from the user
st.subheader('Prediksi Stroke')
gender = st.sidebar.selectbox ('Jenis Kelamin', ('Laki-laki','Perempuan'))
gender = 1 if gender=='Laki-laki'else 0
age =st.sidebar.slider ('Umur',0,85)
hypertension =st.sidebar.selectbox ('Hipertensi', ('Tidak','Ya'))
hypertension = 0 if hypertension=='Tidak' else 1
heart_disease =st.sidebar.selectbox ('Penyakit Jantung', ('Tidak','Ya'))
heart_disease = 0 if heart_disease=='Tidak' else 1
ever_married =st.sidebar.selectbox ('Menikah', ('Ya','Tidak'))
ever_married = 1 if ever_married=='Ya'else 0
work_type =st.sidebar.selectbox ('Jenis Pekerjaan', ('Swasta','Wiraswasta', 'Pekerjaan_Pemerintah', 'Anak-anak','Belum_Kerja'))
if work_type == 'Swasta':
    work_type = 0
elif work_type == 'Wiraswasta':
    work_type = 1
elif work_type == 'Pekerjaan_Pemerintah':
    work_type = 2
elif work_type == 'Anak-anak':
    work_type = 3
elif work_type == 'Belum_Kerja':
    work_type = 4
Residence_type =st.sidebar.selectbox ('Tempat tinggal', ('Pedesaan','Perkotaan'))
if Residence_type == 'Pedesaan':
    Residence_type = 1
elif Residence_type == 'Perkotaan':
    Residence_type = 0
avg_glucose_level =st.sidebar.slider ('Kadar Glukosa Rata-Rata',55,272)
bmi =st.sidebar.slider ('Indeks Masa Tubuh',10,98)
smoking_status=st.sidebar.selectbox ('Status Merokok', ('Sebelumnya Merokok', 'Tidak Pernah Merokok', 'Merokok', 'Tidak Diketahui'))
if smoking_status == 'Sebelumnya Merokok':
    smoking_status = 0
elif smoking_status == 'Tidak Pernah Merokok':
    smoking_status = 1
elif smoking_status == 'Merokok':
    smoking_status = 2
elif smoking_status == 'Tidak Diketahui':
    smoking_status = 3


# Make prediction
prediction = clf.predict(np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]]))
st.write('Prediction: ', prediction)

if prediction == 1:
    st.write('Prediksi')
    st.write('Terindikasi Stroke')
else:
    st.write('Prediction')
    st.write('Tidak ada Indikasi')