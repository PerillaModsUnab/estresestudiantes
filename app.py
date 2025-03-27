import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Cargar dataset
@st.cache_data
def load_data():
    url = "https://github.com/PerillaModsUnab/estresestudiantes.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Preprocesamiento de datos
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Course'] = le.fit_transform(df['Course'])
df['Year'] = le.fit_transform(df['Year'])
df['Marital status'] = le.fit_transform(df['Marital status'])
df['Depression'] = le.fit_transform(df['Depression'])
df['Anxiety'] = le.fit_transform(df['Anxiety'])
df['Panic Attack'] = le.fit_transform(df['Panic Attack'])
df['Treatment'] = le.fit_transform(df['Treatment'])

# Separar datos en entrenamiento y prueba
X = df[['Gender', 'Age', 'Course', 'Year', 'Marital status']]
y = df['Depression']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred_nb)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

# Interfaz de Streamlit
st.title("Análisis del Estrés en Estudiantes Universitarios")
st.write("Este dashboard permite visualizar y analizar el estrés en estudiantes universitarios basado en el dataset de Kaggle.")

st.subheader("Vista previa del dataset")
st.dataframe(df.head())

# Visualización de datos
st.subheader("Distribución de la Edad")
fig, ax = plt.subplots()
sns.histplot(df['Age'], bins=10, kde=True, ax=ax)
st.pyplot(fig)

st.subheader("Comparación de Modelos de Machine Learning")
st.write(f"Precisión de Naive Bayes: {nb_accuracy:.2f}")
st.write(f"Precisión de Regresión Logística: {lr_accuracy:.2f}")

# Predicción con entrada del usuario
st.subheader("Predicción de Depresión en Estudiantes")
gender = st.selectbox("Género", ['Masculino', 'Femenino'])
age = st.number_input("Edad", min_value=18, max_value=30, value=20)
course = st.number_input("Curso (codificado)", min_value=0, max_value=df['Course'].nunique()-1, value=0)
year = st.number_input("Año de estudio", min_value=1, max_value=5, value=1)
marital_status = st.selectbox("Estado civil", ['Soltero', 'Casado'])

input_data = np.array([[1 if gender == 'Femenino' else 0, age, course, year, 1 if marital_status == 'Casado' else 0]])

if st.button("Predecir con Naive Bayes"):
    pred_nb = nb_model.predict(input_data)[0]
    st.write(f"Predicción (0 = No Depresión, 1 = Depresión): {pred_nb}")

if st.button("Predecir con Regresión Logística"):
    pred_lr = lr_model.predict(input_data)[0]
    st.write(f"Predicción (0 = No Depresión, 1 = Depresión): {pred_lr}")


