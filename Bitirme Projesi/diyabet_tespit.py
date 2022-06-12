# Kütüphaneleri import etme
import numpy as np
import pandas as pd
import pickle

# Data seti yüklüyoruz.
df = pd.read_csv('diabetes.csv')

# Yeniden isimlendirme
df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

# ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] öğelerindeki 0 değerleri NaN ile değiştirme
df_copy = df.copy(deep=True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# NaN değerinin ortalama ile değiştirilmesi, dağılıma bağlı olarak medyan değeri buluyoruz. Burada amaç hamilelik için değil fakat diğerleri için geçerli olan 0 değerlei yerine nan koyup
#anlamsız değerleri olasılık işlemine dahil etmemek.
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

from sklearn.model_selection import train_test_split
feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DPF','Age']
X = df[feature_cols]
y = df.Outcome 

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7)

#sklern kütüphanesinden logistik regresyonu oluşturma
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

filename = 'model.pkl'
pickle.dump(model, open('model.pkl','wb'))
