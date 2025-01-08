import pandas as pd

df = pd.read_csv('parkinsons.csv')
df.head()

features = ['MDVP:Flo(Hz)', 'NHR']
target = ['status']

x=df[features]
y=df[target]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC

model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

#Accuracy: 0.8125

import joblib

joblib.dump(model, 'my_model.joblib')
