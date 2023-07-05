import pandas as p
from sklearn.tree import DecisionTreeClassifier

data_juego = p.read_csv('juegos-ml.csv')
X = data_juego.drop(columns=['juegos'])
y = data_juego['juegos']

modelo = DecisionTreeClassifier()
modelo.fit(X.values, y)

predicciones = modelo.predict([[14,0],[29,0],[30,0]])

print(predicciones)

