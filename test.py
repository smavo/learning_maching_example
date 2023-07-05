import pandas as p
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_juego = p.read_csv('juegos-ml.csv')
X = data_juego.drop(columns=['juegos'])
y = data_juego['juegos']

X_entrenar, X_prueba, y_entrenar, y_prueba = train_test_split(X.values, y, test_size=0.2)

modelo = DecisionTreeClassifier()
modelo.fit(X_entrenar,y_entrenar)

# predicciones = modelo.predict([[14,0],[29,0],[30,0]])
predicciones = modelo.predict(X_prueba)

puntaje = accuracy_score(y_prueba, predicciones)

print(predicciones)
print(puntaje)
