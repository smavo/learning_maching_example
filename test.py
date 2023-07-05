import pandas as p

data_juego = p.read_csv('juegos-ml.csv')
# data_juego.values
# data_juego.shape
# data_juego.describe()
X = data_juego.drop(columns=['juegos'])
y = data_juego['juegos']


print(y)

