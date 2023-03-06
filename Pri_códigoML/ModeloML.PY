# Meu primeiro código Machine Learning - Classificação de vinhos

# Importando bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

# importando e vizualisando dados

tabela = pd.read_csv('wine_dataset.csv')
tabela

# tratamento dos valores
tabela['style'] = tabela['style'].replace('red', 0)
tabela['style'] = tabela['style'].replace('white', 1)

# separando variáveis alvo e características

y = tabela['style']
x = tabela.drop('style', axis=1)

# criando conjuntos de treino e testes

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# criando o modelo e treinando modelo
modelo = ExtraTreesClassifier(n_estimators=100)
modelo.fit(x_treino, y_treino)

# testando modelo
percentual_acertos = modelo.score(x_teste, y_teste)
print('Percentual de acertos foi:', percentual_acertos)

# Fazendo previsões aleatórias

previsões = modelo.predict(x_teste[200:250])
previsões

y_teste[200:250]
