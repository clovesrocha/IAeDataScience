# Importando bibliotecas necessárias
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Carregar um dataset de exemplo
# Para este exemplo, usaremos um dataset hipotético
# Supondo que 'data.csv' tenha as seguintes colunas: 
# 'Transacao_Valor', 'Idade', 'Numero_Transacoes', 'Fraude'
# onde 'Fraude' é 1 para fraude e 0 para não-fraude.

data = pd.read_csv('data.csv')

# Definindo variáveis dependentes e independentes
X = data.drop('Fraude', axis=1)  # Features (Variáveis independentes)
y = data['Fraude']               # Target (Variável dependente)

# Dividindo o dataset em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criando o modelo de árvore de decisão
clf = DecisionTreeClassifier()

# Treinando o modelo
clf.fit(X_train, y_train)

# Fazendo previsões
y_pred = clf.predict(X_test)

# Avaliando o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Para visualizar a árvore, você pode usar a biblioteca graphviz ou export_graphviz
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Não Fraude', 'Fraude'], rounded=True)
plt.show()
