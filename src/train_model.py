import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    print("--- Iniciando Pipeline de Machine Learning ---")

    # 1. Escolha de um conjunto de dados
    print("1. Carregando dataset Breast Cancer Wisconsin...")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    print(f"   Dados carregados: {X.shape[0]} amostras, {X.shape[1]} características.")
    print(f"   Classes: {target_names}")

    # 2. Divisão treino/teste
    # Mantemos um conjunto de teste separado para validação final
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"2. Dados divididos: {len(X_train)} treino, {len(X_test)} teste.")

    # 3. Definição do Pipeline
    # Justificativa: O Pipeline garante que o scaler seja ajustado apenas no treino
    # dentro de cada fold da validação cruzada, evitando vazamento de dados.
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Padronização é crucial para regularização
        ('classifier', LogisticRegression(solver='liblinear', max_iter=1000)) # Solver liblinear suporta l1 e l2
    ])

    # 4. Configuração da Validação Cruzada e Regularização
    # Enunciado: Aplicar adequadamente validação cruzada e regularização.
    # Vamos buscar os melhores hiperparâmetros para a regularização (C e penalty)
    param_grid = {
        'classifier__penalty': ['l1', 'l2'], # L1 (Lasso) e L2 (Ridge)
        'classifier__C': [0.01, 0.1, 1, 10, 100] # Força da regularização (menor C = maior regularização)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("3. Iniciando GridSearchCV com Validação Cruzada (5-folds)...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # Treinamento com busca de hiperparâmetros
    grid_search.fit(X_train, y_train)

    print("\n--- Resultados da Otimização ---")
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    print(f"Melhor acurácia na validação cruzada: {grid_search.best_score_:.4f}")

    # 5. Avaliação Final no Conjunto de Teste
    print("\n4. Avaliando modelo final no conjunto de teste...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia no Teste: {acc:.4f}")
    
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()

