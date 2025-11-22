# Projeto de Machine Learning - Classificação de Câncer de Mama

## Integrantes
1. Guilherme Fernandes de Freitas - RM554323
2. João Pedro Chizzolini de Freitas - RM553172

## Descrição
Este projeto implementa uma solução de Machine Learning para prever se um tumor é maligno ou benigno utilizando o dataset *Breast Cancer Wisconsin*. O projeto aplica validação cruzada e regularização para garantir a robustez do modelo.

## Estrutura do Projeto
- `src/train_model.py`: Script principal contendo o pipeline de treinamento e avaliação.
- `report.md`: Relatório técnico com a especificação da solução, estratégia e justificativas.
- `requirements.txt`: Lista de dependências do projeto.

## Como Executar

1. **Instalar Dependências**
   Certifique-se de ter o Python instalado. Execute o comando abaixo para instalar as bibliotecas necessárias:
   ```bash
   pip install -r requirements.txt
   ```

2. **Executar o Modelo**
   Navegue até a pasta raiz do projeto e execute o script:
   ```bash
   python src/train_model.py
   ```
   O script irá carregar os dados, treinar o modelo usando validação cruzada e exibir as métricas de desempenho no terminal.

