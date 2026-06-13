import os
import warnings
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

# Conexão inteligente: pega a URI do Docker se existir, senão usa localhost (fallback)
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Prisma_key_kF")

df = pd.read_csv("base_tratada.csv")

colunas_alvo = ['total_score', 'total_grade', 'total_level']
X = df.drop(columns=[col for col in colunas_alvo if col in df.columns] + ['name'], errors='ignore')
y = df['total_score']

categorical_features = ['industry', 'environment_grade', 'environment_level', 
                        'social_grade', 'social_level', 'governance_grade', 'governance_level']
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌟 CORREÇÃO: Valores fixos para o modelo base (Sem listas)
knn_params = {
    "n_neighbors": 5,
    "weights": "distance",
    "metric": "manhattan",
    "p": 1
}

knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor(**knn_params))
])

# 🌟 CORREÇÃO: Ativar o autolog ANTES do fit para capturar os dados corretamente
mlflow.sklearn.autolog(log_models=False) # Evita duplicar o artefato do modelo final

with mlflow.start_run(run_name="KNN_Regressor_Score"):
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores = cross_val_score(knn_pipeline, X_train, y_train, cv=kf, scoring='r2')
    cv_r2_mean = cv_r2_scores.mean()
    
    # O treinamento agora é monitorado pelo autolog automaticamente
    knn_pipeline.fit(X_train, y_train)
    
    train_preds = knn_pipeline.predict(X_train)
    test_preds = knn_pipeline.predict(X_test)
    
    tr_rmse = root_mean_squared_error(y_train, train_preds)
    tr_mae = mean_absolute_error(y_train, train_preds)
    tr_r2 = r2_score(y_train, train_preds)
    
    te_rmse = root_mean_squared_error(y_test, test_preds)
    te_mae = mean_absolute_error(y_test, test_preds)
    te_r2 = r2_score(y_test, test_preds)
    
    # Logs manuais complementares
    mlflow.log_params(knn_params)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_metric("cv_mean_r2", cv_r2_mean)
    mlflow.log_metric("tr_rmse", tr_rmse)
    mlflow.log_metric("tr_mae", tr_mae)
    mlflow.log_metric("tr_r2", tr_r2)
    mlflow.log_metric("rmse", te_rmse)
    mlflow.log_metric("mae", te_mae)
    mlflow.log_metric("r2_score", te_r2)
    
    mlflow.sklearn.log_model(
        sk_model=knn_pipeline,
        name="knn_score_model",
        serialization_format="skops",
        skops_trusted_types=["scipy.sparse._csr.csr_matrix"],
        input_example=X_train.head(3)
        )
    
    # GERAÇÃO DA CURVA DE APRENDIZAGEM
    print("\nGerando Curva de Aprendizagem para o modelo de Regressão...")

    # Define tamanhos progressivos do dataset (de 10% a 100%)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=knn_pipeline,  
        X=X_train,               
        y=y_train,               
        cv=kf,                  
        scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    # Médias e desvios padrões
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Construção do Gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', label='R² de Treino')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', label='R² de Validação (CV)')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.title('Curva de Aprendizagem - Regressor (Previsão de Score)')
    plt.xlabel('Tamanho da Base de Treinamento')
    plt.ylabel('Métrica R²')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.01])  # Foco na zona de alta performance

    # Salva e envia o artefato para o servidor do MLflow
    plot_path_reg = "learning_curve_regression.png"
    plt.savefig(plot_path_reg, bbox_inches='tight')
    plt.close()
    
    mlflow.log_artifact(plot_path_reg)
    print(f"Grafico salvo com sucesso no MLflow: {plot_path_reg}")

    print("\n--- PERFORMANCE DO KNN REGRESSOR ---")
    print(f"CV R² Médio: {cv_r2_mean:.4f}")
    print(f"Teste RMSE: {te_rmse:.2f} | Teste MAE: {te_mae:.2f} | Teste R²: {te_r2:.4f}")