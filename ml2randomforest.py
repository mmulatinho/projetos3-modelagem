import os
import warnings
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

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
y = df['total_level']

categorical_features = ['industry', 'environment_grade', 'environment_level', 
                        'social_grade', 'social_level', 'governance_grade', 'governance_level']
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🌟 CORREÇÃO: Valores fixos para o modelo base (Sem listas)
rf_params = {
    "n_estimators": 200,
    "max_depth": 7,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1
}

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(**rf_params))
])

# 🌟 CORREÇÃO: Ativar o autolog ANTES do fit para salvar matriz de confusão e curvas de avaliação
mlflow.sklearn.autolog(log_models=False)

with mlflow.start_run(run_name="RF_Classifier_Risk"):
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=skf, scoring='accuracy')
    cv_acc_mean = cv_acc_scores.mean()
    
    # O treinamento agora gera gráficos no MLflow automaticamente
    rf_pipeline.fit(X_train, y_train)
    
    train_preds = rf_pipeline.predict(X_train)
    test_preds = rf_pipeline.predict(X_test)
    
    tr_acc = accuracy_score(y_train, train_preds)
    
    te_acc = accuracy_score(y_test, test_preds)
    te_prec = precision_score(y_test, test_preds, average='macro', zero_division=0)
    te_rec = recall_score(y_test, test_preds, average='macro', zero_division=0)
    te_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)
    
    mlflow.log_params(rf_params)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_metric("cv_mean_accuracy", cv_acc_mean)
    mlflow.log_metric("tr_accuracy", tr_acc)
    mlflow.log_metric("accuracy", te_acc)
    mlflow.log_metric("precision", te_prec)
    mlflow.log_metric("recall", te_rec)
    mlflow.log_metric("f1_score", te_f1)
    
    mlflow.sklearn.log_model(
        sk_model=rf_pipeline,
        name="rf_risk_model",
        serialization_format="skops",
        skops_trusted_types=["scipy.sparse._csr.csr_matrix"],
        input_example=X_train.head(3)
        )
    
    
    # GERAÇÃO DA CURVA DE APRENDIZAGEM (CLASSIFICAÇÃO)
    print("\nGerando Curva de Aprendizagem para o modelo de Classificação...")

    # Define tamanhos progressivos do dataset (de 10% a 100%)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=rf_pipeline,
        X=X_train,
        y=y_train,
        cv=skf,
        scoring='accuracy',
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
    plt.plot(train_sizes, train_mean, color='blue', marker='o', label='Acurácia de Treino')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', label='Acurácia de Validação (CV)')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.title('Curva de Aprendizagem - Classificador de Risco ESG')
    plt.xlabel('Tamanho da Base de Treinamento')
    plt.ylabel('Acurácia')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.01])

    # Salva e envia o artefato para o servidor do MLflow
    plot_path_clf = "learning_curve_classification.png"
    plt.savefig(plot_path_clf, bbox_inches='tight')
    plt.close()
    
    mlflow.log_artifact(plot_path_clf)
    print(f"Gráfico salvo com sucesso no MLflow: {plot_path_clf}")

    print("\n--- PERFORMANCE DO RANDOM FOREST CLASSIFIER ---")
    print(f"CV Acurácia Média: {cv_acc_mean:.4f}")
    print(f"Teste Acurácia: {te_acc:.4f} | F1-Score (Macro): {te_f1:.4f}")
    print("\nRelatório Completo de Classificação:")
    print(classification_report(y_test, test_preds))