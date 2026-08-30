import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Crear carpeta para guardar gráficas
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

"""

ETL

"""

# Cargar dataset
RAW_PATH = Path("../data/raw/real+estate+valuation+data+set/Real_estate_valuation_data_set.xlsx")
PROCESSED_PATH = Path("../data/processed/real_estate_valuation_data_set.csv")
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(RAW_PATH)

# Verificar valores nulos
print(f"\nValores nulos:\n{df.isna().sum()}")

# Eliminar columna 'No' (identificador)
df = df.drop(columns=["No"])

# Renombrar columnas
df = df.rename(columns={
    "X1 transaction date": "transaction_date",
    "X2 house age": "house_age",
    "X3 distance to the nearest MRT station": "distance_mrt",
    "X4 number of convenience stores": "n_convenience_stores",
    "X5 latitude": "latitude",
    "X6 longitude": "longitude",
    "Y house price of unit area": "price_per_unit_area",
})

# Transformar fecha (formato decimal) a año y mes
df["transaction_year"] = df["transaction_date"].astype(int)
df["transaction_month"] = ((df["transaction_date"] - df["transaction_year"]) * 12).round().astype(int) + 1
df = df.drop(columns=["transaction_date"])

# Reordenar columnas
column_order = [
    "transaction_year", "transaction_month", "house_age", "distance_mrt",
    "n_convenience_stores", "latitude", "longitude", "price_per_unit_area",
]

df = df[column_order]

# Guardar dataset procesado
df.to_csv(PROCESSED_PATH, index=False)
print(f"Datos procesados guardados")
print(df.head())

"""

EDA

"""

df = pd.read_csv(PROCESSED_PATH)
print(f"\nShape del dataset: {df.shape}")

# Definir features y target
feature_cols = ['transaction_year', 'transaction_month', 'house_age', 'distance_mrt',
                'n_convenience_stores', 'latitude', 'longitude']
target = 'price_per_unit_area'
all_cols = feature_cols + [target]

# Boxplots de todas las variables
print("\nGenerar boxplots")
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, col in enumerate(all_cols):
    axes[i].boxplot(df[col], vert=True)
    axes[i].set_title(col, fontsize=11)
    axes[i].grid(alpha=0.3)
axes[-1].axis('off')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_boxplots_todas_variables.png", dpi=300)
plt.close()

# Análisis de correlación
print("Matriz de correlación")
corr_matrix = df[all_cols].corr()
print(f"\nCorrelación con la variable objetivo:")
print(corr_matrix[target].sort_values(ascending=False))

# Graficar matriz de correlación
fig, ax = plt.subplots(figsize=(8,7))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_matrix.columns))); ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax.set_yticks(range(len(corr_matrix.columns))); ax.set_yticklabels(corr_matrix.columns)
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        ax.text(j, i, f"{corr_matrix.iloc[i,j]:.2f}", ha='center', va='center',
                 color='white' if abs(corr_matrix.iloc[i,j])>0.5 else 'black', fontsize=8)
plt.colorbar(im)
plt.title('Matriz de correlación')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_matriz_correlacion.png", dpi=300)
plt.close()

# Scatter plots de cada variable contra objetivo
print("Scatter plots con variable objetivo")
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, col in enumerate(all_cols):
    axes[i].scatter(df[col], df[target], alpha=0.4, s=15)
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel(target if i % 4 == 0 else '', fontsize=10)
    axes[i].set_title(f"{col} vs {target}", fontsize=11)
    axes[i].grid(alpha=0.3)
axes[-1].axis('off')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_scatter_plots_todas_features.png", dpi=300)
plt.close()

# Análisis de distance_mrt con y sin logaritmo
print("Analizando distance_mrt")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(df["distance_mrt"], df[target], alpha=0.4, s=15)
axes[0].set_xlabel("distance_mrt")
axes[0].set_ylabel(target)
axes[0].set_title("Sin transformar")
axes[0].grid(alpha=0.3)

axes[1].scatter(np.log(df["distance_mrt"]), df[target], alpha=0.4, s=15, color="orange")
axes[1].set_xlabel("log(distance_mrt)")
axes[1].set_ylabel(target)
axes[1].set_title("Varaible transformada con log — relación más lineal")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_transformacion_distance.png", dpi=300)
plt.close()

"""
Transformación de Variables
"""

# Aplicar transformación logarítmica a distance_mrt
df["log_distance_mrt"] = np.log(df["distance_mrt"])

# Mostrar mejora en correlación
original_corr = df["distance_mrt"].corr(df[target])
log_corr = df["log_distance_mrt"].corr(df[target])

print(f"\nCorrelación antes de transformar (distance_mrt): {original_corr:.4f}")
print(f"Correlación después de transformar (log_distance_mrt): {log_corr:.4f}")

# Actualizar columnas de features para usar log_distance_mrt
feature_cols = [
    "transaction_year", "transaction_month", "house_age",
    "log_distance_mrt", "n_convenience_stores", "latitude", "longitude"
]

"""
Preparación de Datos para el Modelo
"""

# Separar features y target
df_x = df[feature_cols]
df_y = df[[target]]

# Shuffle y split train/test (80/20)
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(len(df_shuffled) * 0.8)

df_train = df_shuffled[:split]
df_test = df_shuffled[split:]

df_train_x = df_train[feature_cols]
df_train_y = df_train[[target]]
df_test_x = df_test[feature_cols]
df_test_y = df_test[[target]]

print(f"\ntrain shuffle: {len(df_train_x)}")
print(f"test shuffle: {len(df_test_x)}")

# Convertir a listas y agregar bias
train_samples = df_train_x.values.tolist()
train_y = df_train_y[target].tolist()

test_samples = df_test_x.values.tolist()
test_y = df_test_y[target].tolist()

# Agregar término de bias (1) a cada muestra
for i in range(len(train_samples)):
    train_samples[i] = [1] + train_samples[i]

for i in range(len(test_samples)):
    test_samples[i] = [1] + test_samples[i]

print(f"\nInstancia con bias: {train_samples[0]}")

# Estandarización Z-score
def scaling(samples):
    """
    Normaliza valores usando estandarización Z-score.
    Ignora la columna 0 (término de sesgo).
    """
    samples = np.asarray(samples, dtype=float).T.tolist()
    col_stats = []
    for i in range(1, len(samples)):
        acum = 0
        #Loop para sacar desviación estándar y promedio
        for j in range(len(samples[i])):
            acum += samples[i][j]
        avg = acum / len(samples[i])
        std = (sum((v - avg) ** 2 for v in samples[i]) / len(samples[i])) ** 0.5
        if std == 0:
            std = 1
        #Sacar Z
        for j in range(len(samples[i])):
            samples[i][j] = (samples[i][j] - avg) / std
        col_stats.append((avg, std))
    return np.asarray(samples).T.tolist(), col_stats

# Escalar datos de entrenamiento
train_samples, col_stats = scaling(train_samples)

# Aplicar mismas estadísticas a datos de test (evitar data leakage)
test_arr = np.asarray(test_samples, dtype=float).T.tolist()
for i in range(1, len(test_arr)):
    avg, std = col_stats[i - 1]
    for j in range(len(test_arr[i])):
        test_arr[i][j] = (test_arr[i][j] - avg) / std
test_samples = np.asarray(test_arr).T.tolist()

print(f"\nMuestra escalada de train: {train_samples[0]}")
print(f"Muestra escalada de test: {test_samples[0]}")

"""
Construcción del modelo
"""

# Inicializar parámetros
params = [0.0] * len(train_samples[0])
print(f"\nNúmero de parámetros: {len(params)}")

# Hiperparámetros
alpha = 0.1  # Tasa de aprendizaje
max_epochs = 60000

def h(params, sample):
    """This evaluates a generic linear function h(x) with current parameters.  h stands for hypothesis
    
    Args:
        params (lst) a list containing the corresponding parameter for each element x of the sample
        sample (lst) a list containing the values of a sample 

    Returns:
        Evaluation of h(x)
    """
    acum = 0
    for i in range(len(params)):
        acum = acum + params[i] * sample[i]
    return acum

def GD(params, samples, y, alpha):
    """Gradient Descent algorithm 
        Args:
            params (lst) a list containing the corresponding parameter for each element x of the sample
            samples (lst) a 2 dimensional list containing the input samples 
            y (lst) a list containing the corresponding real result for each sample
            alfa(float) the learning rate
        Returns:
            temp(lst) a list with the new values for the parameters after 1 run of the sample set
        """
    temp = list(params)
    for j in range(len(params)):
        acum = 0
        for i in range(len(samples)):
            error = h(params, samples[i]) - y[i]
            acum = acum + error * samples[i][j]
        temp[j] = params[j] - alpha * (1 / len(samples)) * acum
    return temp

# Entrenamiento
print("\nEntrenando modelo...")
train_errors = []
test_errors = []
epochs = 0

while True:
    oldparams = list(params)
    params = GD(params, train_samples, train_y, alpha)
    
    # Calcular MSE de train
    error_acum = 0
    for i in range(len(train_samples)):
        error_acum += (h(params, train_samples[i]) - train_y[i]) ** 2
    train_errors.append(error_acum / len(train_samples))
    
    # Calcular MSE de test
    error_acum = 0
    for i in range(len(test_samples)):
        error_acum += (h(params, test_samples[i]) - test_y[i]) ** 2
    test_errors.append(error_acum / len(test_samples))
    
    epochs += 1
    if oldparams == params or epochs == max_epochs:
        break

print(f"\nEntrenamiento completado")
print(f"Épocas: {epochs}")
print(f"Parámetros finales: {[round(p, 6) for p in params]}")

"""
Evaluación del modelo
"""

# Curva de pérdida
print("\nGenerando curva de pérdidaa")
plt.figure(figsize=(8, 4))
plt.plot(train_errors, label="Train")
plt.plot(test_errors, label="Test")
plt.xlabel("Época")
plt.ylabel("MSE")
plt.title("Curva de Pérdida — Train vs Test")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_curva_perdida.png", dpi=300)
plt.close()

# Calcular métricas
def r2(y_real, y_pred):
    """
    Calcula el coeficiente de determinación R².
    """
    mean_y = sum(y_real) / len(y_real)
    ss_tot = sum((yr - mean_y) ** 2 for yr in y_real)
    ss_res = sum((yr - yp) ** 2 for yr, yp in zip(y_real, y_pred))
    return 1 - ss_res / ss_tot

train_preds = [h(params, s) for s in train_samples]
test_preds = [h(params, s) for s in test_samples]

print("MÉTRICAS DE EVALUACIÓN")
print(f"\n=== Train ===")
print(f"  MSE : {train_errors[-1]:.4f}")
print(f"  R²  : {r2(train_y, train_preds):.4f}")
print(f"\n=== Test ===")
print(f"  MSE : {test_errors[-1]:.4f}")
print(f"  R²  : {r2(test_y, test_preds):.4f}")

# Predicciones vs valores reales
print("\nGenerando gráfica de predicciones vs reales")
plt.figure(figsize=(6, 6))
plt.scatter(test_y, test_preds, alpha=0.6, s=20)

lim = [min(test_y + test_preds), max(test_y + test_preds)]
plt.plot(lim, lim, color="red", linewidth=1.5, label="predicción perfecta")

plt.xlabel("Precio Real")
plt.ylabel("Precio Predicho")
plt.title("Real vs Predicho (Test set)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_predicciones_vs_reales.png", dpi=300)
plt.close()

print("Tarea completada")
