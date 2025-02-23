import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Définition des chemins
DATA_PATH = "data/employes.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# Vérifier si le fichier de données existe
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Le fichier {DATA_PATH} est introuvable.")

# Charger le dataset
df = pd.read_csv(DATA_PATH,encoding="utf-8")
# print(df.columns)
X = df[['anciennete']]
y = df['salaire']

# Diviser les données
dev_split = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=dev_split, random_state=42)

# Entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Erreur absolue moyenne : {mae}")

# Vérifier si le dossier 'models' existe, sinon le créer
os.makedirs(MODEL_DIR, exist_ok=True)

# Sauvegarder le modèle
joblib.dump(model, MODEL_PATH)
print(f"Modèle sauvegardé sous {MODEL_PATH}")
