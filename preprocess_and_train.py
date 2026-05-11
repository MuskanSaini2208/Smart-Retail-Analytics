# src/preprocess_and_train.py
import csv
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

BASE_DIR = os.path.dirname(__file__)   # project root
DATA_DIR = os.path.join(BASE_DIR, "DATASET")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def read_csv_raw(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def write_csv(path, rows, headers):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def build_maps(aisles, depts, products):
    aisle_map = {r["aisle_id"]: r["aisle"] for r in aisles}
    dept_map = {r["department_id"]: r["department"] for r in depts}
    prod_map = {r["product_id"]: r for r in products}
    return aisle_map, dept_map, prod_map

def preprocess_raw():
    """
    Merges instacart CSVs into DATA/cleaned_orders.csv.
    """
    print("Preprocessing: loading raw CSVs from", DATA_DIR)
    aisles = read_csv_raw(os.path.join(DATA_DIR, "aisles.csv"))
    depts = read_csv_raw(os.path.join(DATA_DIR, "departments.csv"))
    products = read_csv_raw(os.path.join(DATA_DIR, "products.csv"))
    order_prior = read_csv_raw(os.path.join(DATA_DIR, "order_products__prior.csv"))

    aisle_map, dept_map, prod_map = build_maps(aisles, depts, products)

    print("Merging datasets...")
    cleaned = []
    for op in order_prior:
        pid = op.get("product_id", "")
        prod = prod_map.get(pid)
        if prod:
            cleaned.append({
                "order_id": op.get("order_id", ""),
                "product_id": pid,
                "add_to_cart_order": op.get("add_to_cart_order", ""),
                "reordered": op.get("reordered", "0"),
                "aisle_id": prod.get("aisle_id", ""),
                "department_id": prod.get("department_id", ""),
                "product_name": prod.get("product_name", ""),
                "aisle": aisle_map.get(prod.get("aisle_id", ""), ""),
                "department": dept_map.get(prod.get("department_id", ""), "")
            })

    out_path = os.path.join(DATA_DIR, "cleaned_orders.csv")
    headers = ["order_id","product_id","add_to_cart_order","reordered",
               "aisle_id","department_id","product_name","aisle","department"]
    write_csv(out_path, cleaned, headers)
    print("Saved:", out_path, f" (rows: {len(cleaned)})")

def train_nn(epochs=20, batch_size=256):
    """
    Train a simple NN on DATA/cleaned_orders.csv and save model to models/.
    """
    cleaned_path = os.path.join(DATA_DIR, "cleaned_orders.csv")
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"{cleaned_path} not found — run preprocess_raw() first.")

    print("Training neural network using:", cleaned_path)
    df = pd.read_csv(cleaned_path)
    # convert & clean
    for col in ["add_to_cart_order","aisle_id","department_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)

    le = LabelEncoder()
    df["product_id_enc"] = le.fit_transform(df["product_id"])
    df["reordered"] = df["reordered"].astype(int)

    X = df[["add_to_cart_order","aisle_id","department_id","product_id_enc"]]
    y = df["reordered"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)
    # further split for validation inside fit via validation_split

    model = Sequential([
        Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}  Test accuracy: {acc:.4f}")

    # save model (native keras format)
    model_path = os.path.join(MODELS_DIR, "reorder_nn_model.keras")
    model.save(model_path)
    # also save scaler and label encoder so predictions can be reproduced if needed
    import joblib
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print("Saved NN model and preprocessors to", MODELS_DIR)
