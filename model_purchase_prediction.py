# src/model_purchase_prediction.py
import csv
import json
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "DATASET")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_product_names():
    products_path = os.path.join(DATASET_DIR, "products.csv")
    products = read_csv(products_path)
    return {row["product_id"]: row["product_name"] for row in products}

def train():
    cleaned_path = os.path.join(DATASET_DIR, "cleaned_orders.csv")
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"{cleaned_path} not found — run preprocessing first.")
    print("Training purchase-rate model from", cleaned_path)

    data = read_csv(cleaned_path)
    total_counts = defaultdict(int)
    reorder_counts = defaultdict(int)

    for r in data:
        pid = r.get("product_id")
        total_counts[pid] += 1
        if r.get("reordered", "0") == "1":
            reorder_counts[pid] += 1

    model = {}
    for pid, tot in total_counts.items():
        model[pid] = {"reorder_rate": (reorder_counts[pid] / tot) if tot else 0.0}

    out_file = os.path.join(MODELS_DIR, "purchase_model.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=4)

    print("Saved purchase_model.json ->", out_file)
    return model

def predict_interactive(model=None):
    if model is None:
        # load model if not provided
        model_path = os.path.join(MODELS_DIR, "purchase_model.json")
        with open(model_path, "r", encoding="utf-8") as f:
            model = json.load(f)

    product_names = load_product_names()
    pid = input("Enter product_id to predict reorder probability (or 'exit'): ").strip()
    if pid.lower() == "exit":
        return
    if pid not in model:
        print("Product ID not found.")
        return
    rate = model[pid]["reorder_rate"]
    print(f"Product ID: {pid}")
    print(f"Product Name: {product_names.get(pid,'Unknown')}")
    print(f"Predicted reorder rate: {round(rate,4)}")
