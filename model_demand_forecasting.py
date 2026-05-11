# src/model_demand_forecasting.py
import csv
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "DATASET")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_product_names():
    products_path = os.path.join(DATA_DIR, "products.csv")
    products = read_csv(products_path)
    return {row["product_id"]: row["product_name"] for row in products}

def train():
    cleaned_path = os.path.join(DATA_DIR, "cleaned_orders.csv")
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"{cleaned_path} not found — run preprocessing first.")
    print("Training demand forecasting (simple counts) from", cleaned_path)

    data = read_csv(cleaned_path)
    product_names = load_product_names()

    demand_counts = defaultdict(int)
    daily_counts = defaultdict(lambda: defaultdict(int))

    # `order_dow` may not exist in cleaned_orders.csv; handle gracefully
    for r in data:
        pid = r.get("product_id")
        day = r.get("order_dow") or "0"
        demand_counts[pid] += 1
        daily_counts[pid][day] += 1

    out_file = os.path.join(MODELS_DIR, "demand_model.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(demand_counts, f, indent=4)
    print("Saved demand_model.json ->", out_file)

    # interactive forecast menu
    forecast_menu(demand_counts, product_names, daily_counts)

def forecast_menu(model, product_names, daily_counts):
    while True:
        pid = input("Enter Product ID to forecast (or 'exit'): ").strip()
        if pid.lower() == "exit":
            break
        if pid not in model:
            print("Product ID not found.")
            continue
        product_name = product_names.get(pid, "Unknown Product")
        total_demand = model[pid]
        week = round(total_demand / 10, 2)
        month = round(total_demand / 3, 2)
        print(f"Product: {product_name} (ID: {pid})")
        print(f"Total demand: {total_demand}, Next week ~ {week}, Next month ~ {month}")
        plot_forecast(pid, product_name, daily_counts.get(pid, {}), week, month)

def plot_forecast(pid, product_name, daily_data, week_f, month_f):
    if not daily_data:
        print("No daily data available to plot.")
        return
    days = list(map(int, daily_data.keys()))
    counts = list(daily_data.values())
    plt.figure(figsize=(8,5))
    plt.plot(days, counts, marker="o", label="Historical Demand")
    plt.axhline(week_f, linestyle="--", label="Next Week Forecast")
    plt.axhline(month_f, linestyle=":", label="Next Month Forecast")
    plt.title(f"Demand for {product_name} (ID: {pid})")
    plt.xlabel("Day of week (0=Sun)")
    plt.ylabel("Order count")
    plt.legend()
    plt.grid(True)
    plt.show()
