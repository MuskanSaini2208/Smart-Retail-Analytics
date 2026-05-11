# main.py
import argparse
from preprocess_and_train import preprocess_raw, train_nn
from model_purchase_prediction import train as train_purchase_model, predict_interactive
from model_demand_forecasting import train as train_demand_model

def run_all():
    preprocess_raw()
    train_nn()
    purchase_model = train_purchase_model()
    print("\nNow you can try purchase-model prediction (interactive):")
    predict_interactive(purchase_model)
    print("\nNow launching demand forecasting (interactive):")
    train_demand_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Retail Analytics pipeline")
    parser.add_argument("--all", action="store_true", help="Run full pipeline (preprocess, train nn, train purchase model, forecasting)")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train-nn", action="store_true")
    parser.add_argument("--train-purchase", action="store_true")
    parser.add_argument("--train-demand", action="store_true")
    args = parser.parse_args()

    if args.all:
        run_all()
    else:
        if args.preprocess:
            preprocess_raw()
        if args.train_nn:
            train_nn()
        if args.train_purchase:
            train_purchase_model()
        if args.train_demand:
            train_demand_model()
        if not (args.preprocess or args.train_nn or args.train_purchase or args.train_demand or args.all):
            print("No arguments supplied. Try `python main.py --all` to run full pipeline.")
