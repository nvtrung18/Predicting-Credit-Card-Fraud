import pandas as pd
import numpy as np

def make_split(data_path="data/train.csv", output_path="data/train_split.csv", seed=9, val_ratio=0.2):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Stratified split by is_fraud with fixed seed
    np.random.seed(seed)
    
    val_indices = []
    for label, group in df.groupby("is_fraud"):
        val_sample = group.sample(frac=val_ratio, random_state=seed)
        val_indices.extend(val_sample.index)
        
    df["is_val"] = False
    df.loc[val_indices, "is_val"] = True
    
    # Validation stats
    n_total = len(df)
    n_train = (~df["is_val"]).sum()
    n_val = df["is_val"].sum()
    
    train_fraud_rate = df[~df["is_val"]]["is_fraud"].mean() * 100
    val_fraud_rate = df[df["is_val"]]["is_fraud"].mean() * 100
    
    print("\n" + "="*40)
    print("        STRATIFIED SPLIT REPORT")
    print("="*40)
    print(f"Total samples:     {n_total:,}")
    print(f"Train samples:     {n_train:,} ({n_train / n_total * 100:.2f}%)")
    print(f"Validation samples:{n_val:,} ({n_val / n_total * 100:.2f}%)")
    print("-" * 40)
    print(f"Train Fraud Rate:  {train_fraud_rate:.4f}%")
    print(f"Val Fraud Rate:    {val_fraud_rate:.4f}%")
    print("="*40)
    
    # Save full dataset with is_val column
    print(f"\nSaving split dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done! File saved successfully.")

if __name__ == "__main__":
    make_split()
