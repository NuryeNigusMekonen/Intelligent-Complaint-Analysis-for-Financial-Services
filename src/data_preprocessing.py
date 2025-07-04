import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()
def map_to_target_product(product, narrative=""):
    product = str(product).lower()
    narrative = str(narrative).lower()

    if "credit card" in product or "prepaid card" in product:
        return "Credit card"
    
    elif any(term in product for term in [
        "personal loan", "consumer loan", "payday loan",
        "title loan", "advance loan", "vehicle loan"
    ]):
        if "buy now pay later" in narrative or "bnpl" in narrative:
            return "Buy Now, Pay Later"
        return "Personal loan"
    
    elif "checking or savings" in product or "bank account" in product or "savings" in product:
        return "Savings account"

    elif "money transfer" in product or "virtual currency" in product or "money service" in product:
        return "Money transfer"

    return None
def run_preprocessing(
    input_path="../data/complaints.csv",
    output_path="../data/filtered_complaints.csv",
    plot_dir="plots",
    chunksize=100_000
):
    print(" Starting chunked preprocessing...")

    filtered_chunks = []
    total_raw = 0
    total_kept = 0
    total_with_narrative = 0
    total_without_narrative = 0

    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize, low_memory=False)):
        print(f" Processing chunk {i+1}...")
        chunk.columns = chunk.columns.str.strip()
        total_raw += len(chunk)

        chunk = chunk[['Complaint ID', 'Product', 'Consumer complaint narrative']].copy()
        chunk.columns = ['complaint_id', 'product', 'narrative']

        total_with_narrative += chunk['narrative'].notna().sum()
        total_without_narrative += chunk['narrative'].isna().sum()

        chunk = chunk.dropna(subset=['narrative'])

        chunk['mapped_product'] = chunk.apply(
            lambda row: map_to_target_product(row['product'], row['narrative']),
            axis=1
        )
        chunk = chunk[chunk['mapped_product'].notna()]

        chunk['narrative'] = chunk['narrative'].astype(str)
        chunk = chunk[chunk['narrative'].str.strip().str.len() > 30]
        chunk['cleaned_narrative'] = chunk['narrative'].apply(clean_text)
        chunk['narrative_len'] = chunk['cleaned_narrative'].apply(lambda x: len(x.split()))
        chunk = chunk[(chunk['narrative_len'] > 10) & (chunk['narrative_len'] < 1000)]

        total_kept += len(chunk)
        filtered_chunks.append(chunk[['complaint_id', 'mapped_product', 'cleaned_narrative']])

    # Combine all filtered chunks
    df_final = pd.concat(filtered_chunks, ignore_index=True)
    df_final.columns = ['complaint_id', 'product', 'cleaned_narrative']

    # EDA Summary
    print("\n EDA Summary - Task 1")
    print(f"Total complaints processed: {total_raw:,}")
    print(f"Complaints with narratives: {total_with_narrative:,}")
    print(f"Complaints without narratives: {total_without_narrative:,}")
    print(f"Complaints retained after filtering: {total_kept:,}")
    print("\nProduct distribution (filtered):")
    print(df_final['product'].value_counts())

    # Save cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"\n Saved filtered dataset to: {output_path}")

    # Create plot dir
    os.makedirs(plot_dir, exist_ok=True)

    #  Plot 1: Narrative length distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df_final['cleaned_narrative'].apply(lambda x: len(x.split())), bins=50, kde=True)
    plt.title("Distribution of Narrative Length (Words)")
    plt.xlabel("Number of Words")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    path1 = os.path.join(plot_dir, "narrative_length_distribution.png")
    plt.savefig(path1)
    print(f" Saved narrative length distribution plot to: {path1}")

    #  Plot 2: Product distribution
    plt.figure(figsize=(8, 4))
    sns.countplot(y=df_final['product'], order=df_final['product'].value_counts().index, palette="Set2")
    plt.title("Complaint Count per Product (Filtered)")
    plt.xlabel("Count")
    plt.ylabel("Product")
    plt.tight_layout()
    path2 = os.path.join(plot_dir, "product_distribution.png")
    plt.savefig(path2)
    print(f" Saved product distribution plot to: {path2}")

    return df_final


if __name__ == "__main__":
    run_preprocessing()
