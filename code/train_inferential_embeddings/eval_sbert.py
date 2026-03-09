import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split

BASELINE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" #'sentence-transformers/all-MiniLM-L6-v2'
DATA_PATH = "llama3-70b_contrastive_noWebis_combined_cleaned_sem_lex_with_high_high_resampled_hard.csv"

def evaluate_baseline(data_path):
    # 1. Load Data (Same split logic to ensure we test on the same rows)
    print("Loading data...")
    test_df = pd.read_csv(data_path).sample(frac=0.10)
    # We only care about the test set (15% split twice)
    #_, test_df = train_test_split(df, test_size=0.15, random_state=42)
    print(f"Evaluating Baseline on {len(test_df)} test samples.")

    # 2. Load Off-the-Shelf Model
    print(f"Loading baseline model: {BASELINE_MODEL_NAME}...")
    model = SentenceTransformer(BASELINE_MODEL_NAME)

    # 3. STORAGE
    results = {
        "Type 1 (Anti-Semantic): P=HighSim vs N=HighSim": [],
        "Type 2 (Boss Fight):    P=LowSim  vs N=HighSim": [],
        "Type 3 (Anchoring):     P=HighSim vs N=LowSim": [],
        "Type 4 (Pure Inf):      P=LowSim  vs N=LowSim": []
    }
    
    # For Plotting
    plot_data = {'sem_sim': [], 'model_dist': [], 'color': []}

    print("Running evaluation...")
    for idx, row in test_df.iterrows():
        for mode in ["inf", "conf"]:
            anchor = row['antecedent']
            
            # Texts
            t_easy_low  = row[f'low_low_consequent_{mode}_final']   # Easy / Low Sim
            t_easy_high = row[f'low_high_consequent_{mode}_final']  # Easy / High Sim
            t_hard_low  = row[f'high_low_consequent_{mode}']  # Hard / Low Sim
            t_hard_high = row[f'high_high_consequent_{mode}_final'] # Hard / High Sim
            
            # Encode
            embs = model.encode([anchor, t_easy_low, t_easy_high, t_hard_low, t_hard_high])
            
            # Distances (Cosine)
            d_easy_low  = cosine(embs[0], embs[1])
            d_easy_high = cosine(embs[0], embs[2])
            d_hard_low  = cosine(embs[0], embs[3])
            d_hard_high = cosine(embs[0], embs[4])
            
            # --- Accuracy Checks ---
            # 1. Anti-Semantic (Easy/High vs Hard/High)
            results["Type 1 (Anti-Semantic): P=HighSim vs N=HighSim"].append(d_easy_high < d_hard_high)
            
            # 2. Boss Fight (Easy/Low vs Hard/High) -> EXPECT FAILURE HERE
            results["Type 2 (Boss Fight):    P=LowSim  vs N=HighSim"].append(d_easy_low < d_hard_high)
            
            # 3. Anchoring (Easy/High vs Hard/Low) -> EXPECT HIGH SUCCESS
            results["Type 3 (Anchoring):     P=HighSim vs N=LowSim"].append(d_easy_high < d_hard_low)
            
            # 4. Pure Inference (Easy/Low vs Hard/Low)
            results["Type 4 (Pure Inf):      P=LowSim  vs N=LowSim"].append(d_easy_low < d_hard_low)

            # --- Collect Plot Data (Sample of 500 points to keep plot clean) ---
            if idx < 500:
                # We plot Semantic Similarity (X) vs Model Distance (Y)
                # Since this IS the baseline model, X and Y are mathematically related.
                # Sim = 1 - Dist. We expect a perfect diagonal.
                
                # Point A: Easy (Low Sim) -> Should be Green
                plot_data['sem_sim'].append(1 - d_easy_low)
                plot_data['model_dist'].append(d_easy_low)
                plot_data['color'].append('green')
                
                # Point B: Hard (High Sim) -> Should be Red
                plot_data['sem_sim'].append(1 - d_hard_high)
                plot_data['model_dist'].append(d_hard_high)
                plot_data['color'].append('red')

    # --- Print Stats ---
    print("\n" + "="*50)
    print("BASELINE (PRE-TRAINING) PERFORMANCE")
    print("="*50)
    for name, scores in results.items():
        print(f"{name:<50} | Accuracy: {np.mean(scores):.2%}")
    print("="*50)

    # --- Generate Plot ---
    plt.figure(figsize=(8, 8))
    plt.scatter(plot_data['sem_sim'], plot_data['model_dist'], c=plot_data['color'], alpha=0.5, s=30)
    plt.xlabel("Semantic Similarity (X-Axis)", fontsize=12)
    plt.ylabel("Model Distance (Y-Axis)", fontsize=12)
    plt.title("Baseline SBERT: Entanglement Visualization", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Easy Inference'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Hard Inference')
    ]
    plt.legend(handles=legend_elements)
    
    plt.show()
    
evaluate_baseline("special_test.csv")