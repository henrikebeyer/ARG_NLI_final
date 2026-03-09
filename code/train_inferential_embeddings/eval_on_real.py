import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import euclidean, cosine
import math
import numpy as np
import os
import csv

path2 = "unrelated_only.csv"
path = "corpus_nli_preds_threshold_hard_0.999_noWebis_cleaned.csv"

unrelated_df = pd.read_csv(path2, usecols=["ID", "relation", "antecedent", "consequent"])#.sample(n=10000)
normal_df = pd.read_csv(path, usecols=["ID", "relation", "antecedent", "consequent"])#.sample(n=10000)

data = pd.concat([unrelated_df, normal_df])

# print(len(unrelated_df["ID"]))
# print(len(normal_df["ID"]))
# print(len(unrelated_df["ID"])+len(normal_df["ID"]))
# print(len(data["ID"]))

# print(data)
# data = pd.read_csv(path2, sep="\t")#.dropna(subset=["ID", "antecedent", "consequent", "nli"])
# print(data)

print(data["relation"].unique())

model_path = "output/cognitive-ease-model_final"
sbert_path = "sentence-transformers/all-mpnet-base-v2"#all-MiniLM-L6-v2"

"""
The aim of this script is to evaluate whether high distance NLI (neutral) 
gets assigned higher inf-distance values than low-distance NLI (entailment/contra)

How I could do this: 
* general check: average over all low distance and compare with average of high-distance
* specific: for every low-distance is it lower than avg of high-distance?
* strict: for every low-distance is it lower than any high-distance?
"""

# load model
print("Loading best model for final testing...")


def run_full_evaluation(my_model_path, test_df):
    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)
    
    my_model = SentenceTransformer(my_model_path)
    sbert = SentenceTransformer(sbert_path)
    
    my_distances = {
        'support':[],
        'neutral':[],
        'attack':[]}
    
    sbert_distances = {
        'support':[],
        'neutral':[],
        'attack':[]}
    
    for idx, row in test_df.iterrows():
        anchor = row['antecedent']
        # Extract Texts
        consequent = row['consequent']
        arg = row["relation"].lower()
        # Batch encode
        my_embs = my_model.encode([anchor, consequent])
        sbert_embs = sbert.encode([anchor, consequent])
        
        # Calculate Distances (Cosine)
        # Note: Scipy cosine is distance (0=same, 1=ortho, 2=opp)
        d_mine  = cosine(my_embs[0], my_embs[1])
        d_sbert = cosine(sbert_embs[0], sbert_embs[1])
        
        # Check Conditions (Success if Easy Dist < Hard Dist)
        my_distances[arg].append(d_mine)
        sbert_distances[arg].append(d_sbert)
            
    # eval are averages of ent & contra below neutral?
    my_average_distances = {
        'support': np.mean(my_distances['support']),
        'neutral': np.mean(my_distances['neutral']),
        'attack': np.mean(my_distances['attack'])}
    
    sbert_average_distances = {
        'support': np.mean(sbert_distances['support']),
        'neutral': np.mean(sbert_distances['neutral']),
        'attack': np.mean(sbert_distances['attack'])}
    
    print("="*50)
    print("Average Distances my model:")
    print("="*50)
    for arg_type, avg_dist in my_average_distances.items():
        print(f"{arg_type:<50} | Average Distance: {avg_dist:.4f}")
        if arg_type != 'neutral':
            if avg_dist < my_average_distances['neutral']:
                print(f"  -> SUCCESS: {arg_type} average distance is lower than neutral.")
            else:
                print(f"  -> FAILURE: {arg_type} average distance is NOT lower than neutral.")
    print("="*50)
    
    print("="*50)
    print("Average Distances SBERT:")
    print("="*50)
    for arg_type, avg_dist in sbert_average_distances.items():
        print(f"{arg_type:<50} | Average Distance: {avg_dist:.4f}")
        if arg_type != 'neutral':
            if avg_dist < sbert_average_distances['neutral']:
                print(f"  -> SUCCESS: {arg_type} average distance is lower than neutral.")
            else:
                print(f"  -> FAILURE: {arg_type} average distance is NOT lower than neutral.")
    print("="*50)
    
    # Specific evaluation
    print("="*50)
    print("Specific Evaluation my model:")
    print("="*50)
    my_success = {
        'support': 0,
        'attack': 0
        }
    
    for arg_type in ['support', 'attack']:
        for dist in my_distances[arg_type]:
            if dist < my_average_distances['neutral']:
                my_success[arg_type] += 1
                
        total = len(my_distances[arg_type])
        success_rate = (my_success[arg_type] / total) * 100 if total > 0 else 0
        print(f"{arg_type:<50} | Success Rate: {success_rate:.2f}% ({my_success[arg_type]}/{total})")
        
    print("="*50)
    
    print("="*50)
    print("Specific Evaluation SBERT:")
    print("="*50)
    sbert_success = {
        'support': 0,
        'attack': 0
        }
    
    for arg_type in ['support', 'attack']:
        for dist in sbert_distances[arg_type]:
            if dist < sbert_average_distances['neutral']:
                sbert_success[arg_type] += 1
                
        total = len(sbert_distances[arg_type])
        success_rate = (sbert_success[arg_type] / total) * 100 if total > 0 else 0
        print(f"{arg_type:<50} | Success Rate: {success_rate:.2f}% ({sbert_success[arg_type]}/{total})")
        
    print("="*50)
    
    # Strict evaluation
    print("="*50)
    print("Strict Evaluation my model:")
    print("="*50)
    my_strict_success = {
        'support': 0,
        'attack': 0
        }
    min_neutral_dist = min(my_distances['neutral']) if my_distances['neutral'] else float('inf')
    for arg_type in ['support', 'attack']:
        for dist in my_distances[arg_type]:
            if dist < min_neutral_dist:
                my_strict_success[arg_type] += 1
                
        total = len(my_distances[arg_type])
        strict_success_rate = (my_strict_success[arg_type] / total) * 100 if total > 0 else 0
        print(f"{arg_type:<50} | Strict Success Rate: {strict_success_rate:.2f}% ({my_strict_success[arg_type]}/{total})")
    print("="*50)
    
    print("="*50)
    print("Strict Evaluation SBERT:")
    print("="*50)
    sbert_strict_success = {
        'support': 0,
        'attack': 0
        }
    min_neutral_dist = min(sbert_distances['neutral']) if sbert_distances['neutral'] else float('inf')
    for arg_type in ['support', 'attack']:
        for dist in sbert_distances[arg_type]:
            if dist < min_neutral_dist:
                sbert_strict_success[arg_type] += 1
                
        total = len(sbert_distances[arg_type])
        strict_success_rate = (sbert_strict_success[arg_type] / total) * 100 if total > 0 else 0
        print(f"{arg_type:<50} | Strict Success Rate: {strict_success_rate:.2f}% ({sbert_strict_success[arg_type]}/{total})")
    print("="*50)
    
    
    print("="*50)
    print("extreme values assigned by my model")
    print("="*50)
    for arg in ["support", "attack", "neutral"]:
        print(f"{arg:<50} | Maximum distance assigned: {max(my_distances[arg])}")
        print(f"{arg:<50} | Minimum distance assigned: {min(my_distances[arg])}")
    print("="*50)
    
    print("="*50)
    print("extreme values assigned by SBERT")
    print("="*50)
    for arg in ["support", "attack", "neutral"]:
        print(f"{arg:<50} | Maximum distance assigned: {max(sbert_distances[arg])}")
        print(f"{arg:<50} | Minimum distance assigned: {min(sbert_distances[arg])}")
    print("="*50)
    
    print("Evaluation Complete.")
 
# print(data.columns)   
# run_full_evaluation(model_path, data)

def evaluate_on_adversarial_subset(df, my_model_path, log_path_hidden, log_path_traps, config):
    """
    Filters the dataset for 'HANS-like' samples:
    High Semantic Overlap BUT Label is Neutral
    This is where SBERT fails and Your Model should shine.
    """
    # 0. Prepare Log Files if needed
    if not os.path.exists(log_path_hidden):
        with open(log_path_hidden, "w") as fout:
            writer = csv.writer(fout)
            writer.writerow(["model", "Hidden_is_leq", "Success_threshold", "hidden_acc_SBERT", "hidden_acc_MINE", "MINE_lower_SBERT", "SBERT_avg_dist", "my_avg_dist", "my_avg_lower"])
                
    if not os.path.exists(log_path_traps):
        with open(log_path_traps, "w") as fout:
            writer = csv.writer(fout)
            writer.writerow(["model", "Traps_is_geq", "Success_threshold", "trap_acc_SBERT", "trap_acc_MINE", "MINE_higher_SBERT", "SBERT_avg_dist", "my_avg_dist", "my_avg_higher"])
    
    
    # 1. Load Models
    sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    my_model = SentenceTransformer(my_model_path)
    
    print("Identifying Adversarial Samples (High Sim but NOT Entailment)...")
    
    adversarial_samples_neu = []
    adversarial_samples_other = []
    
    # We iterate and score with SBERT first to find the "Traps"
    # (This takes a moment, but is necessary)
    antecedents = df['antecedent'].tolist()
    consequents = df['consequent'].tolist()
    labels = df['relation'].tolist() # Assumes 'neutral' or 'contradiction' strings
    
    sbert_embs_ant = sbert.encode(antecedents, batch_size=64, show_progress_bar=True)
    sbert_embs_con = sbert.encode(consequents, batch_size=64, show_progress_bar=True)
    
    for i in range(len(df)):
        # Calculate SBERT Semantic Similarity
        sim = 1 - cosine(sbert_embs_ant[i], sbert_embs_con[i])
        
        # DEFINITION OF ADVERSARIAL:
        # Traps:
        # High Similarity (> 0.7) BUT Label is Neutral
        # SBERT thinks these are close. The Ground Truth says they are far.
        
        # Hidden:
        # Low Similarity (<0.5) BUT Label is Support / Attack
        # SBERT thinks these are far away but they are close
        if sim > 0.7 and labels[i] == 'neutral':
            adversarial_samples_neu.append(i)
        elif sim < 0.5 and labels[i] in ["support", "attack"]:
            adversarial_samples_other.append(i)
            
    print(f"\nFound {len(adversarial_samples_neu)} Adversarial 'Trap' unrelated Samples out of {len(df)}.")
    print(f"\nFound {len(adversarial_samples_other)} Hidden Arguments out of {len(df)}.")
    
    if len(adversarial_samples_neu) == 0 or len(adversarial_samples_other) == 0:
        print("No adversarial samples found in this dataset! It is too easy/biased.")
        return

    # 2. Evaluate Performance on this Subset
    print("Evaluating Your Model on the Traps...")
    
    for mode in ["neu", "other"]:
        my_success = 0
        mine_better = 0
        sbert_success = 0
        
        # Pre-encode for My Model
        if mode == "neu":
            adversarial_samples = adversarial_samples_neu
        elif mode == "other":
            adversarial_samples = adversarial_samples_other
        subset_ant = [antecedents[i] for i in adversarial_samples]
        subset_con = [consequents[i] for i in adversarial_samples]
        
        my_embs_ant = my_model.encode(subset_ant, batch_size=64)
        my_embs_con = my_model.encode(subset_con, batch_size=64)
        
        sbert_dists = []
        my_dists = []
        for idx, orig_idx in enumerate(adversarial_samples):
            # SBERT Score (High Sim = Low Dist)
            sbert_dist = cosine(sbert_embs_ant[orig_idx], sbert_embs_con[orig_idx])
            sbert_dists.append(sbert_dist)
            
            # My Model Score
            my_dist = cosine(my_embs_ant[idx], my_embs_con[idx])
            my_dists.append(my_dist)
            
            # THRESHOLD TEST:
            # If distance > 0.5, we correctly say "This is NOT entailment"
            # (Since the label is Neutral, we WANT high distance)
            
            if mode == "neu" and sbert_dist > 0.5: sbert_success += 1 # SBERT avoids trap
            if mode == "neu" and my_dist > sbert_dist: mine_better += 1
            if mode == "neu" and my_dist > 0.5:    my_success += 1    # My Model avoids trap
            if mode == "other" and sbert_dist < 0.5: sbert_success += 1 # SBERT avoids trap
            if mode == "other" and my_dist < 0.5:    my_success += 1    # My Model avoids trap
            if mode == "other" and my_dist < sbert_dist: mine_better += 1
        
        
        success = 0.5     
        if mode == "neu":
            threshold = 0.7
            log = log_path_traps
        elif mode == "other":
            threshold = 0.4
            log = log_path_hidden
        
        with open(log, "a") as fout:
                writer = csv.writer(fout)
                sbert_acc = sbert_success/len(adversarial_samples)
                my_acc = my_success/len(adversarial_samples)
                better_perc = mine_better/len(adversarial_samples)
                avg_SBERT = np.mean(sbert_dists)
                avg_mine = np.mean(my_dist)
                if mode == "neu":
                    my_avg_better = avg_mine > avg_SBERT
                elif mode == "other":
                    my_avg_better = avg_mine < avg_SBERT
                row = [config, threshold, success, sbert_acc, my_acc, better_perc, avg_SBERT, avg_mine, my_avg_better]
                writer.writerow(row)
                
        # print("\n" + "="*50)
        # print(f"ADVERSARIAL 'BOSS FIGHT' RESULTS (N={len(adversarial_samples)})")
        # if mode == "neu":
        #     print(f"Task: Realize that High Semantic similarity != non-ARG (avoid traps)")
        #     filler = "higher"
        # if mode == "other":
        #     print(f"Task: Realize that Low Semantic Similarity != Unrelated (Find hidden arguments)")
        #     filler = "lower"
        # print("="*50)
        # print(f"SBERT Accuracy:      {sbert_success / len(adversarial_samples):.2%}")
        # print(f"Your Model Accuracy: {my_success / len(adversarial_samples):.2%}")
        # print(f"My model assigned a {filler} distance than SBERT: {mine_better / len(adversarial_samples):.2%}")
        # print("="*50)
        
        # print("Average distance assigned by SBERT:", np.mean(sbert_dists))
        # print("Average distance assigned by my model", np.mean(my_dists))

log_path_traps = "/home/henrike/ARG-NLI_project/code/ID_model_training/ablation_logs/high_sim_traps_eval.csv"
log_path_hidden = "/home/henrike/ARG-NLI_project/code/ID_model_training/ablation_logs/low_sim_hidden_args_eval.csv"

for mode in ["all", "noType3", "AugmentationM5", "AugmentationM6"]:
    for m in [0.5, 0.7]:
        model_path = f"/home/henrike/ARG-NLI_project/code/ID_model_training/output/cognitive-ease-model-{mode}_m={m}"
        print("Evaluating Model:", mode, "with m =", m)      
        evaluate_on_adversarial_subset(data, model_path, log_path_traps=log_path_traps, log_path_hidden=log_path_hidden, config=f"{mode}_m={m}")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

def get_hidden_arguments_df(df):
    # 1. Load your full Real World CSV
    #df = pd.read_csv(original_df_path)
    
    # 2. Load SBERT (to decide what counts as "Low Overlap")
    sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # 3. Compute Embeddings
    print("Encoding data to find Hidden Arguments...")
    ants = df['antecedent'].tolist()
    cons = df['consequent'].tolist()
    
    # Batch encode for speed
    emb_a = sbert.encode(ants, batch_size=64, show_progress_bar=True)
    emb_c = sbert.encode(cons, batch_size=64, show_progress_bar=True)
    
    # 4. Filter for Task 2 Conditions:
    #    Condition A: Similarity < 0.6 (SBERT thinks they are unrelated)
    #    Condition B: Label is 'support' or 'attack' (Ground Truth says they ARE related)
    
    hidden_arg_indices = []
    trap_indices = []
    
    for i in range(len(df)):
        # cosine distance
        dist = cosine(emb_a[i], emb_c[i])
        sim = 1 - dist
        
        # Check logic
        if sim < 0.6 and df.iloc[i]['relation'] in ['support', 'attack']:
            hidden_arg_indices.append(i)
        elif sim > 0.6 and df.iloc[i]['relation'] in ['neutral']:
            trap_indices.append(i)
            
    # 5. Create the subset DataFrame
    hidden_args_df = df.iloc[hidden_arg_indices].copy()
    trap_df = df.iloc[trap_indices].copy()
    
    print(f"Found {len(hidden_args_df)} hidden arguments for plotting.")
    print(f"Found {len(trap_df)} hidden arguments for plotting.")
    return hidden_args_df, trap_df

#hidden_args, traps = get_hidden_arguments_df(data)

def plot_success_distribution(df, my_model_path, save_path, mode):
    """
    Plots the histogram of distances for the 'Hidden Arguments' (Task 2).
    SBERT will look like a wall of failure.
    Your Model will show a 'Tail of Success'.
    """
    print("Loading models...")
    sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    my_model = SentenceTransformer(my_model_path)
    
    # Filter for Task 2: Low Overlap but VALID Argument
    # We simulate this by taking your 32,561 sample logic
    # Assuming 'df' contains these specific samples
    
    # (For demo, we just use a subset to be fast)
    subset = df.sample(n=min(2000, len(df)), random_state=42)
    
    ants = subset['antecedent'].tolist()
    cons = subset['consequent'].tolist()
    
    print("Encoding...")
    sb_a = sbert.encode(ants)
    sb_c = sbert.encode(cons)
    my_a = my_model.encode(ants)
    my_c = my_model.encode(cons)
    
    sbert_dists = [cosine(sb_a[i], sb_c[i]) for i in range(len(subset))]
    my_dists = [cosine(my_a[i], my_c[i]) for i in range(len(subset))]
    
    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))
    
    # Plot SBERT
    sns.histplot(sbert_dists, color="red", alpha=0.3, label="Baseline (SBERT)", kde=True, binwidth=0.05)
    
    # Plot YOUR MODEL
    sns.histplot(my_dists, color="blue", alpha=0.3, label="Your Model", kde=True, binwidth=0.05)
    
    # Add Success Threshold Line
    if mode == "trap":
        plt.axvline(x=0.4, color='green', linestyle='--', linewidth=2, label="Success Threshold (>0.4)")
    elif mode == "hidden":
        plt.axvline(x=0.4, color='green', linestyle='--', linewidth=2, label="Success Threshold (<0.4)")

    if mode == "trap":
        plt.title("Distribution of Distances on Semantic 'Traps' (High Overlap but Unrelated)", fontsize=14)
        plt.xlabel("Cosine Distance (Higher is Better)", fontsize=12)
    elif mode == "hidden":
        plt.title("Distribution of Distances on 'Hidden' Arguments (Low Overlap)", fontsize=14)
        plt.xlabel("Cosine Distance (Lower is Better)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    print(f"Saving plot to {save_path}...")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # Important to free memory
    print("Done.")

# plot_success_distribution(hidden_args, my_model_path=model_path, save_path="plots/hidden_args_final.png", mode="hidden")
# plot_success_distribution(traps, my_model_path=model_path, save_path="plots/traps_final.png", mode="trap")