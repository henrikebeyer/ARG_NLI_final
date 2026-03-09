from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers import SentenceTransformer, losses, models
from torch.utils.data import DataLoader
from prepare_data import (load_and_split_data, 
                          generate_triplets,
                          generate_noType3_triplets, 
                          generate_weighted_triplets, 
                          generate_hard_mode_triplets,
                          mine_hard_negatives,
                          generate_surgical_triplets,
                          generate_M5_triplets,
                          generate_M6_triplets)
from scipy.spatial.distance import euclidean, cosine
import math
import numpy as np
import pandas as pd
import sys
import csv
import os

data_config = sys.argv[1]
triplet_margin = float(sys.argv[2])

class Config:
    base_model = "sentence-transformers/all-mpnet-base-v2" #"sentence-transformers/all-MiniLM-L6-v2" # Smaller, more adaptable model
    output_path = f"output/cognitive-ease-model-{data_config}_m={triplet_margin}"
    batch_size = 16 # MiniLM can handle larger batches
    num_epochs = 6 # Needs time to unlearn semantics
    # model_name = "sentence-transformers/all-mpnet-base-v2"
    # output_path = "output/cognitive-ease-model"
    # batch_size = 16
    # num_epochs = 8
    learning_rate = 2e-5
    # Margin 'm' for Triplet Loss: d(A, P) + m < d(A, N)
    # 1.0 is standard for Euclidean, 0.5 is standard for Cosine.
    triplet_margin = triplet_margin #0.6 #1.0 
    max_seq_length = 256
    weight_decay = 0.01
    data_config = data_config
    
def setup_evaluator(val_triplets, name="dev"):
    """
    Creates an evaluator that monitors the "congitive ease" condition.
    It checks: Distance(anchor, easy) < Disatance(anchor, hard)
    """
    return TripletEvaluator.from_input_examples(
        examples=val_triplets,
        name=f"{name}_evaluator",
        show_progress_bar=True,
        write_csv=True,
        batch_size=Config.batch_size
    )
    
def train_inf_distance_model(train_triplets, evaluator):
    """
    Initialises SBERT and runs the training loop using Triplet loss
    
    :param train_triplets: inferential distance triplets for training
    :param evaluator: evaluator to evaluate the triplets
    """
    
    # --- Model init ---
    print(f"Loading base model: {Config.model_name} ...")
    word_embedding_model = models.Transformer(Config.model_name, max_seq_length=Config.max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # --- DataLoader ---
    # Shuffling is important to prevent batch-level patterns
    train_dataloader = DataLoader(
        train_triplets,
        shuffle=True,
        batch_size=Config.batch_size
    )
    
    # --- Loss Function ---
    # We use Euclidean Distance
    # This aligns with the spatial intuition of "distance" = "cognitive effort".
    # train_loss = losses.TripletLoss(
    #     model=model,
    #     distance_metric=losses.TripletDistanceMetric.EUCLIDEAN,
    #     triplet_margin=Config.triplet_margin
    # )
    
    # Cosine Distance = 1 - Cosine Similarity
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE, 
        triplet_margin=Config.triplet_margin
    )
    
    # --- warmup steps ---
    # Standard practice: 10% of total training steps
    total_steps = len(train_dataloader) * Config.num_epochs
    warmup_steps = int(total_steps*0.1)
    
    # --- Traning Loop ---
    print("Starting Training ...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=Config.num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr":Config.learning_rate},
        output_path=Config.output_path,
        save_best_model=True,
        show_progress_bar=True,
        weight_decay=Config.weight_decay
    )
    
    print(f"Training complete. Model saved to {Config.output_path}")
    return model

def analyze_model_performance(model_path, test_df):
    """
    Loads the trained model and performs a granular error analysis 
    on the test set, breaking down accuracy by triplet strategy.
    """
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = SentenceTransformer(model_path)
    
    # Storage for results
    results = {
        "Type 1: Anti-Semantic (P=HighSim vs N=HighSim)": [],
        "Type 2: Disentanglement (P=LowSim vs N=HighSim)": [], # The Hardest Task
        "Type 3: Anchoring (P=HighSim vs N=LowSim)": [],
        "Type 4: Pure Inference (P=LowSim vs N=LowSim)": []
    }

    print("Running evaluation on Test Set...")
    
    for idx, row in test_df.iterrows():
        anchor = row['antecedent']
        
        # We process 'inf' (Inference) columns. 
        # (You can run this loop again for '_conf' if you want separate stats)
        
        # Define the texts
        pos_low_sim  = row['low_low_consequent_inf_final']   # Easy / Low Sim
        pos_high_sim = row['low_high_consequent_inf_final']  # Easy / High Sim
        neg_low_sim  = row['high_low_consequent_inf']  # Hard / Low Sim
        neg_high_sim = row['high_high_consequent_inf_final'] # Hard / High Sim
        
        # Encode all at once for speed
        texts = [anchor, pos_low_sim, pos_high_sim, neg_low_sim, neg_high_sim]
        embs = model.encode(texts)
        
        emb_anchor       = embs[0]
        emb_pos_low_sim  = embs[1]
        emb_pos_high_sim = embs[2]
        emb_neg_low_sim  = embs[3]
        emb_neg_high_sim = embs[4]
        
        # Helper to check if d(Anchor, Pos) < d(Anchor, Neg)
        def is_correct(pos_emb, neg_emb):
            d_pos = euclidean(emb_anchor, pos_emb)
            d_neg = euclidean(emb_anchor, neg_emb)
            return 1 if d_pos < d_neg else 0

        # --- Check the 4 Strategies ---

        # 1. Low Dist/High Sim (P) x High Dist/High Sim (N)
        # Does it realize the "Hard" one is further away, even though both look similar?
        results["Type 1: Anti-Semantic (P=HighSim vs N=HighSim)"].append(
            is_correct(emb_pos_high_sim, emb_neg_high_sim)
        )

        # 2. Low Dist/Low Sim (P) x High Dist/High Sim (N)
        # THE BOSS FIGHT: Can it pull the "Low Sim" close and push "High Sim" away?
        results["Type 2: Disentanglement (P=LowSim vs N=HighSim)"].append(
            is_correct(emb_pos_low_sim, emb_neg_high_sim)
        )

        # 3. Low Dist/High Sim (P) x High Dist/Low Sim (N)
        # Easy win: P is semantically close AND easy. N is far AND hard.
        results["Type 3: Anchoring (P=HighSim vs N=LowSim)"].append(
            is_correct(emb_pos_high_sim, emb_neg_low_sim)
        )
        
        # 4. Low Dist/Low Sim (P) x High Dist/Low Sim (N)
        # Pure Inference: Semantics are removed (both low).
        results["Type 4: Pure Inference (P=LowSim vs N=LowSim)"].append(
            is_correct(emb_pos_low_sim, emb_neg_low_sim)
        )

    # --- Print Report ---
    print("\n" + "="*50)
    print(f"DETAILED PERFORMANCE REPORT (N={len(test_df)} samples)")
    print("="*50)
    
    overall_acc = []
    
    for strategy_name, scores in results.items():
        acc = np.mean(scores)
        overall_acc.extend(scores)
        print(f"\n{strategy_name}")
        print(f"Accuracy: {acc:.2%} ({sum(scores)}/{len(scores)})")
        
        if "Type 2" in strategy_name:
            if acc < 0.5:
                print("   -> NOTE: Model still biased toward Semantics.")
            else:
                print("   -> SUCCESS: Model is prioritizing Inference over Semantics!")

    print("\n" + "-"*50)
    print(f"OVERALL ACCURACY: {np.mean(overall_acc):.2%}")
    print("="*50)

def full_cosine_evaluation(model_path, test_df):
    model = SentenceTransformer(model_path)
    print(f"Loading model from {model_path}...")
    print("Running Full Cosine Evaluation (Lower distance = Better)...")
    
    results = {
        "Type 1 (Anti-Semantic): P=HighSim vs N=HighSim": [],
        "Type 2 (Boss Fight):    P=LowSim  vs N=HighSim": [],
        "Type 3 (Anchoring):     P=HighSim vs N=LowSim": [],
        "Type 4 (Pure Inf):      P=LowSim  vs N=LowSim": []
    }
    
    for idx, row in test_df.iterrows():
        anchor = row['antecedent']
        
        # Embed all 5 texts
        texts = [
            anchor, 
            row['low_low_consequent_inf_final'],   # P_LowSim (Easy)
            row['low_high_consequent_inf_final'],  # P_HighSim (Easy)
            row['high_low_consequent_inf'],  # N_LowSim (Hard)
            row['high_high_consequent_inf_final']  # N_HighSim (Hard)
        ]
        embs = model.encode(texts)
        
        # Calculate Cosine Distances (0 to 2)
        # Note: cosine() returns distance (1 - similarity), so LOWER is CLOSE.
        d_p_low  = cosine(embs[0], embs[1])
        d_p_high = cosine(embs[0], embs[2])
        d_n_low  = cosine(embs[0], embs[3])
        d_n_high = cosine(embs[0], embs[4])
        
        # --- Check 1: Anti-Semantic (Easy/High vs Hard/High) ---
        # If this fails, model can't distinguish when both are similar.
        results["Type 1 (Anti-Semantic): P=HighSim vs N=HighSim"].append(1 if d_p_high < d_n_high else 0)

        # --- Check 2: The Boss Fight (Easy/Low vs Hard/High) ---
        # The critical disentanglement metric.
        results["Type 2 (Boss Fight):    P=LowSim  vs N=HighSim"].append(1 if d_p_low < d_n_high else 0)

        # --- Check 3: Anchoring (Easy/High vs Hard/Low) ---
        # If this fails, we Overcorrected (model hates similarity now).
        results["Type 3 (Anchoring):     P=HighSim vs N=LowSim"].append(1 if d_p_high < d_n_low else 0)
        
        # --- Check 4: Pure Inference (Easy/Low vs Hard/Low) ---
        results["Type 4 (Pure Inf):      P=LowSim  vs N=LowSim"].append(1 if d_p_low < d_n_low else 0)

    # --- Print Report ---
    print("\n" + "="*60)
    print(f"FULL COSINE PERFORMANCE REPORT (N={len(test_df)})")
    print("="*60)
    
    for name, scores in results.items():
        acc = np.mean(scores)
        print(f"{name:<50} | Accuracy: {acc:.2%}")

    print("-" * 60)

def check_cosine_performance(model_path, test_df):
    model = SentenceTransformer(model_path)
    print("Running Cosine Evaluation...")
    
    results = {"Type 2 (Boss Fight)": [], "Type 4 (Pure Inf)": []}
    
    for idx, row in test_df.iterrows():
        anchor = row['antecedent']
        
        # Embed
        texts = [
            anchor, 
            row['low_low_consequent_inf_final'],   # Pos: Easy / Low Sim
            row['high_high_consequent_inf_final'], # Neg: Hard / High Sim
            row['high_low_consequent_inf']   # Neg: Hard / Low Sim
        ]
        embs = model.encode(texts)
        
        # Calculate Cosine Distances (lower is closer)
        d_pos_low = cosine(embs[0], embs[1]) # Easy/Low
        d_neg_high = cosine(embs[0], embs[2]) # Hard/High
        d_neg_low = cosine(embs[0], embs[3]) # Hard/Low
        
        # Type 2 Check: Is Easy(LowSim) closer than Hard(HighSim)?
        results["Type 2 (Boss Fight)"].append(1 if d_pos_low < d_neg_high else 0)

        # Type 4 Check: Is Easy(LowSim) closer than Hard(LowSim)?
        results["Type 4 (Pure Inf)"].append(1 if d_pos_low < d_neg_low else 0)

    print(f"Type 2 Accuracy: {np.mean(results['Type 2 (Boss Fight)']):.2%}")
    print(f"Type 4 Accuracy: {np.mean(results['Type 4 (Pure Inf)']):.2%}")

# --- 3. EVALUATION METRICS ---
def run_full_evaluation(model, test_df, log_path):
    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)
    
    results = {
        "Type 1 (Anti-Semantic): P=HighSim vs N=HighSim": [],
        "Type 2 (Boss Fight):    P=LowSim  vs N=HighSim": [],
        "Type 3 (Anchoring):     P=HighSim vs N=LowSim": [],
        "Type 4 (Pure Inf):      P=LowSim  vs N=LowSim": []
    }
    
    for idx, row in test_df.iterrows():
        for mode in ["inf", "conf"]:
            anchor = row['antecedent']
            # Extract Texts
            t_easy_low = row[f'low_low_consequent_{mode}_final']
            t_easy_high = row[f'low_high_consequent_{mode}_final']
            t_hard_low = row[f'high_low_consequent_{mode}']
            t_hard_high = row[f'high_high_consequent_{mode}_final']
            
            # Batch encode
            embs = model.encode([anchor, t_easy_low, t_easy_high, t_hard_low, t_hard_high])
            
            # Calculate Distances (Cosine)
            # Note: Scipy cosine is distance (0=same, 1=ortho, 2=opp)
            d_easy_low  = cosine(embs[0], embs[1])
            d_easy_high = cosine(embs[0], embs[2])
            d_hard_low  = cosine(embs[0], embs[3])
            d_hard_high = cosine(embs[0], embs[4])
            
            # Check Conditions (Success if Easy Dist < Hard Dist)
            results["Type 1 (Anti-Semantic): P=HighSim vs N=HighSim"].append(d_easy_high < d_hard_high)
            results["Type 2 (Boss Fight):    P=LowSim  vs N=HighSim"].append(d_easy_low < d_hard_high)
            results["Type 3 (Anchoring):     P=HighSim vs N=LowSim"].append(d_easy_high < d_hard_low)
            results["Type 4 (Pure Inf):      P=LowSim  vs N=LowSim"].append(d_easy_low < d_hard_low)

    if not os.path.exists(log_path):
        with open(log_path, mode="w") as out_file:
            writer = csv.writer(out_file)
            writer.writerow(["id"] + list(results.keys()))    
    
    with open(log_path, mode="a") as out_file:
        accs = [np.mean(scores) for scores in results.values()]
        writer = csv.writer(out_file)
        id = f"{Config.data_config}_m={Config.triplet_margin}"
        writer.writerow([id] + accs)

# --- 4. EVALUATION WITH ERROR ANALYSIS ---
def eval_with_errors(model, test_df, log_path, error_path=""):
    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)
    
    results = {
        "Type 1 (Anti-Semantic): P=HighSim vs N=HighSim": [],
        "Type 2 (Boss Fight):    P=LowSim  vs N=HighSim": [],
        "Type 3 (Anchoring):     P=HighSim vs N=LowSim": [],
        "Type 4 (Pure Inf):      P=LowSim  vs N=LowSim": []
    }
    
    errors = {
        "anchor": [],
        "positive": [],
        "negative": [],
        "type":[]
    }
    
    for idx, row in test_df.iterrows():
        for mode in ["inf", "conf"]:
            anchor = row['antecedent']
            # Extract Texts
            t_easy_low = row[f'low_low_consequent_{mode}_final']
            t_easy_high = row[f'low_high_consequent_{mode}_final']
            t_hard_low = row[f'high_low_consequent_{mode}']
            t_hard_high = row[f'high_high_consequent_{mode}_final']
            
            # Batch encode
            embs = model.encode([anchor, t_easy_low, t_easy_high, t_hard_low, t_hard_high])
            
            # Calculate Distances (Cosine)
            # Note: Scipy cosine is distance (0=same, 1=ortho, 2=opp)
            d_easy_low  = cosine(embs[0], embs[1])
            d_easy_high = cosine(embs[0], embs[2])
            d_hard_low  = cosine(embs[0], embs[3])
            d_hard_high = cosine(embs[0], embs[4])
            
            # Check Conditions (Success if Easy Dist < Hard Dist)
            results["Type 1 (Anti-Semantic): P=HighSim vs N=HighSim"].append(d_easy_high < d_hard_high)
            results["Type 2 (Boss Fight):    P=LowSim  vs N=HighSim"].append(d_easy_low < d_hard_high)
            results["Type 3 (Anchoring):     P=HighSim vs N=LowSim"].append(d_easy_high < d_hard_low)
            results["Type 4 (Pure Inf):      P=LowSim  vs N=LowSim"].append(d_easy_low < d_hard_low)
            
            for index, outcome in enumerate([d_easy_high < d_hard_high, d_easy_low < d_hard_high, d_easy_high < d_hard_low, d_easy_low < d_hard_low]):
                if outcome == False:
                    errors["anchor"].append(anchor)
                    errors["type"].append(index+1)
                    if index == 0:
                        errors["positive"].append(t_easy_high)
                        errors["negative"].append(t_hard_high)
                    elif index == 1:
                        errors["positive"].append(t_easy_low)
                        errors["negative"].append(t_hard_high)
                    elif index == 2:
                        errors["positive"].append(t_easy_high)
                        errors["negative"].append(t_hard_low)
                    elif index == 3:
                        errors["positive"].append(t_easy_low)
                        errors["negative"].append(t_hard_low)

        error_df = pd.DataFrame().from_dict(errors)
        # if error_path != "":
            # error_df.to_csv(f'{error_path}_special_test_errors.csv', index=False)
        error_df.to_csv(f'{Config.output_path.split("/")[1]}_special_test_errors.csv', index=False)
    
    if not os.path.exists(log_path):
        with open(log_path, mode="w") as out_file:
            writer = csv.writer(out_file)
            writer.writerow(["id"] + list(results.keys()))    
    
    with open(log_path, mode="a") as out_file:
        accs = [np.mean(scores) for scores in results.values()]
        writer = csv.writer(out_file)
        id = f"{Config.data_config}_m={Config.triplet_margin}"
        writer.writerow([id] + accs)


# --- 4. MAIN TRAINING LOOP ---
def main():
    # A. Setup
    import pandas as pd
    # print(test_df)
    # train_df, val_df, test_df = load_and_split_data("llama3-70b_contrastive_noWebis_combined_cleaned_sem_lex_with_high_high_resampled_hard.csv")
    # # 2. Mine Negatives (Takes ~2 mins for 43k rows)
    # # We scan the WHOLE dataset to find the best traps
    # # hard_neg_map_train = mine_hard_negatives(train_df)
    # # hard_neg_map_val = mine_hard_negatives(val_df)
    
    # if Config.data_config == "all":
    #     train_examples = generate_triplets(train_df)
    #     val_examples = generate_triplets(val_df)
    # elif Config.data_config == "noType3":
    #     train_examples = generate_noType3_triplets(train_df)
    #     val_examples = generate_noType3_triplets(val_df)
    # elif Config.data_config == "AugmentationM5":
    #     train_examples = generate_M5_triplets(train_df)
    #     val_examples = generate_M5_triplets(val_df)
    # elif Config.data_config == "AugmentationM6":
    #     train_examples = generate_M6_triplets(train_df)
    #     val_examples = generate_M6_triplets(val_df)
    # # train_examples = generate_surgical_triplets(train_df, hard_neg_map_train)
    # # val_examples = generate_surgical_triplets(val_df, hard_neg_map_val)
    
    # print(f"Generated {len(train_examples)} training triplets.")
    
    # # B. Model Init
    # model = SentenceTransformer(Config.base_model)
    
    # train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=Config.batch_size)
    
    # # C. Loss (Cosine)
    # train_loss = losses.TripletLoss(
    #     model=model,
    #     distance_metric=losses.TripletDistanceMetric.COSINE,
    #     triplet_margin=Config.triplet_margin
    # )
    
    # # D. Evaluator (Using standard TripletEvaluator for 'during-training' checks)
    # # We use a subset of validation data to speed up training checks
    # evaluator = TripletEvaluator.from_input_examples(
    #     val_examples[:300], 
    #     name='val_evaluator',
    #     write_csv=True
    # )
    
    # # E. Train
    # print("Starting Training...")
    # model.fit(
    #     train_objectives=[(train_dataloader, train_loss)],
    #     evaluator=evaluator,
    #     epochs=Config.num_epochs,
    #     warmup_steps=int(len(train_dataloader) * 0.1),
    #     output_path=Config.output_path,
    #     save_best_model=True,
    #     show_progress_bar=True
    # )
    
    # # F. Final Test
    # print("Loading best model for final testing...")
    # special_test_df = pd.read_csv("special_test.csv")
    # best_model = SentenceTransformer(Config.output_path)
    
    # full_eval_path = f"/home/henrike/ARG-NLI_project/code/ID_model_training/ablation_logs/full_eval.csv"
    # special_eval_path = f"/home/henrike/ARG-NLI_project/code/ID_model_training/ablation_logs/special_eval.csv"
    
    # run_full_evaluation(best_model, test_df, full_eval_path)
    # print("logging eval outputs of full evaluation to:", full_eval_path, "...")
    
    print("Loading best model for final testing...")
    special_test_df = pd.read_csv("special_test.csv")
    best_model = SentenceTransformer(Config.output_path)
    
    special_eval_path = f"/home/henrike/ARG-NLI_project/code/ID_model_training/ablation_logs/special_eval.csv"

    eval_with_errors(best_model, special_test_df, special_eval_path)
    print("logging eval outputs of special evaluation to:", full_eval_path, "...")

    

if __name__ == "__main__":
    main()