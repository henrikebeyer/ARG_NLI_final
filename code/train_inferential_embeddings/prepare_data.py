import pandas as pd
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split
import random
import spacy
import random
import numpy as np
from sentence_transformers import util, SentenceTransformer
import torch

# --- Configuartion ---
RANDOM_SEED = 42
TEST_SIZE = 0.15 # 15% for test
VAL_SIZE = 0.15  # 15% for validation

def load_and_split_data(file_path):
    """
    Loads data and splits into ROWS (Antecedents) to prevent data leakage.
    Returns train, val, and test DataFrames
    
    :param file_path: Path to the data file
    """
    
    df = pd.read_csv(file_path)
    
    # First split: separate the test and training data
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    # Second split: Separate train and validation set
    # Size of this set needs to be adjusted to be relative to the remaining data
    relative_val_size = VAL_SIZE / (1-TEST_SIZE)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=RANDOM_SEED,
        shuffle =True
    )
    
    print(f"Data Loaded from {file_path}")
    print(f"Total Rows: {len(df)}")
    print(f"Train Rows: {len(train_df)} | Val Rows: {len(val_df)} | Test Rows: {len(test_df)}")
    
    return train_df, val_df, test_df

def generate_triplets(df):
    """
    generates triplets for BOTH inference and conflict columns.
    Logic: 
        Anchor = Antedent
        Positive (Close) = Low distance (easy to infer)
        Negative (Far) = High distance (hard to infer)
    
    :param df: dataframe to generate triplets from
    """
    
    triplets = []
    
    for idx, row in df.iterrows():
        anchor = row["antecedent"]
        
        # Iterate twice: ince for interence and once for conflict
        # This maps both "easy support" and "easy conflict" to the same "close" space
        for mode in ["inf", "conf"]:
            # --- Identify the columns belonging to the mode ---
            # Columns: low_low, low_high (Easy to infer)
            #          high_high, high_low (hard to infer)
            
            pos_low_sim = row[f"low_low_consequent_{mode}_final"]
            pos_high_sim = row[f"low_high_consequent_{mode}_final"]
            
            neg_low_sim = row[f"high_low_consequent_{mode}"]
            neg_high_sim = row[f"high_high_consequent_{mode}_final"]
            
            # --- Removing prevalence of high semantic similarity ---
            # Easy & high similarity vs hard & high similarity
            # Both will look semantically similart but one is hard to infer
            triplets.append(InputExample(
                texts=[anchor, pos_high_sim, neg_high_sim],
                label = 0
            ))
            
            # --- Disentangle similarity from reasoning ---
            # Easy & low similarity vs hard & high similarity
            # The easy to infer has a low semantic and lexical overlap but is obvious. 
            # The hard to infer has a high semantic and lexical overlap but is hard to infer.
            # This strongly frames the signal for "Reasoning > Wording"
            triplets.append(InputExample(
                texts=[anchor, pos_low_sim, neg_high_sim],
                label=0
            ))
            
            # --- Anchoring in semantics ---
            # Easy & high sim vs hard & low sim
            # This is the easy example for a semantics-focused model, 
            # but we don't want to fully unlearn semantics
            triplets.append(InputExample(
                texts=[anchor, pos_high_sim, neg_low_sim],
                label=0
            ))
            
            # --- Pure reliance on inference ---
            # Easy & low sim vs hard & low sim
            # Both samples have a low semantic and lexical overlap. 
            # The difference is purely inferential distance
            triplets.append(InputExample(
                texts=[anchor, pos_low_sim, neg_low_sim],
                label=0
            ))

    return triplets

def generate_noType3_triplets(df):
    """
    generates triplets for BOTH inference and conflict columns.
    No Type 3 (P=high, N=low)
    Logic: 
        Anchor = Antedent
        Positive (Close) = Low distance (easy to infer)
        Negative (Far) = High distance (hard to infer)
    
    :param df: dataframe to generate triplets from
    """
    
    triplets = []
    
    for idx, row in df.iterrows():
        anchor = row["antecedent"]
        
        # Iterate twice: ince for interence and once for conflict
        # This maps both "easy support" and "easy conflict" to the same "close" space
        for mode in ["inf", "conf"]:
            # --- Identify the columns belonging to the mode ---
            # Columns: low_low, low_high (Easy to infer)
            #          high_high, high_low (hard to infer)
            
            pos_low_sim = row[f"low_low_consequent_{mode}_final"]
            pos_high_sim = row[f"low_high_consequent_{mode}_final"]
            
            neg_low_sim = row[f"high_low_consequent_{mode}"]
            neg_high_sim = row[f"high_high_consequent_{mode}_final"]
            
            # --- Removing prevalence of high semantic similarity ---
            # Easy & high similarity vs hard & high similarity
            # Both will look semantically similart but one is hard to infer
            triplets.append(InputExample(
                texts=[anchor, pos_high_sim, neg_high_sim],
                label = 0
            ))
            
            # --- Disentangle similarity from reasoning ---
            # Easy & low similarity vs hard & high similarity
            # The easy to infer has a low semantic and lexical overlap but is obvious. 
            # The hard to infer has a high semantic and lexical overlap but is hard to infer.
            # This strongly frames the signal for "Reasoning > Wording"
            triplets.append(InputExample(
                texts=[anchor, pos_low_sim, neg_high_sim],
                label=0
            ))
            
            # --- Anchoring in semantics ---
            # Easy & high sim vs hard & low sim
            # This is the easy example for a semantics-focused model, 
            # but we don't want to fully unlearn semantics
            # triplets.append(InputExample(
            #     texts=[anchor, pos_high_sim, neg_low_sim],
            #     label=0
            # ))
            
            # --- Pure reliance on inference ---
            # Easy & low sim vs hard & low sim
            # Both samples have a low semantic and lexical overlap. 
            # The difference is purely inferential distance
            triplets.append(InputExample(
                texts=[anchor, pos_low_sim, neg_low_sim],
                label=0
            ))

    return triplets

def generate_weighted_triplets(df, type2_weight=2):
    """
    Generates triplets but OVERSAMPLES the critical 'Type 2' scenario.
    type2_weight: How many times to repeat the difficult disentanglement triplet.
    """
    triplets = []
    
    for idx, row in df.iterrows():
        anchor = row['antecedent']
        
        for mode in ['inf', 'conf']:
            pos_low_sim  = row[f'low_low_consequent_{mode}_final']
            pos_high_sim = row[f'low_high_consequent_{mode}_final']
            neg_low_sim  = row[f'high_low_consequent_{mode}']
            neg_high_sim = row[f'high_high_consequent_{mode}_final']

            # Type 1: Anti-Semantic (Weight: 1)
            triplets.append(InputExample(texts=[anchor, pos_high_sim, neg_high_sim], label=0))

            # Type 2: Disentanglement (Weight: HIGH)
            # This is where the model is failing. We repeat this example multiple times.
            for _ in range(type2_weight):
                triplets.append(InputExample(texts=[anchor, pos_low_sim, neg_high_sim], label=0))

            # Type 3: Anchoring (Weight: 1)
            triplets.append(InputExample(texts=[anchor, pos_high_sim, neg_low_sim], label=0))
            
            # Type 4: Pure Inference (Weight: 1)
            triplets.append(InputExample(texts=[anchor, pos_low_sim, neg_low_sim], label=0))

    return triplets

def generate_hard_mode_triplets(df):
    """
    Generates ONLY the triplets that force the model to ignore semantics.
    Drops any triplet where semantic similarity correlates with the label.
    """
    triplets = []
    
    for idx, row in df.iterrows():
        anchor = row['antecedent']
        
        for mode in ['inf', 'conf']:
            pos_low_sim  = row[f'low_low_consequent_{mode}_final']  # Easy + Low Sim
            neg_low_sim  = row[f'high_low_consequent_{mode}'] # Hard + Low Sim
            neg_high_sim = row[f'high_high_consequent_{mode}_final']# Hard + High Sim

            # STRATEGY A: The Boss Fight (Type 2)
            # Positive is Semantically Different. Negative is Semantically Similar.
            # The ONLY way to solve this is to know the Inference logic.
            # We add this 2x to emphasize it.
            triplets.append(InputExample(texts=[anchor, pos_low_sim, neg_high_sim], label=0))
            triplets.append(InputExample(texts=[anchor, pos_low_sim, neg_high_sim], label=0))
            
            # STRATEGY B: Pure Inference (Type 4)
            # Both are Semantically Different. Pure reasoning required.
            triplets.append(InputExample(texts=[anchor, pos_low_sim, neg_low_sim], label=0))

    return triplets

# Load spacy for grammatical analysis
nlp = spacy.load("en_core_web_sm")

def swap_nouns(text):
    """
    Identifies nouns and swaps them to create grammatical but logically reversed sentences.
    'The earth orbits the sun' -> 'The sun orbits the earth'
    """
    doc = nlp(text)
    # Find indices of nouns and proper nouns
    noun_indices = [i for i, token in enumerate(doc) if token.pos_ in ["NOUN", "PROPN"]]
    
    if len(noun_indices) < 2:
        return text # Need at least 2 nouns to swap
    
    words = [token.text for token in doc]
    
    # Extract the nouns
    nouns_text = [words[i] for i in noun_indices]
    
    # Shuffle them (derange to ensure they actually move)
    shuffled_nouns = nouns_text[:]
    random.shuffle(shuffled_nouns)
    
    # Apply swap
    for idx, new_noun in zip(noun_indices, shuffled_nouns):
        words[idx] = new_noun
        
    return " ".join(words)

def strict_swap(text):
    if len(text.split())>5:
        try:
            doc = nlp(text)
            verb = [t for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"]
            if not verb: return None
            verb = verb[0]
            subj = [c for c in verb.children if "subj" in c.dep_]
            obj  = [c for c in verb.children if "obj" in c.dep_]
            
            if len(subj) == 1 and len(obj) == 1:
                s_span, o_span = subj[0], obj[0]
                # Get spans indices
                s_min, s_max = min([t.i for t in s_span.subtree]), max([t.i for t in s_span.subtree])
                o_min, o_max = min([t.i for t in o_span.subtree]), max([t.i for t in o_span.subtree])
                
                # Only swap if they don't overlap and are in S-V-O order
                if s_max < verb.i < o_min:
                    # Text reconstruction
                    s_text = doc[s_min:s_max+1].text
                    o_text = doc[o_min:o_max+1].text
                    
                    # Construct: Pre-Subj + OBJ + Mid + SUBJ + Post
                    return (doc[:s_min].text_with_ws + 
                            o_text + " " + # Insert Object
                            doc[s_max+1:o_min].text_with_ws + 
                            s_text + " " + # Insert Subject
                            doc[o_max+1:].text_with_ws).strip()
        except:
            return None
        return None
    else:
        return None

def conservative_swap(text):
    doc = nlp(text)
    
    # 1. Find Root Verb
    verbs = [t for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"]
    if not verbs: return None
    verb = verbs[0]
    
    # 2. Find Subject and Object
    subj = [c for c in verb.children if "subj" in c.dep_]
    obj  = [c for c in verb.children if "obj" in c.dep_]
    
    if len(subj) != 1 or len(obj) != 1: return None
    
    s_span = list(subj[0].subtree)
    o_span = list(obj[0].subtree)
    
    # --- CONSERVATIVE FILTERS ---
    
    # A. Length Filter: simple entities only
    if len(s_span) > 4 or len(o_span) > 4: 
        return None # Too complex, risks "word salad"

    # B. Relative Clause Filter: Avoid swapping "The man who..."
    bad_tokens = {'that', 'which', 'who', 'whose', 'whom'}
    if any(t.text.lower() in bad_tokens for t in s_span + o_span):
        return None

    # C. Pronoun Filter: Don't swap "I", "He", "She" (Case issues: "Him saw I")
    if subj[0].pos_ == "PRON" or obj[0].pos_ == "PRON":
        return None

    # --- RECONSTRUCTION ---
    
    # Convert spans to text
    s_text = "".join([t.text_with_ws for t in s_span]).strip()
    o_text = "".join([t.text_with_ws for t in o_span]).strip()
    
    # Get indices to slice the doc
    s_start, s_end = s_span[0].i, s_span[-1].i
    o_start, o_end = o_span[0].i, o_span[-1].i
    
    # Only handle standard S-V-O order
    if s_end < verb.i < o_start:
        prefix = doc[:s_start].text_with_ws
        middle = doc[s_end+1:o_start].text_with_ws
        suffix = doc[o_end+1:].text_with_ws
        
        # Reconstruction: Swap S and O
        return f"{prefix}{o_text}{middle}{s_text}{suffix}"
        
    return None

def mine_hard_negatives(df, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    print("Mining Hard Negatives (Multi-Candidate Mode)...")
    
    # 1. Collect all potential texts
    all_texts = df['low_low_consequent_inf'].tolist() + df['high_high_consequent_inf'].tolist()
    all_texts = list(set(all_texts)) # Unique pool
    
    # 2. Embed
    miner = SentenceTransformer(model_name)
    corpus_embeddings = miner.encode(all_texts, convert_to_tensor=True, show_progress_bar=True)
    anchors = df['antecedent'].tolist()
    anchor_embeddings = miner.encode(anchors, convert_to_tensor=True, show_progress_bar=True)
    
    # 3. Search: Find Top 10 matches for every anchor
    hits = util.semantic_search(anchor_embeddings, corpus_embeddings, top_k=20)
    
    hard_negatives_map = {}
    
    for idx, result_list in enumerate(hits):
        anchor_text = anchors[idx]
        true_pos = df.iloc[idx]['low_low_consequent_inf']
        
        # Collect up to 3 valid negatives
        valid_negatives = []
        for hit in result_list:
            candidate = all_texts[hit['corpus_id']]
            
            # Condition: High Similarity but NOT the correct answer or the anchor itself
            if candidate != true_pos and candidate != anchor_text:
                valid_negatives.append(candidate)
                if len(valid_negatives) >= 2: # Stop once we have 32
                    break
        
        hard_negatives_map[idx] = valid_negatives
        
    return hard_negatives_map

def ultra_conservative_swap(text):
    # 1. Hard Length Limit
    if len(text.split()) > 20: return None
        
    doc = nlp(text)
    
    # 2. Root Verb Check
    verbs = [t for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"]
    if len(verbs) != 1: return None
    verb = verbs[0]
    
    # 3. Subject/Object Check
    subjs = [c for c in verb.children if "subj" in c.dep_]
    objs  = [c for c in verb.children if "obj" in c.dep_]
    
    if len(subjs) != 1 or len(objs) != 1: return None
    
    subj_token = subjs[0]
    obj_token = objs[0]
    
    # --- CRITICAL FILTER: NO PRONOUNS ---
    # Fixes "migrating we" -> "migrating us" issues.
    # We only allow proper nouns (PROPN) or regular nouns (NOUN)
    if subj_token.pos_ not in ["NOUN", "PROPN"]: return None
    if obj_token.pos_ not in ["NOUN", "PROPN"]: return None
    
    # 4. Span Check
    subj_span = list(subj_token.subtree)
    obj_span = list(obj_token.subtree)
    
    # Keep it simple (no relative clauses)
    if len(subj_span) > 4 or len(obj_span) > 4: return None
    
    # 5. Passive Voice Filter
    if "pass" in subj_token.dep_: return None

    # --- RECONSTRUCTION ---
    s_start, s_end = subj_span[0].i, subj_span[-1].i
    o_start, o_end = obj_span[0].i, obj_span[-1].i
    
    # Ensure S-V-O order
    if s_end < verb.i < o_start:
        
        # Helper to get text
        def get_text(start, end):
            return doc[start:end+1].text_with_ws
            
        new_text = (
            doc[:s_start].text_with_ws +       # Prefix
            get_text(o_start, o_end).strip() + # Object (Moved to front)
            " " + 
            doc[s_end+1:o_start].text_with_ws + # Middle/Verb
            get_text(s_start, s_end).strip() +  # Subject (Moved to back)
            " " + 
            doc[o_end+1:].text_with_ws         # Suffix
        )
        return new_text.strip()
        
    return None

def generate_surgical_triplets(df, hard_neg_map):
    triplets = []
    
    for idx, row in df.iterrows():
        anchor = row['antecedent']
        
        # Get Mined Hard Negative (The "Pasta" Trap)
        # This is a REAL, GRAMMATICAL sentence from another context
        mined_negs = hard_neg_map.get(idx, [])
        
        
        for mode in ['inf', 'conf']:
            pos_low  = row[f'low_low_consequent_{mode}_final'] # Easy / Low Sim
            pos_high = row[f'low_high_consequent_{mode}_final'] # Easy / High Sim
            neg_high = row[f'high_high_consequent_{mode}_final'] # Hard / High Sim
            neg_low  = row[f'high_low_consequent_{mode}'] # Hard / Low Sim

            # Strategy 1: Boss Fight (Weight 2)
            triplets.append(InputExample(texts=[anchor, pos_low, neg_high], label=0))
            triplets.append(InputExample(texts=[anchor, pos_low, neg_high], label=0))

            # Strategy 2: Anchoring (Weight 1)
            # Crucial to prevent "Contrarian" behavior
            triplets.append(InputExample(texts=[anchor, pos_high, neg_low], label=0))
            
            # Strategy 3: Mined "Pasta Trap" (Weight 2)
            # Anchor + Valid Pos + Valid Grammatical Negative from other context
            if len(mined_negs) >= 2:
                # Use two different traps
                triplets.append(InputExample(texts=[anchor, pos_low, mined_negs[0]], label=0))
                triplets.append(InputExample(texts=[anchor, pos_low, mined_negs[1]], label=0))
            elif len(mined_negs) == 1:
                # Fallback: Use the same one twice
                triplets.append(InputExample(texts=[anchor, pos_low, mined_negs[0]], label=0))
                triplets.append(InputExample(texts=[anchor, pos_low, mined_negs[0]], label=0))
            
            
            # Strategy 4: Strict Subject Swap (Weight 2)
            # Only if the swap is clean and valid
            swapped_neg = ultra_conservative_swap(anchor)
            if swapped_neg and swapped_neg != anchor:
                # print(anchor)
                # print(swapped_neg)
                # print("*"*20, "\n")
                
                # Pair 1: Hard Pos vs Syntactic Neg (Forces Logic > Syntax)
                triplets.append(InputExample(texts=[anchor, pos_low, swapped_neg], label=0))
                
                # Pair 2: Easy Pos vs Syntactic Neg (Standard grounding)
                triplets.append(InputExample(texts=[anchor, pos_high, swapped_neg], label=0))
                
            # Reintroducing Type 4
            # triplets.append(InputExample(texts=[anchor, pos_low, neg_low], label=0))

    return triplets

def generate_M5_triplets(df):
    triplets = []
    
    print("Generating M5 Triplets (Type 4 Focused)...")
    
    for idx, row in df.iterrows():
        anchor = row['antecedent']
        #mined_negs = hard_neg_map.get(idx, [])
        
        # Swapping the ANCHOR creates a Type 2 (Boss Fight) Negative
        swapped_anchor = ultra_conservative_swap(anchor)
        
        for mode in ['inf', 'conf']:
            # P_low: The Valid, Low-Similarity Consequent (Target for Type 4)
            pos_low  = row[f'low_low_consequent_{mode}_final'] 
            
            # P_high & N_high: Used for Type 1/2
            pos_high = row[f'low_high_consequent_{mode}_final'] 
            neg_high = row[f'high_high_consequent_{mode}_final']
            neg_low  = row[f'high_low_consequent_{mode}']

            # --- 1. CORE CURRICULUM (M3 Base) ---
            
            # Type 1: Anti-Semantic
            triplets.append(InputExample(texts=[anchor, pos_high, neg_high], label=0))

            # Type 2: Boss Fight (High Sim Negative)
            triplets.append(InputExample(texts=[anchor, pos_low, neg_high], label=0))
            
            # Type 2 Augmentation: Swapped Anchor
            # (Anchor vs Swapped Anchor is High Sim)
            if swapped_anchor and swapped_anchor != anchor:
                triplets.append(InputExample(texts=[anchor, pos_low, swapped_anchor], label=0))

            # --- 2. TYPE 4 AUGMENTATION (The Fix) ---
            
            # Standard Type 4 (Mined from Corpus)
            # P=LowSim vs N=LowSim (Existing)
            triplets.append(InputExample(texts=[anchor, pos_low, neg_low], label=0))

            # Augmentation A: The "Consequent Swap" (NEW)
            # Swap the POSITIVE to create a High-Quality Type 4 Negative.
            # This ensures Sim(A, P) == Sim(A, N) perfectly.
            swapped_pos = ultra_conservative_swap(pos_low)
            
            if swapped_pos and swapped_pos != pos_low:
                # This is a "Perfect" Type 4 Triplet
                triplets.append(InputExample(texts=[anchor, pos_low, swapped_pos], label=0))
                # Weight x2 because it's rare and valuable
                # triplets.append(InputExample(texts=[anchor, pos_low, swapped_pos], label=0))
            
            # Augmentation B: Mined Hard Negatives (From your script)
            # These are Type 4 candidates (Low Sim to Anchor)
            # if len(mined_negs) > 0:
            #     # Use the first mined negative
            #     triplets.append(InputExample(texts=[anchor, pos_low, mined_negs[0]], label=0))
                
            #     # If we have more, use them to flood Type 4
            #     if len(mined_negs) > 1:
            #         triplets.append(InputExample(texts=[anchor, pos_low, mined_negs[1]], label=0))

    return triplets

def generate_M6_triplets(df):
    triplets = []
    
    print("Generating M6 Triplets (with all types and augmentation of Type 4)...")
    
    for idx, row in df.iterrows():
        anchor = row['antecedent']
        #mined_negs = hard_neg_map.get(idx, [])
        
        # Swapping the ANCHOR creates a Type 2 (Boss Fight) Negative
        swapped_anchor = ultra_conservative_swap(anchor)
        
        for mode in ['inf', 'conf']:
            # P_low: The Valid, Low-Similarity Consequent (Target for Type 4)
            pos_low  = row[f'low_low_consequent_{mode}_final'] 
            
            # P_high & N_high: Used for Type 1/2
            pos_high = row[f'low_high_consequent_{mode}_final'] 
            neg_high = row[f'high_high_consequent_{mode}_final']
            neg_low  = row[f'high_low_consequent_{mode}']

            # --- 1. CORE CURRICULUM (M3 Base) ---
            
            # Type 1: Anti-Semantic
            triplets.append(InputExample(texts=[anchor, pos_high, neg_high], label=0))

            # Type 2: Boss Fight (High Sim Negative)
            triplets.append(InputExample(texts=[anchor, pos_low, neg_high], label=0))
            
            # Type 2 Augmentation: Swapped Anchor
            # (Anchor vs Swapped Anchor is High Sim)
            if swapped_anchor and swapped_anchor != anchor:
                triplets.append(InputExample(texts=[anchor, pos_low, swapped_anchor], label=0))
                
            # Type 3: Anchoring (Low Sim Negative)
            triplets.append(InputExample(texts=[anchor, pos_high, neg_low], label=0))

            # --- 2. TYPE 4 AUGMENTATION (The Fix) ---
            
            # Standard Type 4 (Mined from Corpus)
            # P=LowSim vs N=LowSim (Existing)
            triplets.append(InputExample(texts=[anchor, pos_low, neg_low], label=0))
            triplets.append(InputExample(texts=[anchor, pos_low, neg_low], label=0))

            # Augmentation A: The "Consequent Swap" (NEW)
            # Swap the POSITIVE to create a High-Quality Type 4 Negative.
            # This ensures Sim(A, P) == Sim(A, N) perfectly.
            swapped_pos = ultra_conservative_swap(pos_low)
            
            if swapped_pos and swapped_pos != pos_low:
                # This is a "Perfect" Type 4 Triplet
                triplets.append(InputExample(texts=[anchor, pos_low, swapped_pos], label=0))
                # Weight x2 because it's rare and valuable
                # triplets.append(InputExample(texts=[anchor, pos_low, swapped_pos], label=0))
            
            # Augmentation B: Mined Hard Negatives (From your script)
            # These are Type 4 candidates (Low Sim to Anchor)
            # if len(mined_negs) > 0:
            #     # Use the first mined negative
            #     triplets.append(InputExample(texts=[anchor, pos_low, mined_negs[0]], label=0))
                
            #     # If we have more, use them to flood Type 4
            #     if len(mined_negs) > 1:
            #         triplets.append(InputExample(texts=[anchor, pos_low, mined_negs[1]], label=0))

    return triplets