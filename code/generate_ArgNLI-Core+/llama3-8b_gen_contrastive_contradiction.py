from ollama import generate
import pandas as pd
import re
from difflib import SequenceMatcher
import numpy as np
import argparse
import os
import math
import time

try:
    from sentence_transformers import SentenceTransformer
    from numpy import dot
    from numpy.linalg import norm
    _HAS_ST = True
except Exception:
    _HAS_ST = False

data_path = "/home/henrike/ARG-NLI_project/gen_contrastive_samples/corpus/corpus_nli_preds_threshold_hard_0.999.tsv"
data = pd.read_csv(data_path, sep="\t")

data_sample = data.sample(n=10, random_state=42)

# seg2 = antecedent
antecedents = data_sample["antecedent"]

# Prompt1
# low_high_system_prompt = """Generate a statement that is argumentatively supported by the given statement and is easy to infer from this statement. 
# Make sure that there is a high level of lexical and semantic overlap between the statement you get and the statement you generate. 
# For example:  'John likes to eat bananas.' -> 'Therefore, he eats bananas for breakfast every morning.';  
# Don't add prefixes or explanations to the output."""

# low_low_system_prompt = """Generate a statement that is argumentatively supported by the given statement and is easy to infer from this statement. 
# Make sure that there is as little lexical and semantic overlap as possible between the statement you get and the statement you generate. 
# For example:  "The sky is blue." -> "Therefore, the weather is good.";  
# Don't add prefixes or explanations to the output."""

# Prompt2
# low_high_system_prompt = """Generate a statement that is argumentatively supported by the given statement and is easy to infer from this statement. 
# Use as many words as possible from the given statement to generate your statement. 
# For example:  'John likes to eat bananas.' -> 'Therefore, John likes to eat bananas for breakfast every morning.';  
# Don't add prefixes or explanations to the output."""

low_low_system_prompt = """Generate a statement that argumentatively attacks the given statement while this attack relation is easy to infer. 
Use as few words from the given statement as possible to generate your statement.
For example:  "The sky is blue." -> "We are expecting plenty of rain.";  
Don't add prefixes or explanations to the output."""

# Prompt3
low_high_system_prompt = """Generate a statement that trivially contradicts from the given statement and this contradiction is easy to infer from this statement. 
Use as many words as possible from the given statement to generate your statement. 
For example:  'John likes to eat bananas.' -> 'John does not like to eat bananas.';  
Don't add prefixes or explanations to the output."""


high_high_system_prompt = """Generate a statement that argumentatively attacks the given statement but is hard to infer from the given statement. 
Use as many words as possible from the given statement to generate your statement.
For example: 'John likes to eat bananas.' -> 'Bananas are never harvested to eat them.' or
'John likes to eat bananas.' -> 'Bananas like to eat John.'
Don't add prefixes or explanations to the output."""

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def jaccard_tokens(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 1.0
    inter = ta.intersection(tb)
    union = ta.union(tb)
    return len(inter) / len(union)

def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def cosine_sim(a_vec: np.ndarray, b_vec: np.ndarray) -> float:
    if a_vec is None or b_vec is None:
        return float('nan')
    denom = (norm(a_vec) * norm(b_vec))
    if denom == 0:
        return float('nan')
    return float(dot(a_vec, b_vec) / denom)

# prepare sentence transformer if available
st_model = None
if _HAS_ST:
    try:
        st_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception:
        st_model = None

def generate_with_params(prompt, target_sim="low", target_dist = "low", model='llama3.1:70b',):
    """Generate text with custom parameters"""
    options = {}
    if target_sim == "low" and target_dist == "low":
        prompt = low_low_system_prompt + "\n" + prompt
        options={
            'temperature': 0.7,     # Creativity level (0.0 to 1.0)
            'top_p': 0.5,          # Nucleus sampling
            'top_k': 40,           # Top-k sampling
            'repeat_penalty': 1.3,  # Penalty for repetition
            'num_ctx': 2048,       # Context window size
        }
    elif target_sim == "high" and target_dist == "low":
        prompt = low_high_system_prompt + "\n" + prompt
        options={
            'temperature': 0.4,     # Creativity level (0.0 to 1.0)
            'top_p': 0.9,          # Nucleus sampling
            'top_k': 20,           # Top-k sampling
            'repeat_penalty': 1.0,  # Penalty for repetition
            'num_ctx': 2048,       # Context window size
        }
        
    elif target_sim == "high" and target_dist == "high":
        prompt = high_high_system_prompt + "\n" + prompt
        options={
            'temperature': 1.0,     # Creativity level (0.0 to 1.0)
            'top_p': 0.7,          # Nucleus sampling
            'top_k': 60,           # Top-k sampling
            'repeat_penalty': 1.0,  # Penalty for repetition
            'num_ctx': 2048,       # Context window size
        }
        
    response = generate(
        model=model,
        prompt=prompt,
        options = options        
    )
    return response['response']

def process_chunk(chunk_df, model, output_path, append_mode=False):
    low_low = []
    low_high = []
    high_high = []
    
    
    for antecedent in chunk_df["antecedent"]:
        ll = generate_with_params(prompt=antecedent, target_sim="low", target_dist="low")
        lh = generate_with_params(prompt=antecedent, target_sim="high", target_dist="low")
        hh = generate_with_params(prompt=antecedent, target_sim="high", target_dist="high")

        low_low.append(ll)
        low_high.append(lh)
        high_high.append(hh)
    
    out_df = pd.DataFrame({
        "ID": chunk_df["ID"].values,
        "antecedent": chunk_df["antecedent"].values,
        "low_low_consequent": low_low,
        "low_high_consequent": low_high,
        "high_high_consequent": high_high
    })

    # Compute similarity measures
    lex_j_low = []
    lex_j_high = []
    lex_j_high_high = []
    sem_c_low = []
    sem_c_high = []
    sem_c_high_high = []

    for ant, c_lowlow, c_lowhigh, c_highhigh in zip(out_df['antecedent'], out_df['low_low_consequent'], out_df['low_high_consequent'], out_df['high_high_consequent']):
        a = normalize_text(str(ant))
        ll = normalize_text(str(c_lowlow))
        lh = normalize_text(str(c_lowhigh))
        hh = normalize_text(str(c_highhigh))
    

        # lexical: jaccard on tokens
        lex_low = jaccard_tokens(a, ll)
        lex_high = jaccard_tokens(a, lh)
        lex_high_high = jaccard_tokens(a, hh)

        # sequence (character) similarity
        # seq_low = seq_ratio(a, l)
        # seq_high = seq_ratio(a, h)
        # seq_between = seq_ratio(l, h)

        # semantic similarity via sentence-transformers if available
        if st_model is not None:
            try:
                emb_a = st_model.encode([ant])[0]
                emb_ll = st_model.encode([c_lowlow])[0]
                emb_lh = st_model.encode([c_lowhigh])[0]
                emb_hh = st_model.encode([c_highhigh])[0]
                sem_low_low = cosine_sim(emb_a, emb_ll)
                sem_low_high = cosine_sim(emb_a, emb_lh)
                sem_high_high = cosine_sim(emb_a, emb_hh)
                # sem_between = cosine_sim(emb_l, emb_h)
            except Exception:
                sem_low_low = sem_low_high = sem_high_high = float('nan')
        else:
            sem_lowlow = sem_lowhigh = sem_high_high = float('nan')

        lex_j_low.append(lex_low)
        lex_j_high.append(lex_high)
        lex_j_high_high.append(lex_high_high)

        sem_c_low.append(sem_low_low)
        sem_c_high.append(sem_low_high)
        sem_c_high_high.append(sem_high_high)
    
    out_df['lex_jaccard_low'] = lex_j_low
    out_df['lex_jaccard_high'] = lex_j_high
    out_df['lex_jaccard_high_high'] = lex_j_high_high

    out_df['sem_sim_low'] = sem_c_low
    out_df['sem_sim_high'] = sem_c_high
    out_df['sem_sim_high_high'] = sem_c_high_high

    out_df['lowhigh_more_similar_jaccard'] = out_df['lex_jaccard_low'] < out_df['lex_jaccard_high']
    out_df['lowhigh_more_similar_sem'] = out_df['sem_sim_low'] < out_df['sem_sim_high']
    out_df['highhigh_more_similar_jaccard'] = out_df['lex_jaccard_low'] < out_df['lex_jaccard_high_high']
    out_df['highhigh_more_similar_sem'] = out_df['sem_sim_low'] < out_df['sem_sim_high_high']

    # save chunk: append if requested
    header = not (append_mode and os.path.exists(output_path))
    out_df.to_csv(output_path, sep='\t', index=False, mode='a' if append_mode else 'w', header=header)
    return out_df

def main():
    parser = argparse.ArgumentParser(description="Generate contrastive consequents in chunks with resume and preview options.")
    parser.add_argument("--input", "-i", default="/home/henrike/ARG-NLI_project/gen_contrastive_samples/corpus/corpus_nli_preds_threshold_hard_0.999.tsv")
    parser.add_argument("--output", "-o", default="/home/henrike/ARG-NLI_project/gen_contrastive_samples/sample_test/test_llama3-70b_contrastive_with_sims_prompt3_params_highdist_contra2.tsv")
    parser.add_argument("--chunk-size", type=int, default=500, help="Number of rows to process per chunk.")
    parser.add_argument("--model", type=str, default="llama3.1:70b")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output; skip already processed IDs.")
    parser.add_argument("--preview", action="store_true", help="Run a preliminary generation on a small random subset.")
    parser.add_argument("--preview-size", type=int, default=100, help="Number of samples for preview (when --preview is set).")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.input, sep="\t")
    if "ID" not in data.columns:
        data = data.reset_index().rename(columns={"index":"ID"})

    if args.preview:
        sample_df = data.sample(n=min(args.preview_size, len(data)), random_state=args.random_seed)
        print(f"Running preview on {len(sample_df)} samples -> saving to {args.output}")
        try:
            start = time.time()
            process_chunk(sample_df, model=args.model, output_path=args.output, append_mode=False)
            elapsed = time.time() - start
            per_sample = elapsed / len(sample_df) if len(sample_df) > 0 else float('nan')
            estimated_total_seconds = per_sample * len(data)
            print(f"Preview time: {elapsed:.2f}s ({per_sample:.2f}s/sample). Estimated total for {len(data)} samples: {estimated_total_seconds:.2f}s ({estimated_total_seconds/3600:.2f}h)")
        except Exception as e:
            print("Preview generation failed:", e)
        return

    # normal run with chunking + resume
    remaining_df = data
    append_mode = False
    if args.resume and os.path.exists(args.output):
        try:
            processed = pd.read_csv(args.output, sep="\t", usecols=["ID"])
            processed_ids = set(processed["ID"].tolist())
            remaining_df = data[~data["ID"].isin(processed_ids)].reset_index(drop=True)
            append_mode = True
            print(f"Resuming: {len(processed_ids)} already processed, {len(remaining_df)} remaining.")
        except Exception as e:
            print("Could not read existing output for resume, starting fresh:", e)
            remaining_df = data
            append_mode = False

    total = len(remaining_df)
    if total == 0:
        print("Nothing to do. Exiting.")
        return

    n_chunks = math.ceil(total / args.chunk_size)
    for idx in range(n_chunks):
        start = idx * args.chunk_size
        end = min(start + args.chunk_size, total)
        chunk_df = remaining_df.iloc[start:end].reset_index(drop=True)
        print(f"Processing chunk {idx+1}/{n_chunks} (rows {start}..{end-1}) size={len(chunk_df)}")
        try:
            process_chunk(chunk_df, model=args.model, output_path=args.output, append_mode=append_mode)
            append_mode = True  # after first write, always append
        except Exception as e:
            # save what we can and exit to allow resume
            print(f"Error during chunk {idx+1}: {e}. Saved completed chunks; you can resume with --resume.")
            return

    print(f"Wrote results with similarity columns to {args.output}")

if __name__ == "__main__":
    main()