import pandas as pd
import os

config = "hard" #, "soft"]
corpus_path = "C:\\Users\\henri\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\corpus\\corpus_nli_preds_threshold"

def read_in_corpus(path, config):
    df = pd.read_csv(f"{path}_{config}_0.999.tsv", sep="\t")#.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
    return df

df = read_in_corpus(corpus_path, config)

output_prefix = f"C:\\Users\\henri\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\automatic\\new_features\\single_feature_categories\\{config}_"

def add_structural_feats(df, batch_size=100, path="structural_features.csv"):
    """
    Extracts structural features in batches and saves intermediate results.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with "seg1_text" and "seg2_text".
    - batch_size (int): Number of rows to process per batch.
    - output_path (str): Path to save the output CSV file.
    """

    output_path = output_prefix + path

    import textdescriptives as td

    keep = ['sentence_length_mean', 
            'sentence_length_median', 
            'sentence_length_std', 
            'n_tokens',
            'n_sentences', 
            'dependency_distance_mean',
            'dependency_distance_std',
            'prop_adjacent_dependency_relation_mean',
            'prop_adjacent_dependency_relation_std']

    if not os.path.exists(output_path):
        pd.DataFrame(columns=[f"seg1_{col}" for col in keep] + [f"seg2_{col}" for col in keep]).to_csv(output_path, index=False)

    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]

        seg1s = [str(seg1) for seg1 in batch["seg1_text"]]
        seg2s = [str(seg2) for seg2 in batch["seg2_text"]]

        df1 = td.extract_metrics(text=seg1s, spacy_model="en_core_web_lg", metrics=["descriptive_stats", "dependency_distance"])
        df1 = df1.loc[:, keep]
        df1.rename(columns={col: f"seg1_{col}" for col in df1.columns}, inplace=True)

        df2 = td.extract_metrics(text=seg2s, spacy_model="en_core_web_lg", metrics=["descriptive_stats", "dependency_distance"])
        df2 = df2.loc[:, keep]
        df2.rename(columns={col: f"seg2_{col}" for col in df2.columns}, inplace=True)

        batch_features = pd.concat([df1, df2], axis=1)
        batch_features.to_csv(output_path, mode="a", header=False, index=False)

        print(f"Processed batch {start} to {end}")

def add_cosine_sim(df, batch_size=100, path="cosine_similarity_features.csv"):
    """
    Computes cosine similarity in batches and saves intermediate results.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with "seg1_text" and "seg2_text".
    - batch_size (int): Number of rows to process per batch.
    - output_path (str): Path to save the output CSV file.
    """
    output_path = output_prefix + path

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer('all-MiniLM-L6-v2')

    if not os.path.exists(output_path):
        pd.DataFrame(columns=["cosine_similarity"]).to_csv(output_path, index=False)

    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]

        seg1_embeddings = model.encode(batch["seg1_text"].astype(str).tolist(), convert_to_tensor=True)
        seg2_embeddings = model.encode(batch["seg2_text"].astype(str).tolist(), convert_to_tensor=True)

        similarities = cosine_similarity(seg1_embeddings.cpu().numpy(), seg2_embeddings.cpu().numpy())
        batch["cosine_similarity"] = [similarities[i, i] for i in range(len(batch))]

        batch[["cosine_similarity"]].to_csv(output_path, mode="a", header=False, index=False)

        print(f"Processed batch {start} to {end}")


def add_syntactic_features(df, batch_size=100, path="syntactic_features.csv", max_sentence_length=100):
    """
    Computes syntactic features in batches and saves intermediate results.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with "seg1_text" and "seg2_text".
    - batch_size (int): Number of rows to process per batch.
    - output_path (str): Path to save the output CSV file.
    - max_sentence_length (int): Maximum allowed sentence length for parsing.
    """
    import benepar
    from benepar import Parser
    import nltk

    output_path = output_prefix + path

    #benepar.download('benepar_en3')
    # nltk.download('punkt_tab')
    parser = Parser("benepar_en3")

    if not os.path.exists(output_path):
        pd.DataFrame(columns=["seg1_avg_subclauses", "seg1_avg_tree_depth",
                              "seg2_avg_subclauses", "seg2_avg_tree_depth"]).to_csv(output_path, index=False)

    def compute_syntactic_features(text):
        sentences = nltk.sent_tokenize(text)
        total_subclauses = 0
        total_depth = 0
        valid_sentences = 0

        for sentence in sentences:
            if len(sentence.split()) > max_sentence_length:
                continue  # Skip overly long sentences

            try:
                tree = parser.parse(sentence)
                subclauses = sum(1 for _ in tree.subtrees(lambda t: t.label() != "S"))
                total_subclauses += subclauses
                total_depth += tree.height()
                valid_sentences += 1
            except Exception as e:
                print(f"Error parsing sentence: {sentence[:512]}...\\n{e}")

        avg_subclauses = total_subclauses / valid_sentences if valid_sentences > 0 else 0
        avg_depth = total_depth / valid_sentences if valid_sentences > 0 else 0
        return avg_subclauses, avg_depth

    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]

        seg1_features = batch["seg1_text"].astype(str).apply(compute_syntactic_features)
        batch.loc[start:end-1, "seg1_avg_subclauses"] = seg1_features.apply(lambda x: x[0]).values
        batch.loc[start:end-1, "seg1_avg_tree_depth"] = seg1_features.apply(lambda x: x[1]).values

        seg2_features = batch["seg2_text"].astype(str).apply(compute_syntactic_features)
        batch.loc[start:end-1, "seg2_avg_subclauses"] = seg2_features.apply(lambda x: x[0]).values
        batch.loc[start:end-1, "seg2_avg_tree_depth"] = seg2_features.apply(lambda x: x[1]).values

        batch[["seg1_avg_subclauses", "seg1_avg_tree_depth",
               "seg2_avg_subclauses", "seg2_avg_tree_depth"]].to_csv(output_path, mode="a", header=False, index=False)

        print(f"Processed batch {start} to {end}")


def add_sentiment(df, batch_size=100, path="sentiment_features.csv"):
    """
    Computes sentiment scores in batches and saves intermediate results.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with "seg1_text" and "seg2_text".
    - batch_size (int): Number of rows to process per batch.
    - output_path (str): Path to save the output CSV file.
    """
    from flair.models import TextClassifier
    from flair.data import Sentence

    classifier = TextClassifier.load("en-sentiment")
    output_path = output_prefix + path

    if not os.path.exists(output_path):
        pd.DataFrame(columns=["seg1_senti", "seg2_senti"]).to_csv(output_path, index=False)

    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]

        labels_seg1 = []
        labels_seg2 = []

        for seg1, seg2 in zip(batch["seg1_text"], batch["seg2_text"]):
            sentence1 = Sentence(str(seg1))
            classifier.predict(sentence1)
            labels_seg1.append(sentence1.labels[0].value)

            sentence2 = Sentence(str(seg2))
            classifier.predict(sentence2)
            labels_seg2.append(sentence2.labels[0].value)

        batch["seg1_senti"] = labels_seg1
        batch["seg2_senti"] = labels_seg2

        batch[["seg1_senti", "seg2_senti"]].to_csv(output_path, mode="a", header=False, index=False)

        print(f"Processed batch {start} to {end}")

def add_tfidf_cosine_similarity(df, batch_size=100, path="tfidf_cosine_similarity_features.csv"):
    """
    Computes TF-IDF cosine similarity in batches and saves intermediate results.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with "seg1_text" and "seg2_text".
    - batch_size (int): Number of rows to process per batch.
    - output_path (str): Path to save the output CSV file.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    output_path = output_prefix + path

    if not os.path.exists(output_path):
        pd.DataFrame(columns=["tfidf_cosine_similarity"]).to_csv(output_path, index=False)

    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]

        # Combine all text for TF-IDF fitting
        all_text = batch["seg1_text"].astype(str).tolist() + batch["seg2_text"].astype(str).tolist()

        # Fit and transform the text
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_text)

        # Split the TF-IDF matrix into seg1 and seg2 parts
        seg1_tfidf = tfidf_matrix[:len(batch)]
        seg2_tfidf = tfidf_matrix[len(batch):]

        # Compute cosine similarity for each pair of segments
        similarities = cosine_similarity(seg1_tfidf, seg2_tfidf).diagonal()

        # Save intermediate results
        batch["tfidf_cosine_similarity"] = similarities
        batch[["tfidf_cosine_similarity"]].to_csv(output_path, mode="a", header=False, index=False)

        print(f"Processed batch {start} to {end}")

def add_perplexity_measures_llama(df, batch_size=10, path="perplexity_features.csv", model_name="meta-llama/Llama-3.1-8B"):
    """
    Computes perplexity measures in batches and saves intermediate results.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with "seg1_text" and "seg2_text".
    - batch_size (int): Number of rows to process per batch.
    - output_path (str): Path to save the output CSV file.
    - model_name (str): The Hugging Face model name for the LLaMA model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    output_path = output_prefix + path

    # Load the LLaMA model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()

    output_path = output_prefix + path

    if not os.path.exists(output_path):
        pd.DataFrame(columns=["seg1_perplexity", "seg2_perplexity",
                              "seg1_given_seg2_perplexity", "seg2_given_seg1_perplexity"]).to_csv(output_path, index=False)

    def compute_perplexity(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()

    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]

        seg1_perplexities = []
        seg2_perplexities = []
        seg1_given_seg2_perplexities = []
        seg2_given_seg1_perplexities = []

        for seg1, seg2 in zip(batch["seg1_text"], batch["seg2_text"]):
            seg1 = str(seg1)
            seg2 = str(seg2)

            # Compute perplexities
            seg1_perplexities.append(compute_perplexity(seg1))
            seg2_perplexities.append(compute_perplexity(seg2))
            seg1_given_seg2_perplexities.append(compute_perplexity(seg2 + " " + seg1))
            seg2_given_seg1_perplexities.append(compute_perplexity(seg1 + " " + seg2))

        # Save intermediate results
        batch["seg1_perplexity"] = seg1_perplexities
        batch["seg2_perplexity"] = seg2_perplexities
        batch["seg1_given_seg2_perplexity"] = seg1_given_seg2_perplexities
        batch["seg2_given_seg1_perplexity"] = seg2_given_seg1_perplexities

        batch[["seg1_perplexity", "seg2_perplexity",
               "seg1_given_seg2_perplexity", "seg2_given_seg1_perplexity"]].to_csv(output_path, mode="a", header=False, index=False)

        print(f"Processed batch {start} to {end}")


def add_lexical_features(df, batch_size=100, path="lexical_features.csv"):
    """
    Extracts lexical features in batches and resumes calculations if interrupted.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with "seg1_text" and "seg2_text".
    - batch_size (int): Number of rows to process per batch.
    - output_path (str): Path to save the output CSV file.
    """
    import spacy
    output_path = output_prefix + path
    nlp = spacy.load("en_core_web_sm")

    file_paths = {
        "deduction": "C:\\Users\\henri\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\reference_lists\\deduction_markers.csv",
        "refutation": "C:\\Users\\henri\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\reference_lists\\refutation_markers.csv",
        "condition": "C:\\Users\\henri\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\reference_lists\\condition_markers.csv",
        "explanation": "C:\\Users\\henri\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\reference_lists\\explanation_markers.csv",
        "adverbs_of_emphasis": "C:\\Users\\henri\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\reference_lists\\emphasis_adverbs.csv",
        "justification": "C:\\Users\\henri\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\reference_lists\\justification_markers.csv"
    }

    marker_dict = {key: pd.read_csv(path)["Marker"].str.strip().str.lower().tolist() for key, path in file_paths.items()}

    # Check if the output file exists and determine the starting point
    if os.path.exists(output_path):
        processed_df = pd.read_csv(output_path)
        processed_ids = set(processed_df["ID"])  # Track already processed IDs
    else:
        processed_ids = set()
        pd.DataFrame(columns=["ID"]+[f"seg1_has_{key}" for key in marker_dict.keys()] +
                              [f"seg2_has_{key}" for key in marker_dict.keys()] +
                              ["seg1_has_modal_verbs", "seg2_has_modal_verbs"]).to_csv(output_path, index=False)

    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]

        # Skip already processed rows
        batch = batch[~batch["ID"].isin(processed_ids)]
        if batch.empty:
            continue

        seg1_features = {key: [] for key in marker_dict.keys()}
        seg2_features = {key: [] for key in marker_dict.keys()}
        seg1_modal_verbs = []
        seg2_modal_verbs = []

        for seg1_text, seg2_text in zip(batch["seg1_text"], batch["seg2_text"]):
            seg1_text_lower = str(seg1_text).lower()
            seg2_text_lower = str(seg2_text).lower()

            # Check for lexical markers
            for key, markers in marker_dict.items():
                seg1_features[key].append(any(marker in seg1_text_lower for marker in markers))
                seg2_features[key].append(any(marker in seg2_text_lower for marker in markers))

            # Check for modal verbs
            seg1_doc = nlp(str(seg1_text))
            seg2_doc = nlp(str(seg2_text))
            seg1_modal_verbs.append(any(token.tag_ == "MD" for token in seg1_doc))
            seg2_modal_verbs.append(any(token.tag_ == "MD" for token in seg2_doc))

        # Save intermediate results
        for key in marker_dict.keys():
            batch[f"seg1_has_{key}"] = seg1_features[key]
            batch[f"seg2_has_{key}"] = seg2_features[key]
        batch["seg1_has_modal_verbs"] = seg1_modal_verbs
        batch["seg2_has_modal_verbs"] = seg2_modal_verbs

        batch[["ID"] + [f"seg1_has_{key}" for key in marker_dict.keys()] +
              [f"seg2_has_{key}" for key in marker_dict.keys()] +
              ["seg1_has_modal_verbs", "seg2_has_modal_verbs"]].to_csv(output_path, mode="a", header=False, index=False)

        print(f"Processed batch {start} to {end}")


def add_imp_score_features(df, path="imp_score_features.csv"):
    import impscore
    output_path = output_prefix + path
    s1_list = [str(s) for s in df["seg1_text"]]
    s2_list = [str(s) for s in df["seg2_text"]]

    model = impscore.load_model(load_device = "cuda")

    imp_score1, imp_score2, prag_distance = model.infer_pairs(s1_list, s2_list)


    imp_score1_np = imp_score1.cpu().detach().numpy()
    imp_score2_np = imp_score2.cpu().detach().numpy()
    prag_distance_np = prag_distance.cpu().detach().numpy()

    df["seg1_impScore"] = imp_score1_np
    df["seg2_impScore"] = imp_score2_np
    df["prag_distance"] = prag_distance_np

    df.to_csv(output_path, index=False)

    # return df



add_perplexity_measures_llama(df, batch_size=50, path="perplexity_features.csv", model_name="gpt2")
# add_cosine_sim(df, batch_size=100, path="cosine_similarity_features.csv")
# add_structural_feats(df, batch_size=100, path="structural_features.csv")
# add_syntactic_features(df, batch_size=100, path="syntactic_features.csv")
# add_sentiment(df, batch_size=100, path="sentiment_features.csv")
# add_tfidf_cosine_similarity(df, batch_size=100, path="tfidf_cosine_similarity_features.csv")
# add_lexical_features(df, batch_size=100, path="lexical_features.csv")
# add_imp_score_features(df, path="imp_score_features.csv")
