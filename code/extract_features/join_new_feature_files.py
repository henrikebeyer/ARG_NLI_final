import pandas as pd
import os

config = ["soft", "hard"]

corpus_path = "C:\\Users\\henri\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\corpus\\corpus_nli_preds_threshold"
feature_path = f"C:\\Users\\henri\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\automatic\\new_features\\single_feature_categories"

features = [
    "cosine_similarity",
    "seg1_impScore", "seg2_impScore", "prag_distance", 
    "seg1_has_deduction", "seg1_has_refutation", "seg1_has_condition", "seg1_has_explanation", "seg1_has_adverbs_of_emphasis", "seg1_has_justification", 
    "seg2_has_deduction", "seg2_has_refutation", "seg2_has_condition", "seg2_has_explanation", "seg2_has_adverbs_of_emphasis", "seg2_has_justification", 
    "seg1_has_modal_verbs", "seg2_has_modal_verbs",
    "seg1_perplexity", "seg2_perplexity", "seg1_given_seg2_perplexity", "seg2_given_seg1_perplexity",
    "seg1_senti", "seg2_senti",
    "seg1_sentence_length_mean", "seg1_sentence_length_median", "seg1_sentence_length_std", 
    "seg1_n_tokens", "seg1_n_sentences",
    "seg1_dependency_distance_mean", "seg1_dependency_distance_std",
    "seg1_prop_adjacent_dependency_relation_mean", "seg1_prop_adjacent_dependency_relation_std",
    "seg2_sentence_length_mean", "seg2_sentence_length_median",
    "seg2_sentence_length_std", "seg2_n_tokens", "seg2_n_sentences",
    "seg2_dependency_distance_mean", "seg2_dependency_distance_std", 
    "seg2_prop_adjacent_dependency_relation_mean", "seg2_prop_adjacent_dependency_relation_std",
    "seg1_avg_subclauses", "seg1_avg_tree_depth", 
    "seg2_avg_subclauses", "seg2_avg_tree_depth",
    "tfidf_cosine_similarity"
]

def read_in_feature_files(feature_file_path, config):
    feature_dfs = []
    id_set = False
    for file in os.listdir(feature_file_path):
        if config in file and file.endswith(".csv"):
            df = pd.read_csv(os.path.join(feature_file_path, file))
            print(file, df.shape)
            if file in ["soft_syntactic_features.csv", "soft_perplexity_features.csv", "soft_lexical_features.csv"]:
                df = df[-164205:].copy().reset_index(drop=True)
                # print(df.tail())
                #print(df.shape)
                
            if not id_set and "ID" in df.columns:
                id_set = True
            elif id_set and "ID" in df.columns:
                df = df.drop(columns=["ID"])
            feature_dfs.append(df)
    full_df = pd.concat(feature_dfs, axis=1)
        
    feature_col_df = full_df[["ID"] + features]
    print(feature_col_df.head())
    # print(feature_col_df.tail())
    # for feature in feature_columns:
    #     print(f"Unique values for feature: {feature}")
    #     print(feature_col_df[feature].unique())
    
    return feature_col_df
        
feature_columns = read_in_feature_files(feature_path, "hard")

def join_samples_with_features(corpus_path, feature_columns, config):
    corpus_file = f"{corpus_path}_{config}_0.999.tsv"
    corpus_df = pd.read_csv(corpus_file, sep="\t")
    print(f"Corpus shape: {corpus_df.shape}")
    
    merged_df = pd.concat([corpus_df, feature_columns], keys="ID", axis=1)
    print(f"Merged shape: {merged_df.shape}")
    print(merged_df.head())
    print(merged_df.tail())
    
    output_file = os.path.join("C:\\Users\\henri\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\automatic\\new_features", f"corpus_nli_preds_with_new_features_threshold_{config}_0.999.tsv")
    merged_df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved merged file to: {output_file}")
    
join_samples_with_features(corpus_path, feature_columns, "hard")