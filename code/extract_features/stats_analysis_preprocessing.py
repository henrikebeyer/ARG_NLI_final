# questions about the data:
# 1. What is the distribution of NLI labels (entailment, neutral, contradiction)
# across different datasets
# 2. How is the distribution of ARG labels (support, attack) across different
# datasets
# 3. How do the NLI and ARG labels correlate with each other within each dataset
# 4. How are the categorical and continous features distributed across the ARG
# and NLI labels
# 5. are there correlations between features and labels
# 6. are there correlations between features themselves

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

corpus_path = "C:\\Users\\henri\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\automatic\\new_features\\corpus_nli_preds_with_new_features_threshold"

def load_corpus(corpus_path, config):
    corpus_df = pd.read_csv(f"{corpus_path}_{config}_0.999.csv")
    # print(corpus_df.shape)
    # print(corpus_df.head())
    return corpus_df

# plot distibution of NLI labels across datasets
def plot_nli_label_distribution(corpus_df):
    dataset_groups = corpus_df.groupby("Corpus_ID")
    nli_label_counts = dataset_groups["nli_preds"].value_counts(normalize=True).unstack().fillna(0)
    
    nli_label_counts.plot(kind="bar", stacked=True)
    plt.title("Distribution of NLI Labels Across Datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Proportion")
    plt.legend(title="NLI Label")
    plt.tight_layout()
    plt.show()

# plot distribution of ARG labels across datasets
def plot_arg_label_distribution(corpus_df):
    dataset_groups = corpus_df.groupby("Corpus_ID")
    arg_label_counts = dataset_groups["relation"].value_counts(normalize=True).unstack().fillna(0)
    
    arg_label_counts.plot(kind="bar", stacked=True)
    plt.title("Distribution of ARG Labels Across Datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Proportion")
    plt.legend(title="ARG Label")
    plt.tight_layout()
    plt.show()

# plot correlation between NLI and ARG labels within each dataset as heatmap
def heatmap_nli_arg_correlation(corpus_df):
    import seaborn as sns
    dataset_groups = corpus_df.groupby("Corpus_ID")
    
    for dataset, group in dataset_groups:
        contingency_table = pd.crosstab(group["relation"], group["nli_preds"], normalize="index")
        plt.figure(figsize=(6, 4))
        sns.heatmap(contingency_table, annot=True, cmap="Blues", cbar=False)
        plt.title(f"NLI vs ARG Correlation in {dataset}")
        plt.xlabel("ARG Label")
        plt.ylabel("NLI Label")
        plt.tight_layout()
        plt.show()
        
    # overall corpus plot
    contingency_table = pd.crosstab(corpus_df["relation"], corpus_df["nli_preds"], normalize="index")
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(contingency_table, annot=True, cmap="Blues", cbar=False)
    plt.title("Overall NLI vs ARG Correlation")
    plt.xlabel("ARG Label")
    plt.ylabel("NLI Label")
    plt.tight_layout()
    plt.show()

# plot distribution of features across NLI and ARG labels
def plot_categorical_feature_distributions(corpus_df):
    import seaborn as sns
    features = [
    "seg1_has_deduction", "seg1_has_refutation", "seg1_has_condition", "seg1_has_explanation", "seg1_has_adverb_of_emphasis", "seg1_has_justification", 
    "seg2_has_deduction", "seg2_has_refutation", "seg2_has_condition", "seg2_has_explanation", "seg2_has_adverb_of_emphasis", "seg2_has_justification", 
    "seg1_has_modal_verbs", "seg2_has_modal_verbs",
    "seg1_senti", "seg2_senti",
    ]
    for feature in features:
        # 
        # normalise data for better visualisation
        grouped_data = corpus_df.groupby("nli_preds")[feature].value_counts(normalize=True).unstack().fillna(0)
        #plot normalised data
        plt.figure(figsize=(8, 6))
        sns.barplot(data=grouped_data.reset_index().melt(id_vars="nli_preds"), x=feature, y="value", hue="nli_preds")
        plt.title(f"Normalised Distribution of {feature} across NLI Labels")
        plt.xlabel(feature)
        plt.ylabel("Proportion")
        plt.legend(title="NLI Label")
        plt.tight_layout()
        plt.show()
        
        grouped_data = corpus_df.groupby("relation")[feature].value_counts(normalize=True).unstack().fillna(0)
        print(f"Normalised distribution for {feature}:\n{grouped_data}\n")
        plt.figure(figsize=(8, 6))
        sns.barplot(data=grouped_data.reset_index().melt(id_vars="relation"), x=feature, y="value", hue="relation")
        plt.title(f"Normalised Distribution of {feature} across ARG Labels")
        plt.xlabel(feature)
        plt.ylabel("Proportion")
        plt.legend(title="ARG Label")
        plt.tight_layout()
        plt.show()


def determine_data_distribution(corpus_df, feature):
    from distfit import distfit
    import numpy as np
    data = corpus_df[feature].dropna().values
    
    fitter = distfit(method="parametric")
    
    fitter.fit_transform(data, verbose=False)
    
    print(f"Best fit distribution for {feature}: {fitter.model["name"]}")
    print(f"Parameters: {fitter.model["params"]}")
    
    import matplotlib.pyplot as plt
    
    # create a couple of plots to see how well the data fits the distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # left plot: your data with the fitted curve overlaid
    fitter.plot(chart="PDF", ax=ax1)
    ax1.set_title(f"Fitted Distribution for {feature}")
    
    # right plot: cumulative distribution function (CDF)
    fitter.plot(chart="CDF", ax=ax2)
    ax2.set_title(f"CDF for {feature}")
    plt.tight_layout()
    plt.show()

categorical_features = [
    "seg1_has_deduction", "seg1_has_refutation", "seg1_has_condition", "seg1_has_explanation", "seg1_has_adverbs_of_emphasis", "seg1_has_justification", 
    "seg2_has_deduction", "seg2_has_refutation", "seg2_has_condition", "seg2_has_explanation", "seg2_has_adverbs_of_emphasis", "seg2_has_justification", 
    "seg1_has_modal_verbs", "seg2_has_modal_verbs",
    "seg1_senti", "seg2_senti"]
continuous_features = [
    "cosine_similarity",
    "seg1_impScore", "seg2_impScore", "prag_distance", 
    "seg1_perplexity", "seg2_perplexity", #"seg1_given_seg2_perplexity", 
    "seg2_given_seg1_perplexity",
    "seg1_sentence_length_mean", #"seg1_sentence_length_median", 
    "seg1_n_tokens", #"seg1_n_sentences",
    "seg1_dependency_distance_mean",
    "seg1_prop_adjacent_dependency_relation_mean",
    "seg2_sentence_length_mean", #"seg2_sentence_length_median", 
    "seg2_n_tokens", #"seg2_n_sentences",
    "seg2_dependency_distance_mean",
    "seg2_prop_adjacent_dependency_relation_mean",
    "seg1_avg_subclauses", "seg1_avg_tree_depth", 
    "seg2_avg_subclauses", "seg2_avg_tree_depth",
    "tfidf_cosine_similarity"
    ]

def inspect_continuous_feature_distribution(corpus_df, feature):
    import seaborn as sns
    # violin plot of feature across NLI labels
    plt.figure(figsize=(8, 6))
    if feature in ["seg1_perplexity", "seg2_perplexity", "seg1_given_seg2_perplexity", "seg2_given_seg1_perplexity",]:
        sns.violinplot(data=corpus_df, x="nli_preds", y=feature, inner="quartile", log_scale=True)
    else:
        sns.violinplot(data=corpus_df, x="nli_preds", y=feature, inner="quartile", log_scale=False)
    plt.title(f"Distribution of {feature} across NLI Labels")
    plt.xlabel("NLI Label")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    if feature in ["seg1_perplexity", "seg2_perplexity", "seg1_given_seg2_perplexity", "seg2_given_seg1_perplexity",]:
        sns.violinplot(data=corpus_df, x="relation", y=feature, inner="quartile", log_scale=True)
    else:
        sns.violinplot(data=corpus_df, x="relation", y=feature, inner="quartile", log_scale=False)
    plt.title(f"Distribution of {feature} across ARG Labels")
    plt.xlabel("ARG Label")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()


corpus_df = load_corpus(corpus_path, "soft")
# for feature in continuous_features:
#     print(f"--- Determining distribution for feature: {feature} ---")
#     # determine_data_distribution(corpus_df, feature)
#     inspect_continuous_feature_distribution(corpus_df, feature)

# plot_categorical_feature_distributions(corpus_df)
# distribution_inspection_continuous_variabes(corpus_df)
# plot_continuous_feature_distributions(corpus_df)
# plot_nli_label_distribution(corpus_df)
# plot_arg_label_distribution(corpus_df)
# heatmap_nli_arg_correlation(corpus_df)

def compute_feature_correlations(corpus_df):
    import seaborn as sns
    # set categorical_features as numerical for correlation computation
    corpus_df["seg1_senti"] = corpus_df["seg1_senti"].map({"POSITIVE": 1, "NEGATIVE": 0})
    corpus_df["seg2_senti"] = corpus_df["seg2_senti"].map({"POSITIVE": 1, "NEGATIVE": 0})
    for feature in [
        "seg1_has_deduction", "seg1_has_refutation", "seg1_has_condition", "seg1_has_explanation", "seg1_has_adverb_of_emphasis", "seg1_has_justification", 
        "seg2_has_deduction", "seg2_has_refutation", "seg2_has_condition", "seg2_has_explanation", "seg2_has_adverb_of_emphasis", "seg2_has_justification", 
        "seg1_has_modal_verbs", "seg2_has_modal_verbs"]:
        corpus_df[feature] = corpus_df[feature].astype(int)
    correlation_matrix = corpus_df[continuous_features + categorical_features].corr(method="pearson")
    sorted_correlation = correlation_matrix.abs().unstack().sort_values(ascending=False)
    
    print("Features with high correlation (>0.7):")
    high_corr = sorted_correlation[(sorted_correlation < 1.0) & (sorted_correlation > 0.7)]
    print(high_corr.drop_duplicates())
    
    print("\nFeatures with moderate correlation (0.5-0.7):")
    moderate_corr = sorted_correlation[(sorted_correlation <= 0.7) & (sorted_correlation >= 0.5)]
    print(moderate_corr.drop_duplicates())  

    print("\nFeatures with low correlation (0.5-0.3):")
    low_corr = sorted_correlation[(sorted_correlation <= 0.5) & (sorted_correlation >= 0.3)]
    print(low_corr.drop_duplicates())
    
    plt.figure(figsize=(90, 50))
    sns.set_theme(font_scale=3)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "Correlation Coefficient"})
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

def compute_vif(corpus_df, featrures=continuous_features):
    # compute Variance Inflation Factor (VIF) for each feature
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = corpus_df[continuous_features].dropna()
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_sorted = vif_data.sort_values(by="VIF", ascending=False)
    print("Variance Inflation Factor (VIF) for each feature:")
    print(vif_sorted)
    return vif_sorted

def correlations_with_target_labels(corpus_df):
    # compute correlation of each feature with NLI and ARG labels
    from sklearn.preprocessing import LabelEncoder
    le_nli = LabelEncoder()
    le_arg = LabelEncoder()
    le_ent = LabelEncoder()
    le_cont = LabelEncoder()
    le_supp = LabelEncoder()
    le_att = LabelEncoder()
    le_arg_nli = LabelEncoder()
    corpus_df["arg_nli_support"] = [1 if (nli.lower() == "entailment" and arg == "support") else 0 for nli, arg in zip(corpus_df["nli_preds"], corpus_df["relation"])]
    corpus_df["arg_nli_attack"] = [1 if (nli.lower() == "contradiction" and arg == "attack") else 0 for nli, arg in zip(corpus_df["nli_preds"], corpus_df["relation"])]
    corpus_df["is_entailment"] = [1 if nli.lower() == "entailment" else 0 for nli in corpus_df["nli_preds"]]
    corpus_df["is_contradiction"] = [1 if nli.lower() == "contradiction" else 0 for nli in corpus_df["nli_preds"]]
    corpus_df["is_neutral"] = [1 if nli.lower() not in ["entailment", "contradiction"] else 0 for nli in corpus_df["nli_preds"]]
    corpus_df["is_support"] = [1 if arg == "support" else 0 for arg in corpus_df["relation"]]
    corpus_df["is_attack"] = [1 if arg == "attack" else 0 for arg in corpus_df["relation"]]
    # corpus_df["arg_nli_support_overlap_encoded"] = le_arg_nli.fit_transform(corpus_df["arg_nli_support"])
    # corpus_df["arg_nli_attack_overlap_encoded"] = le_arg_nli.fit_transform(corpus_df["arg_nli_attack"])
    # corpus_df["entailment_encoded"] = le_ent.fit_transform(corpus_df["is_entailment"])
    # corpus_df["contradiction_encoded"] = le_cont.fit_transform(corpus_df["is_contradiction"])
    # # corpus_df["arg_encoded"] = le_arg.fit_transform(corpus_df["relation"])
    # corpus_df["is_support_encoded"] = le_supp.fit_transform(corpus_df["is_support"])
    # corpus_df["is_attack_encoded"] = le_att.fit_transform(corpus_df["is_attack"])
    
    corpus_df["seg1_senti"] = corpus_df["seg1_senti"].map({"POSITIVE": 1, "NEGATIVE": 0})
    corpus_df["seg2_senti"] = corpus_df["seg2_senti"].map({"POSITIVE": 1, "NEGATIVE": 0})
    for feature in [
        "seg1_has_deduction", "seg1_has_refutation", "seg1_has_condition", "seg1_has_explanation", "seg1_has_adverbs_of_emphasis", "seg1_has_justification", 
        "seg2_has_deduction", "seg2_has_refutation", "seg2_has_condition", "seg2_has_explanation", "seg2_has_adverbs_of_emphasis", "seg2_has_justification", 
        "seg1_has_modal_verbs", "seg2_has_modal_verbs"]:
        corpus_df[feature] = corpus_df[feature].astype(int)
            
    feature_target_corr = {}
    for feature in continuous_features + categorical_features:
        ent_corr = corpus_df[feature].corr(corpus_df["is_entailment"])
        cont_corr = corpus_df[feature].corr(corpus_df["is_contradiction"])
        neu_corr = corpus_df[feature].corr(corpus_df["is_neutral"])
        supp_corr = corpus_df[feature].corr(corpus_df["is_support"])
        att_corr = corpus_df[feature].corr(corpus_df["is_attack"])
        arg_nli_attack_corr = corpus_df[feature].corr(corpus_df["arg_nli_attack"])
        arg_nli_support_corr = corpus_df[feature].corr(corpus_df["arg_nli_support"])
        
        # arg_corr = corpus_df[feature].corr(corpus_df["arg_encoded"])
        # arg_nli_corr = corpus_df[feature].corr(corpus_df["arg_nli_overlap_encoded"])
        feature_target_corr[feature] = {"entailment_Correlation": ent_corr, 
                                        "contradiction_Correlation": cont_corr, 
                                        "neutral_Correlation": neu_corr,
                                        "support_Correlation": supp_corr,
                                        "attack_Correlation": att_corr,
                                        "support_Overlap_Correlation": arg_nli_support_corr,
                                        "attack_Overlap_Correlation": arg_nli_attack_corr,}
    
    feature_target_corr_df = pd.DataFrame.from_dict(feature_target_corr, orient="index")
    for target in ["entailment", "contradiction", "neutral", "support", "attack"]:
        print("Top correlations with ", target)
        print(feature_target_corr_df.sort_values(by=[f"{target}_Correlation"], ascending=False)[f"{target}_Correlation"])
        if target in ["support", "attack"]:
            print("Top correlations with ARG-NLI Overlap ", target)
            print(feature_target_corr_df.sort_values(by=[f"{target}_Overlap_Correlation"], ascending=False)[f"{target}_Overlap_Correlation"])
    # print(feature_target_corr_df.sort_values(by=["ARG_Correlation"], ascending=False)["ARG_Correlation"])
    # print(feature_target_corr_df.sort_values(by=["ARG_NLI_Overlap_Correlation"], ascending=False)["ARG_NLI_Overlap_Correlation"])
    return feature_target_corr_df

# correlation_matrix = compute_feature_correlations(corpus_df)
#print(correlation_matrix)

# x = correlations_with_target_labels(corpus_df)

#compute_vif(corpus_df)

def z_score(series):
    # z_scored = [((val-series.mean()) / series.std()) for val in series]
    return (series - series.mean())/series.std()

def log_scale(series):
    import numpy as np
    return np.log(series)

def scale_data_add_target_cols(corpus_df, out_path):
    print(corpus_df.columns)
    predictor_df = pd.DataFrame()
    predictor_df["ID"] = corpus_df["ID"]
    predictor_df["CorpusID"] = corpus_df["Corpus_ID"]
    predictor_df["nli"] = corpus_df["nli_preds"]
    predictor_df["arg"] = corpus_df["relation"]
    predictor_df["seg1_text"] = corpus_df["seg1_text"]
    predictor_df["seg2_text"] = corpus_df["seg2_text"]
    predictor_df["arg_nli_support"] = [1 if (nli.lower() == "entailment" and arg == "support") else 0 for nli, arg in zip(corpus_df["nli_preds"], corpus_df["relation"])]
    predictor_df["arg_nli_attack"] = [1 if (nli.lower() == "contradiction" and arg == "attack") else 0 for nli, arg in zip(corpus_df["nli_preds"], corpus_df["relation"])]
    predictor_df["is_entailment"] = [1 if nli.lower() == "entailment" else 0 for nli in corpus_df["nli_preds"]]
    predictor_df["is_contradiction"] = [1 if nli.lower() == "contradiction" else 0 for nli in corpus_df["nli_preds"]]
    predictor_df["is_neutral"] = [1 if nli.lower() not in ["entailment", "contradiction"] else 0 for nli in corpus_df["nli_preds"]]
    predictor_df["is_support"] = [1 if arg == "support" else 0 for arg in corpus_df["relation"]]
    predictor_df["is_attack"] = [1 if arg == "attack" else 0 for arg in corpus_df["relation"]]
    
    # z-score scale continuous variables
    for feature in continuous_features:
        if feature not in ["seg1_perplexity", "seg2_perplexity", "seg2_given_seg1_perplexity"]:
            predictor_df[f"{feature}"] = z_score(corpus_df[feature])
        else:
            predictor_df[feature] = z_score(log_scale(corpus_df[feature]))
        # print(predictor_df[feature].mean(), predictor_df[feature].std())
        
    for feature in categorical_features:
        if feature == "seg1_has_adverbs_of_emphasis":
            predictor_df["seg1_has_adverb_of_emphasis"] = corpus_df[feature]
        elif feature == "seg2_has_adverbs_of_emphasis":
            predictor_df["seg2_has_adverb_of_emphasis"] = corpus_df[feature]
        else:
            predictor_df[feature] = corpus_df[feature]  
    predictor_df.to_csv(out_path, index=False)
        
    
        
config = "soft"        
out_path = f"D:\\OneDrive - University of Dundee\\PhD\ARG-NLI_project\\feature_analysis\\automatic\\new_features\\corpus_threshold_{config}_0.999_standardised_and_log_scaled.csv"      
scale_data_add_target_cols(corpus_df, out_path)