import pandas as pd
import re

def clean_base_corpus():
    path_hard = "D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\corpus\\corpus_nli_preds_threshold_hard_0.999.tsv"
    path_soft = "D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\corpus\\corpus_nli_preds_threshold_soft_0.999.tsv"

    corpus_hard = pd.read_csv(path_hard, sep= "\t")
    corpus_hard.rename(columns={"seg1_text":"consequent",
                        "seg2_text":"antecedent"}, inplace=True)
    corpus_soft = pd.read_csv(path_soft, sep = "\t")
    corpus_soft.rename(columns={"seg1_text":"consequent",
                        "seg2_text":"antecedent"}, inplace=True)

    # remove Webis corpus because it is shit!
    corpus_hard_no_webis = corpus_hard[corpus_hard["Corpus_ID"]!="Webis-Debate"]
    corpus_soft_no_webis = corpus_soft[corpus_soft["Corpus_ID"]!="Webis-Debate"]

    # filter out any antecedents that are URLs
    URL_RE = r'\b(?:https?://|www\.)\S+\b'
    corpus_hard_no_webis["antecedent"] = (corpus_hard_no_webis["antecedent"]
                                        .str.replace(URL_RE, "", regex=True)
                                        .str.replace("(","", regex=False)
                                        .str.replace("[","", regex=False)
                                        .str.replace(")","", regex=False)
                                        .str.replace("]","", regex=False)
                                        .str.replace(r"\s{2,}", " ", regex=True)        # collapse spaces
                                        .str.replace(r"\s+([.,;:!?])", r"\1", regex=True)
                                        .str.strip())
    corpus_hard_no_webis["consequent"] = (corpus_hard_no_webis["consequent"]
                                        .str.replace(URL_RE, "", regex=True)
                                        .str.replace("(","", regex=False)
                                        .str.replace("[","", regex=False)
                                        .str.replace(")","", regex=False)
                                        .str.replace("]","", regex=False)
                                        .str.replace(r"\s{2,}", " ", regex=True)        # collapse spaces
                                        .str.replace(r"\s+([.,;:!?])", r"\1", regex=True)
                                        .str.strip())

    mask_ukp = corpus_hard_no_webis["Corpus_ID"] == "UKP"

    corpus_hard_no_webis.loc[mask_ukp, "antecedent"] = (
        corpus_hard_no_webis.loc[mask_ukp, "antecedent"]
        .astype(str)
        .str.split(r"[\r\n]", n=1)
        .str[0]
        .str.strip()
    )

    corpus_hard_no_webis_len = corpus_hard_no_webis.loc[corpus_hard_no_webis["antecedent"].str.split().str.len()>=2]
    corpus_hard_no_webis_len = corpus_hard_no_webis_len.loc[corpus_hard_no_webis_len["consequent"].str.split().str.len()>=2]

    print(corpus_hard_no_webis_len.value_counts("Corpus_ID"))

    counts = (
        corpus_hard_no_webis_len
        .groupby("Corpus_ID")["relation"]
        .value_counts()
        .rename("count")
        .reset_index()
    )

    counts["percentage"] = (
        counts
        .groupby("Corpus_ID")["count"]
        .transform(lambda x: x / x.sum())
    )
    print(counts)

    corpus_soft_no_webis["antecedent"] = (corpus_soft_no_webis["antecedent"]
                                        .str.replace(URL_RE, "", regex=True)
                                        .str.replace("(","", regex=False)
                                        .str.replace("[","", regex=False)
                                        .str.replace(")","", regex=False)
                                        .str.replace("]","", regex=False)
                                        .str.replace(r"\s{2,}", " ", regex=True)
                                        .str.replace(r"\s+([.,;:!?])", r"\1", regex=True)
                                        .str.strip())
    corpus_soft_no_webis["consequent"] = (corpus_soft_no_webis["consequent"]
                                        .str.replace(URL_RE, "", regex=True)
                                        .str.replace("(","", regex=False)
                                        .str.replace("[","", regex=False)
                                        .str.replace(")","", regex=False)
                                        .str.replace("]","", regex=False)
                                        .str.replace(r"\s{2,}", " ", regex=True)        # collapse spaces
                                        .str.replace(r"\s+([.,;:!?])", r"\1", regex=True)
                                        .str.strip())

    mask_ukp = corpus_soft_no_webis["Corpus_ID"] == "UKP"

    corpus_soft_no_webis.loc[mask_ukp, "antecedent"] = (
        corpus_soft_no_webis.loc[mask_ukp, "antecedent"]
        .astype(str)
        .str.split(r"[\r\n]", n=1)
        .str[0]
        .str.strip()
    )

    corpus_soft_no_webis_len = corpus_soft_no_webis.loc[corpus_soft_no_webis["antecedent"].str.split().str.len()>=2]
    corpus_soft_no_webis_len = corpus_soft_no_webis_len.loc[corpus_soft_no_webis_len["consequent"].str.split().str.len()>=2]

    print(corpus_soft_no_webis_len.value_counts("Corpus_ID"))
    counts = (
        corpus_soft_no_webis_len
        .groupby("Corpus_ID")["relation"]
        .value_counts()
        .rename("count")
        .reset_index()
    )

    counts["percentage"] = (
        counts
        .groupby("Corpus_ID")["count"]
        .transform(lambda x: x / x.sum())
    )
    print(counts)

    # fix duplicates
    # mark duplicate rows except the first occurrence
    dup_rows_soft = corpus_soft_no_webis_len["ID"].duplicated(keep="first")

    # generate duplicate counters per ID
    corpus_soft_no_webis_len.loc[dup_rows_soft, "dup_index"] = (
        corpus_soft_no_webis_len[dup_rows_soft]
        .groupby("ID")
        .cumcount() + 1
    )

    # create new IDs only for duplicates
    corpus_soft_no_webis_len.loc[dup_rows_soft, "ID"] = (
        corpus_soft_no_webis_len.loc[dup_rows_soft, "ID"]
        + "_dup"
        + corpus_soft_no_webis_len.loc[dup_rows_soft, "dup_index"].astype(int).astype(str)
    )

    # cleanup
    corpus_soft_no_webis_len = corpus_soft_no_webis_len.drop(columns="dup_index")

    # mark duplicate rows except the first occurrence
    dup_rows_hard = corpus_hard_no_webis_len["ID"].duplicated(keep="first")

    # generate duplicate counters per ID
    corpus_hard_no_webis_len.loc[dup_rows_hard, "dup_index"] = (
        corpus_hard_no_webis_len[dup_rows_hard]
        .groupby("ID")
        .cumcount() + 1
    )

    # create new IDs only for duplicates
    corpus_hard_no_webis_len.loc[dup_rows_hard, "ID"] = (
        corpus_hard_no_webis_len.loc[dup_rows_hard, "ID"]
        + "_dup"
        + corpus_hard_no_webis_len.loc[dup_rows_hard, "dup_index"].astype(int).astype(str)
    )

    # cleanup
    corpus_hard_no_webis_len = corpus_hard_no_webis_len.drop(columns="dup_index")

    print(corpus_soft_no_webis_len.loc[corpus_soft_no_webis_len["ID"].isin(["AAAI2018_09_11", "AAAI2018_09_11_dup1", "AAAI2018_09_15", "AAAI2018_09_15_dup1"])])
    print(corpus_hard_no_webis_len.loc[corpus_hard_no_webis_len["ID"].isin(["AAAI2018_09_11", "AAAI2018_09_11_dup1", "AAAI2018_09_15", "AAAI2018_09_15_dup1"])])

    out_hard = "D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\corpus\\corpus_nli_preds_threshold_hard_0.999_noWebis_cleaned.csv"
    out_soft = "D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\corpus\\corpus_nli_preds_threshold_soft_0.999_noWebis_cleaned.csv"

    # TODO: Make the same fixes on the stuff with features; hopefully index matching helps ...

    corpus_hard_no_webis_len.to_csv(out_hard, index=False)
    corpus_soft_no_webis_len.to_csv(out_soft, index=False)
    
def clean_feature_corpus():
    base_hard = "D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\corpus\\corpus_nli_preds_threshold_hard_0.999_noWebis_cleaned.csv"
    base_soft = "D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\corpus\\corpus_nli_preds_threshold_soft_0.999_noWebis_cleaned.csv"
    
    feature_hard = "D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\automatic\\new_features\\corpus_threshold_hard_0.999_standardised_log_scaled.csv"
    feature_soft = "D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\automatic\\new_features\\corpus_threshold_soft_0.999_standardised_and_log_scaled.csv"
    
    base_hard_df = pd.read_csv(base_hard)
    base_soft_df = pd.read_csv(base_soft)
    
    print(base_hard_df.columns)
    feature_hard_df = pd.read_csv(feature_hard).rename(columns={"seg2_text":"antecedent", "seg1_text":"consequent"})
    feature_soft_df = pd.read_csv(feature_soft).rename(columns={"seg2_text":"antecedent", "seg1_text":"consequent"})

    # reindex duplicates in feature_corpora:
    dup_rows_hard = feature_hard_df["ID"].duplicated(keep="first")
    feature_hard_df.loc[dup_rows_hard, "dup_index"] = (
        feature_hard_df[dup_rows_hard]
        .groupby("ID")
        .cumcount() + 1
    )

    # create new IDs only for duplicates
    feature_hard_df.loc[dup_rows_hard, "ID"] = (
        feature_hard_df.loc[dup_rows_hard, "ID"]
        + "_dup"
        + feature_hard_df.loc[dup_rows_hard, "dup_index"].astype(int).astype(str)
    )

    # cleanup
    feature_hard_df = feature_hard_df.drop(columns="dup_index")
    
    
    dup_rows_soft = feature_soft_df["ID"].duplicated(keep="first")
    feature_soft_df.loc[dup_rows_soft, "dup_index"] = (
        feature_soft_df[dup_rows_soft]
        .groupby("ID")
        .cumcount() + 1
    )

    # create new IDs only for duplicates
    feature_soft_df.loc[dup_rows_soft, "ID"] = (
        feature_soft_df.loc[dup_rows_soft, "ID"]
        + "_dup"
        + feature_soft_df.loc[dup_rows_soft, "dup_index"].astype(int).astype(str)
    )

    # cleanup
    feature_soft_df = feature_soft_df.drop(columns="dup_index")
    
    feature_cleaned_hard = feature_hard_df.merge(base_hard_df[["ID", "antecedent"]], "inner", on="ID", suffixes=["_old", ""])
    feature_cleaned_soft = feature_soft_df.merge(base_soft_df[["ID", "antecedent"]], "inner", on="ID", suffixes=["_old", ""])
    
    
    out_hard = "D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\automatic\\new_features\\corpus_threshold_hard_0.999_standardised_log_scaled_cleaned.csv"
    out_soft = "D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\automatic\\new_features\\corpus_threshold_soft_0.999_standardised_and_log_scaled_cleaned.csv"
        
    feature_cleaned_hard.to_csv(out_hard, index=False)
    feature_cleaned_soft.to_csv(out_soft, index=False)
    
# clean_feature_corpus()

out_hard = "D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\corpus\\corpus_nli_preds_threshold_hard_0.999_noWebis_cleaned.csv"
#"D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\automatic\\new_features\\corpus_threshold_hard_0.999_standardised_log_scaled_cleaned.csv"
out_soft = "D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\corpus\\corpus_nli_preds_threshold_soft_0.999_noWebis_cleaned.csv"
#"D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\feature_analysis\\automatic\\new_features\\corpus_threshold_soft_0.999_standardised_and_log_scaled_cleaned.csv"

df_soft = pd.read_csv(out_soft)
df_hard = pd.read_csv(out_hard)

# Get percentages per row (index)
percentages = pd.crosstab(
    index=[df_soft['Corpus_ID'], df_soft['relation']], 
    columns=df_soft['nli_preds'],
    #normalize='index' # This ensures rows sum to 1 (100%)
)
# counts = df_soft.groupby(["Corpus_ID", "relation", "nli_preds"]).size()
# counts_df = counts.reset_index(name="count")
percentages.to_csv("D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\corpus\\corpus_nli_preds_threshold_soft_0.999_noWebis_cleaned_nli_counts.csv")


percentages = pd.crosstab(
    index=[df_hard['Corpus_ID'], df_hard['relation']], 
    columns=df_hard['nli_preds'],
    #normalize='index' # This ensures rows sum to 1 (100%)
)
# counts = df_hard.groupby(["Corpus_ID", "relation", "nli_preds"]).size()
# counts_df = counts.reset_index(name="count")
percentages.to_csv("D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\corpus\\corpus_nli_preds_threshold_hard_0.999_noWebis_cleaned_nli_counts.csv")
# for corpus in df_soft["Corpus_ID"].unique():
#     for arg in ["support", "attack"]:
#         arg_count = df_soft[df_soft["Corpus_ID"] == corpus & df_soft["relation"] == arg].sum()
#         print(corpus, arg, arg_count)
# corpora = df_soft["CorpusID"].unique()
# for corpus in corpora:
#     sub_df = df_soft[df_soft["CorpusID"]==corpus]
#     print(corpus, sub_df.value_counts("arg"))