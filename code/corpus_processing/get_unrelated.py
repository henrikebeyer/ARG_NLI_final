import pandas as pd
import os

# Corpora that contain unrelated: Kialo, UKP, Web, NK
dfs = []
for corpus in ['JoKialo', 'UKP', 'Web', 'NK']:
    corpus_path = 'D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\corpus\\corpus_splits\\per_corpus'
    file_path = os.path.join(corpus_path, f"{corpus}_qwen_labeled.tsv")
    corpus_df = pd.read_csv(file_path, sep="\t").rename(columns={"seg2_text":"antecedent", "seg1_text":"consequent"})
    unrelated_df = corpus_df[corpus_df["relation"]=="neutral"]
    dfs.append(unrelated_df)
    
joined_df = pd.concat(dfs, axis=0)

URL_RE = r'\b(?:https?://|www\.)\S+\b'
joined_df["antecedent"] = (joined_df["antecedent"]
                                        .str.replace(URL_RE, "", regex=True)
                                        .str.replace("(","", regex=False)
                                        .str.replace("[","", regex=False)
                                        .str.replace(")","", regex=False)
                                        .str.replace("]","", regex=False)
                                        .str.replace(r"\s{2,}", " ", regex=True)
                                        .str.replace(r"\s+([.,;:!?])", r"\1", regex=True)
                                        .str.strip())

joined_df["consequent"] = (joined_df["consequent"]
                                        .str.replace(URL_RE, "", regex=True)
                                        .str.replace("(","", regex=False)
                                        .str.replace("[","", regex=False)
                                        .str.replace(")","", regex=False)
                                        .str.replace("]","", regex=False)
                                        .str.replace(r"\s{2,}", " ", regex=True)        # collapse spaces
                                        .str.replace(r"\s+([.,;:!?])", r"\1", regex=True)
                                        .str.strip())

joined_df = joined_df[~joined_df["antecedent"].str.contains('\n')]
# mask_ukp = joined_df["Corpus_ID"] == "UKP"

# joined_df.loc[mask_ukp, "antecedent"] = (
#     joined_df.loc[mask_ukp, "antecedent"]
#     .astype(str)
#     .str.split(r"[\r\t]", n=1)
#     .str[0]
#     .str.strip()
# )

print(joined_df.value_counts('Corpus_ID'))
# joined_df.to_csv('D:\\OneDrive - University of Dundee\\PhD\\ARG-NLI_project\\corpus\\unrelated_only.csv', index=False)