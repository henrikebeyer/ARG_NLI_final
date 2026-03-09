import pandas as pd

# This parse assumes the following columns in the dataset:
# ID, 

corpus_path = "/home/oenni/Dokumente/ArgumentationCorpora/WebContent/ACMToIT2017_dataset.xlsx"

df = pd.read_excel(corpus_path, names=["ID", "source", "topic", "seg1_text", "seg2_text", "relation"])
df = df.reindex(["ID", "relation", "seg1_text", "seg2_text", "topic", "source"], axis="columns", copy=False)

mapping_df = pd.DataFrame()
mapping_df["ID_old"] = [f"Web_000{i}" if i < 10 else f"Web_00{i}" if i < 100 else f"Web_0{i}" if i < 1000 else f"Web_{i}" for i, src in zip(df.ID, df.source)]
mapping_df["ID_new"] = [f"Web_{src}_000{i}" if i < 10 else f"Web_{src}_00{i}" if i < 100 else f"Web_{src}_0{i}" if i < 1000 else f"Web_{src}_{i}" for i, src in zip(df.ID, df.source)]

df["ID"] = [f"Web_{src}_000{i}" if i < 10 else f"Web_{src}_00{i}" if i < 100 else f"Web_{src}_0{i}" if i < 1000 else f"Web_{src}_{i}" for i, src in zip(df.ID, df.source)]
df["relation"] = ["support" if rel=="s" else "attack" if rel=="a" else "unrelated" for rel in df.relation]

print(df)
#print(df.value_counts("relation"))
#print(len(df.value_counts("topic")))
print(df.value_counts("source"))

mapping_df.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/corpus/parsed_corpora/WebContent_parsed/WebContent_IDmapping.tsv", sep="\t", index=False)
#df.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/corpus/WebContent_parsed/WebContent_parsed.tsv", sep="\t", index=False)