import pandas as pd

# This is a parse from the IBM Stance Corpus. This parse selects the topic text, 
# which is a genreally positive or negative stance towards the topic as arg1 and the
# claims.claimCorrectedText as arg2
# CON is read as attack relation and PRO is read as support

corpus_path = "/home/oenni/Dokumente/ArgumentationCorpora/IBM_Debater_(R)_CS_EACL-2017.v1/claim_stance_dataset_v1.csv"

df = pd.read_csv(corpus_path)

arg1 = list(df.topicText)
arg2 = list(df["claims.claimCorrectedText"])
relation = ["support" if rel == "PRO" else "attack" for rel in list(df["claims.stance"])]
topic = list(df.topicTarget)
ID = [f"IBM_stance_000{i}" if i < 10 else f"IBM_stance_00{i}" if i < 100 else f"IBM_stance_0{i}" if i < 1000 else f"IBM_stance_{i}" for i in range(len(relation))]

out_df = pd.DataFrame()
out_df["ID"] = ID
out_df["relation"] = relation
out_df["seg1_text"] = arg1
out_df["seg2_text"] = arg2
out_df["topic"] = topic

print(out_df.value_counts("relation"))
print(len(out_df.value_counts("topic")))

out_df.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/corpus/IBM_stance/IBMStance_parsed.tsv", sep="\t", index=False)