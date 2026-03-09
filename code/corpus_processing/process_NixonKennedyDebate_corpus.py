import pandas as pd

corpus_path = "/home/oenni/Dokumente/ArgumentationCorpora/Political-Argumentation-version-1.0/full_dataset.tsv"

df = pd.read_csv(corpus_path, sep="\t")

ID = [f"NK_deb_000{i}" if int(i) < 10 else f"NK_deb_00{i}" if int(i) < 100 else f"NK_deb_0{i}" if int(i) < 1000 else f"NK_deb_{i}" for i in df.pair_id]

df["ID"] = ID

df.rename(columns={"argument1":"seg1_text", "argument2":"seg2_text","source_arg_1":"source_seg1", "source_arg_2":"source_seg2"}, inplace=True)
df2 = df.reindex(["ID", "relation", "seg1_text", "seg2_text", "topic", "pair_id", "source_seg1", "source_seg2"], axis="columns", copy=False)

#print(df.value_counts("topic"))
print(df2)

df2.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/corpus/Nixon-Kennedy_parsed/Nixon-Kennedy_parsed.tsv", sep="\t", index=False)