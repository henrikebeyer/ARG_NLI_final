import pandas as pd

debate_causal_path = "/home/oenni/Dokumente/ArgumentationCorpora/tacl_arg_rel-main/data/debate/debate_causal_pairs-bi.csv"
debate_normative_path = "/home/oenni/Dokumente/ArgumentationCorpora/tacl_arg_rel-main/data/debate/debate_normative_pairs-bi.csv"

kialo_causal_neu_path = "/home/oenni/Dokumente/ArgumentationCorpora/tacl_arg_rel-main/data/kialo/kialo_causal_pairs-neu.csv"
kialo_normative_neu_path = "/home/oenni/Dokumente/ArgumentationCorpora/tacl_arg_rel-main/data/kialo/kialo_normative_pairs-neu.csv"
kialo_causal_bi_path = "/home/oenni/Dokumente/ArgumentationCorpora/tacl_arg_rel-main/data/kialo/kialo_causal_pairs-bi.csv"
kialo_normative_bi_path = "/home/oenni/Dokumente/ArgumentationCorpora/tacl_arg_rel-main/data/kialo/kialo_normative_pairs-bi.csv"


debate_causal_df = pd.read_csv(debate_causal_path)
debate_normative_df = pd.read_csv(debate_normative_path)

kialo_causal_df = pd.concat([pd.read_csv(kialo_causal_neu_path),pd.read_csv(kialo_causal_bi_path)], axis=0)
kialo_normative_df = pd.concat([pd.read_csv(kialo_normative_neu_path),pd.read_csv(kialo_normative_bi_path)], axis=0)

ID_debate_causal = [f"JoDeb_causal_000{i}" if i < 10 
                    else f"JoDeb_causal_00{i}" if i < 100 
                    else f"JoDeb_causal_0{i}" if i < 1000 
                    else f"JoDeb_causal_{i}" for i in range(len(debate_causal_df["pairid"]))]
ID_debate_normative = [f"JoDeb_norm_000{i}" if i < 10 
                       else f"JoDeb_norm_00{i}" if i < 100 
                       else f"JoDeb_norm_0{i}" if i < 1000 
                       else f"JoDeb_norm_{i}" for i in range(len(debate_normative_df["pairid"]))]

ID_kialo_causal = [f"JoKialo_causal_000{i}" if i < 10 
                    else f"JoKialo_causal_00{i}" if i < 100 
                    else f"JoKialo_causal_0{i}" if i < 1000 
                    else f"JoKialo_causal_{i}" for i in range(len(kialo_causal_df["pairid"]))]
ID_kialo_normative = [f"JoKialo_norm_000{i}" if i < 10 
                    else f"JoKialo_norm_00{i}" if i < 100 
                    else f"JoKialo_norm_0{i}" if i < 1000 
                    else f"JoKialo_norm_{i}" for i in range(len(kialo_normative_df["pairid"]))]

relation_debate_causal = ["support" if str(r)=="1" else "attack" if str(r)=="-1" else "unrelated" for r in debate_causal_df["relation"]]
relation_debate_normative = ["support" if str(r)=="1" else "attack" if str(r)=="-1" else "unrelated" for r in debate_normative_df["relation"]]

relation_kialo_causal = ["support" if str(r)=="1" else "attack" if str(r)=="-1" else "unrelated" for r in kialo_causal_df["relation"]]
relation_kialo_normative = ["support" if str(r)=="1" else "attack" if str(r)=="-1" else "unrelated" for r in kialo_normative_df["relation"]]

debate_df = pd.DataFrame()
debate_df["ID"] = ID_debate_causal + ID_debate_normative
debate_df["relation"] = relation_debate_causal + relation_debate_normative
debate_df["seg1_id"] = list(debate_causal_df["propid_to"]) + list(debate_normative_df["propid_to"])
debate_df["seg2_id"] = list(debate_causal_df["propid_from"]) + list(debate_normative_df["propid_from"])
debate_df["seg1_text"] = list(debate_causal_df["text_to"]) + list(debate_normative_df["text_to"])
debate_df["seg2_text"] = list(debate_causal_df["text_from"]) + list(debate_normative_df["text_from"])

kialo_df = pd.DataFrame()
kialo_df["ID"] = ID_kialo_causal + ID_kialo_normative
kialo_df["relation"] = relation_kialo_causal + relation_kialo_normative
kialo_df["seg1_id"] = list(kialo_causal_df["propid_to"]) + list(kialo_normative_df["propid_to"])
kialo_df["seg2_id"] = list(kialo_causal_df["propid_from"]) + list(kialo_normative_df["propid_from"])
kialo_df["seg1_text"] = list(kialo_causal_df["text_to"]) + list(kialo_normative_df["text_to"])
kialo_df["seg2_text"] = list(kialo_causal_df["text_from"]) + list(kialo_normative_df["text_from"])

print(debate_df.value_counts("relation"))
print(kialo_df.value_counts("relation"))

debate_df.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/corpus/Jo_tacl_parsed/Jo_debate_parsed.tsv", sep="\t", index=False)
kialo_df.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/corpus/Jo_tacl_parsed/Jo_kialo_parsed.tsv", sep="\t", index=False)