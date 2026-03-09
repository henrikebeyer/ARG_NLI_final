import pandas as pd
import impscore 

corpus_list = [# "AAAI2018",
#                "AbstRCT",
#                "ArgEss",
#                "ComArg",
#                "FinArg",
#                "IBM",
#                "IJCAI2015",
#                "JoDeb",
#                "JoKialo",
#                "micro",
#                "NK",
#                "QT30",
#                "QT50",
#                "UKP",
#                "Web",
#                "Webis-Debate",
#                "US2016",
                "12AngryMen",
                "Debatepedia"]


corpus_path = "/home/oenni/Dokumente/NLI-Argumentation-project/feature_analysis/automatic/corpus_nli_preds_threshold"

config_list = ["hard", "soft"]
for config in config_list:
    path = f"{corpus_path}{config}_0.999_withFeatures.tsv"
    df = pd.read_csv(path, sep="\t")

    #print(df.columns)

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

    df.to_csv(path, sep="\t", index=False)

    #print(score1_np, imp_score2, prag_distance)

# sentence_pairs = [
#     ["I have to leave now. Talk to you later.", "I can't believe we've talked for so long."],
#     ["You must find a new place and move out by the end of this month.",
#      "Maybe exploring other housing options could benefit us both?"]
# ]

# s1_list = [pair[0] for pair in sentence_pairs]  # list of the first sentence in pairs
# s2_list = [pair[1] for pair in sentence_pairs]  # list of the second sentence in pairs

# # imp_score1 is the implicitness score list for s1 sentences,
# # imp_score2 is the implicitness score list for s2 sentences.
# # prag_distance is the pragmatic distance list, where prag_distance[i] is the pragmatic distance between s1[i] and s2[i].
# imp_score1, imp_score2, prag_distance = model.infer_pairs(s1_list, s2_list)

# print(imp_score1, imp_score2, prag_distance)

# the outputs: tensor([0.6709, 0.9273]) tensor([1.0984, 1.3642]) tensor([0.6660, 0.7115])