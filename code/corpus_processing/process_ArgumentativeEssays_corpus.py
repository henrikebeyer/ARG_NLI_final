import pandas as pd
import os

# The files parsed by this script are 

corpus_path = "/home/oenni/Dokumente/ArgumentationCorpora/ArgumentAnnotatedEssays-2.0/ArgumentAnnotatedEssays-2.0/brat-project-final/brat-project-final" #/essay01.ann"

for filename in os.listdir(corpus_path):
    if filename.endswith(".ann"):
        print(filename)
        with open(f"{corpus_path}/{filename}", "r") as f:
            data = f.read().split("\n")

        #print(data)

        stances = []
        relations = []
        arguments = []

        for line in data:
            l = line.split("\t")
            if l[0].startswith("A"):
                stances.append({"ID": l[0],
                                "label": l[1]})
            elif l[0].startswith("R"):
                rest = l[1].split(" ")
                relations.append({"ID":l[0],
                                "relation":rest[0],
                                "seg1_id":rest[1],
                                "seg2_id":rest[2]})
            elif l[0].startswith("T"):
                rest = l[1].split(" ")
                label = rest[0]
                start = rest[1]
                end = rest[2]
                text = l[2]
                arguments.append({"ID":l[0],
                            "relation":label,
                            "start":start,
                            "end":end,
                            "text":text}
                )

        relations_df = pd.DataFrame(relations)
        argument_df = pd.DataFrame(arguments)

        seg1 = [arg.split(":")[1] for arg in relations_df["seg1_id"]]
        seg2 = [arg.split(":")[1] for arg in relations_df["seg2_id"]]

        seg1_text = []
        seg2_text = []

        # Based on the IDs of Arg1 and Arg2 in the relations
        # the argument_df is searched for the text that belongs
        # to the IDs in question
        for arg1, arg2 in zip(seg1, seg2):
            text1 = list(argument_df.loc[argument_df["ID"]==arg1]["text"])[0]
            text2 = list(argument_df.loc[argument_df["ID"]==arg2]["text"])[0]
            seg1_text.append(text1)
            seg2_text.append(text2)

        ID = []
        for i in range(len(seg1)):
            if i < 10:
                ID.append(f"ArgEss_{filename.split(".")[0]}_0{i}")
            else:
                ID.append(f"ArgEss_{filename.split(".")[0]}_{i}")

        # Attach the texts of Arg1 and Arg1 to the relations df
        relations_df["ID"] = ID
        relations_df["seg1_text"] = seg1_text
        relations_df["seg2_text"] = seg2_text

        print(relations_df)

        # write the resulting df to a .tsv
        relations_df.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/corpus/ArgumentativeEssays-2017_parsed/ArgEss_{filename.split(".")[0]}_parsed.tsv", sep="\t", index=False)
        #except: # this is to catch potentially empty files (there was one in this corpus)
        #print(f"problem with: {filename}")