import pandas as pd
import os

# The files parsed by this script are 

corpus_path = "/home/oenni/Dokumente/ArgumentationCorpora/abstrct-master/AbstRCT_corpus/data/dev/neoplasm_dev" #/9093725.ann"
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
            if l[0].startswith("R"):
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
                            "label":label,
                            "start":start,
                            "end":end,
                            "text":text}
                )

        #print(arguments)

        relations_df = pd.DataFrame(relations)
        argument_df = pd.DataFrame(arguments)

        seg1_id = [arg.split(":")[1] for arg in relations_df["seg1_id"]]
        seg2_id = [arg.split(":")[1] for arg in relations_df["seg1_id"]]

        seg1_text = []
        seg2_text = []

        # Based on the IDs of Arg1 and Arg2 in the relations
        # the argument_df is searched for the text that belongs
        # to the IDs in question
        for arg1, arg2 in zip(seg1_id, seg2_id):
            text1 = list(argument_df.loc[argument_df["ID"]==arg1]["text"])[0]
            text2 = list(argument_df.loc[argument_df["ID"]==arg2]["text"])[0]
            seg1_text.append(text1)
            seg2_text.append(text2)

        ID = []
        for i in range(len(seg1_id)):
            if i < 10:
                ID.append(f"AbstRCT_{filename.split(".")[0]}_0{i}")
            else:
                ID.append(f"AbstRCT_{filename.split(".")[0]}_{i}")

        # Attach the texts of Arg1 and Arg1 to the relations df
        relations_df["ID"] = ID
        relations_df["seg1_text"] = seg1_text
        relations_df["seg2_text"] = seg2_text
        relations_df["relation"] = ["support" if rel.lower()=="support" else "attack" for rel in relations_df["relation"]]

        #print(relations_df)
        # write the resulting df to a .tsv
        relations_df.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/corpus/AbstRCT_parsed/AbstRCT_{filename.split(".")[0]}_parsed.tsv", sep="\t", index=False)


corpus_path = "/home/oenni/Dokumente/ArgumentationCorpora/abstrct-master/AbstRCT_corpus/data/test" #/9093725.ann"
for subdir in os.listdir(corpus_path):
    for filename in os.listdir(f"{corpus_path}/{subdir}"):
        if filename.endswith(".ann"):
            print(subdir, filename)
            with open(f"{corpus_path}/{subdir}/{filename}", "r") as f:
                data = f.read().split("\n")

            #print(data)

            stances = []
            relations = []
            arguments = []

            for line in data:
                l = line.split("\t")
                if l[0].startswith("R"):
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
                                "label":label,
                                "start":start,
                                "end":end,
                                "text":text}
                    )

            #print(arguments)

            relations_df = pd.DataFrame(relations)
            argument_df = pd.DataFrame(arguments)

            if relations != []:

                seg1_id = [arg.split(":")[1] for arg in relations_df["seg1_id"]]
                seg2_id = [arg.split(":")[1] for arg in relations_df["seg2_id"]]

                seg1_text = []
                seg2_text = []

                # Based on the IDs of Arg1 and Arg2 in the relations
                # the argument_df is searched for the text that belongs
                # to the IDs in question
                for arg1, arg2 in zip(seg1_id, seg2_id):
                    text1 = list(argument_df.loc[argument_df["ID"]==arg1]["text"])[0]
                    text2 = list(argument_df.loc[argument_df["ID"]==arg2]["text"])[0]
                    seg1_text.append(text1)
                    seg2_text.append(text2)

                ID = []
                for i in range(len(seg1_id)):
                    if i < 10:
                        ID.append(f"AbstRCT_{filename.split(".")[0]}_0{i}")
                    else:
                        ID.append(f"AbstRCT_{filename.split(".")[0]}_{i}")

                # Attach the texts of Arg1 and Arg1 to the relations df
                relations_df["ID"] = ID
                relations_df["seg1_text"] = seg1_text
                relations_df["seg2_text"] = seg2_text
                relations_df["relation"] = ["support" if rel.lower()=="support" else "attack" for rel in relations_df["relation"]]

                #print(relations_df)
                # write the resulting df to a .tsv
                relations_df.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/corpus/AbstRCT_parsed/AbstRCT_{filename.split(".")[0]}_parsed.tsv", sep="\t", index=False)

corpus_path = "/home/oenni/Dokumente/ArgumentationCorpora/abstrct-master/AbstRCT_corpus/data/train/neoplasm_train" #/9093725.ann"
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
            if l[0].startswith("R"):
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
                            "label":label,
                            "start":start,
                            "end":end,
                            "text":text}
                )

        #print(arguments)
        if relations != []:
            relations_df = pd.DataFrame(relations)
            argument_df = pd.DataFrame(arguments)

            seg1_id = [arg.split(":")[1] for arg in relations_df["seg1_id"]]
            seg2_id = [arg.split(":")[1] for arg in relations_df["seg2_id"]]

            seg1_text = []
            seg2_text = []

            # Based on the IDs of Arg1 and Arg2 in the relations
            # the argument_df is searched for the text that belongs
            # to the IDs in question
            for arg1, arg2 in zip(seg1_id, seg2_id):
                text1 = list(argument_df.loc[argument_df["ID"]==arg1]["text"])[0]
                text2 = list(argument_df.loc[argument_df["ID"]==arg2]["text"])[0]
                seg1_text.append(text1)
                seg2_text.append(text2)

            ID = []
            for i in range(len(seg1_id)):
                if i < 10:
                    ID.append(f"AbstRCT_{filename.split(".")[0]}_0{i}")
                else:
                    ID.append(f"AbstRCT_{filename.split(".")[0]}_{i}")

            # Attach the texts of Arg1 and Arg1 to the relations df
            relations_df["ID"] = ID
            relations_df["relation"] = ["support" if rel.lower()=="support" else "attack" for rel in relations_df["relation"]]
            relations_df["seg1_text"] = seg1_text
            relations_df["seg2_text"] = seg2_text

            #print(relations_df)
            # write the resulting df to a .tsv
            relations_df.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/corpus/AbstRCT_parsed/AbstRCT_{filename.split(".")[0]}_parsed.tsv", sep="\t", index=False)