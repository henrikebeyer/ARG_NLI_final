import pandas as pd
import os

# the files in this corpus are written white-space seperated
# they are structured as follows: 
# The lines containing the text of arguments contain the ID of the argument, 
# the label of the partial argument (Premise, Conclusion, Non-arg & equivalents),
# the start of the sequence, the end of the sequence, 
# the text of the argument.
# This is followed by the information on the relations:
# The ID of the relation, the label of the relation (SUPPORT, ATTACK),
# The ID of the first argument, the ID of the second argument.

# define path to corpus 
corpus_path = "/home/oenni/Dokumente/ArgumentationCorpora/FinArg/argument mining" #/AAPL_Q1_2015_17544_9.ann"

for filename in os.listdir(corpus_path):
    #print(filename)

    # read in the .ann files in the corpus directory
    if filename.endswith(".ann"):
        with open(f"{corpus_path}/{filename}", "r", encoding="utf-8") as f:
            data = f.read().split("\n") # the data in the .ann files is written whitespace-seperated

        relations = []
        arguments = []

        try:
            for line in data:
                if len(line.split(" ")) == 4: # The lines containing relations can be identified since they will always have Lenght 4 when splited by whitespaces
                    relations.append(line.split(" "))
                else: # The remaining lines contain the arguments
                    splitted = line.split(" ")
                    labels = splitted[:4] # if we split these lines by whitespaces, the first 4 entries will contain annotation information and metadata
                    text = " ".join(splitted[4:]) # the text starts after this
                    labels.append(text.lower())
                    arguments.append(labels)

            # Transfer the arguments and the relations to dataframes for easier processing
            argument_df = pd.DataFrame(arguments, columns=["ID", "relation", "start", "stop", "text"])
            relations_df = pd.DataFrame(relations, columns=["ID", "relation", "seg1_id", "seg2_id"])

            seg1_id = [arg.split(":")[1] for arg in relations_df["seg1_id"]]
            seg2_id = [arg.split(":")[1] for arg in relations_df["seg2_id"]]

            seg1_text = []
            seg2_text = []

            # Based on the IDs of seg1 and seg2 in the relations
            # the argument_df is searched for the text that belongs
            # to the IDs in question
            for arg1, arg2 in zip(seg1_id, seg2_id):
                text1 = list(argument_df.loc[argument_df["ID"]==arg1]["text"])[0]
                text2 = list(argument_df.loc[argument_df["ID"]==arg2]["text"])[0]
                seg1_text.append(text1)
                seg2_text.append(text2)

            # create a new ID over the full corpus:
            ID = []
            for i in range(len(seg1_id)):
                if i < 10:
                    ID.append(f"{filename.split(".")[0]}_0{i}")
                else:
                    ID.append(f"{filename.split(".")[0]}_{i}")

            # Attach the texts of seg1 and seg1 to the relations df
            relations_df["ID"] = [f"FinARG_{id}" for id in ID]
            relations_df["seg1_text"] = seg1_text
            relations_df["seg2_text"] = seg2_text

            # write the resulting df to a .tsv
            relations_df.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/corpus/FinArg_parsed/FinArg_{filename}_parsed.tsv", sep="\t", index=False)
        except: # this is to catch potentially empty files (there was one in this corpus)
            print(f"problem with: {filename}")