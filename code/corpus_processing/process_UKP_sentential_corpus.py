import pandas as pd
import os

corpus_path = "/home/oenni/Dokumente/ArgumentationCorpora/UKP_sentential_argument_mining/data" #/school_uniforms.tsv"

segment_dict = {"abortion":"Women should have the opportunity to get legal and safe abortion if they want", 
                "cloning":"Cloning should be considered as a promising opportunity for human development.",
                "death_penalty":"Death penalty should be an option in the penal code.",
                "gun_control":"Gun laws should be more restrictive.",
                "marijuana_legalization":"The consumption of marijuana should be legalized.",
                "minimum_wage":"The minimum wage needs to rise and to adapt to the current cost of living.",
                "nuclear_energy":"Nuclear energy is a good energy source for the future.",
                "school_uniforms_newFormat":"School uniforms should be established in schools."}

df_list = []

for filename in os.listdir(corpus_path):
    if filename != "school_uniforms.tsv":
        print(filename)
        data = pd.read_csv(f"{corpus_path}/{filename}", sep="\t")

        topic = filename.split(".")[0]
        seg1_text = [segment_dict[topic] for i in range(len(list(data.topic)))]
        ID = [f"UKP_{topic}_000{i}" if i < 10 
              else f"UKP_{topic}_00{i}" if i < 100 
              else f"UKP_{topic}_0{i}" if i < 1000 
              else f"UKP_{topic}_{i}" for i in range(len(list(data.topic)))]
        relation = ["support" if rel=="Argument_for" 
                    else "attack" if rel=="Argument_against" 
                    else "unrelated" for rel in list(data.annotation)]

        ass_df = pd.DataFrame()
        ass_df["ID"] = ID
        ass_df["relation"] = relation
        ass_df["seg1_text"] = seg1_text
        ass_df["seg2_text"] = data.sentence
        ass_df["topic"] = data.topic
        ass_df["seg2_Hash"] = data.sentenceHash

        df_list.append(ass_df)
        
full_df = pd.concat(df_list) 

print(full_df.value_counts("relation"))
print(full_df.value_counts("topic"))      


full_df.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/corpus/UKP_sentential_parsed/UKP_sentential_parsed.tsv", sep="\t", index=False)