import pandas as pd
import os
import numpy as np
#import torch

#model and tokenizer for embeddings
#from transformers import BertTokenizer, BertModel

#sentiment
from flair.models import TextClassifier
from flair.data import Sentence

#cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

#stemming
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

#semantic texutal similarity
from sentence_transformers import SentenceTransformer

#dependency parser
#from spacy_conll import init_parser

#syntactic analyser
import textdescriptives as td

corpus_list = [#"AAAI2018",
               #"AbstRCT",
               #"ArgEss",
               #"ComArg",
               #"FinArg",
               #"IBM",
               #"IJCAI2015",
               #"JoDeb",
               #"JoKialo",
               #"micro",
               #"NK",
               #"QT30",
               #"QT50",
               #"UKP",
               #"Web",
               #"Webis-Debate",
            #    "US2016",
                "12AngryMen",
                "Debatepedia"]

prelim_path = "/home/oenni/Dokumente/NLI-Argumentation-project/feature_analysis/automatic/preliminary"

in_path = "/home/oenni/Dokumente/NLI-Argumentation-project/corpus/corpus_splits/per_corpus"

corpus_path = "/home/oenni/Dokumente/NLI-Argumentation-project/corpus/corpus_nli_preds_threshold"

def read_in_corpus(path, config):
    df = pd.read_csv(f"{path}_{config}_0.999.tsv", sep="\t")#.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
    return df

#print(df_list[0]["seg1_text"][0])

def get_sentiment(df):
    classifier = TextClassifier.load("en-sentiment")

    #for df in df_list:
    labels_seg1 = []
    labels_seg2 = []

    for seg1, seg2 in zip(df["seg1_text"], df["seg2_text"]):
        sentence1 = Sentence(str(seg1))
        classifier.predict(sentence1)
        labels_seg1.append(sentence1.labels[0].value)

        sentence2 = Sentence(str(seg2))
        classifier.predict(sentence2)
        labels_seg2.append(sentence2.labels[0].value)
    
    df["seg1_senti"] = labels_seg1
    df["seg2_senti"] = labels_seg2

    #print(df)
    
    return df

#

def jaccard_similarity(x,y):
    return len(set(x).intersection(set(y))) / len(set(x).union(set(y)))

# def get_cosine_similarity(df):
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     model = BertModel.from_pretrained("bert-base-uncased")

#     cosine_sims = []
#     for seg1, seg2 in zip(df["seg1_text"], df["seg2_text"]):
#         tokens1 = ["[CLS]"] + word_tokenize(seg1.lower()) + ["SEP"]
#         tokens2 = ["[CLS]"] + word_tokenize(seg2.lower()) + ["SEP"]
        

#         input_ids1 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens1)).unsqueeze(0)
#         input_ids2 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens2)).unsqueeze(0)

#         #Obtain the BERT embeddings
#         with torch.no_grad():
#             outputs1 = model(input_ids1)
#             outputs2 = model(input_ids2)
#             emb1 = outputs1.last_hidden_state[:, 0, :]  # [CLS] token
#             emb2 = outputs2.last_hidden_state[:, 0, :]  # [CLS] token

#         emb1 = model.token.reshape(1,-1)
#         emb2 = model.encode(seg2).reshape(1,-1)
        
#         cosine_sim = cosine_similarity(emb1, emb2)
#         print(cosine_sim)
#         cosine_sims.append(cosine_sim[0][0])

#     df["cosine_bert"] = cosine_sims
#     return df

#get_cosine_similarity(df_list)

def get_jaccard_similartiy(df):
    stemmer = SnowballStemmer(language="english")
    #for df in df_list:
    jaccard_sims = []
    for seg1, seg2 in zip(df["seg1_text"], df["seg2_text"]):
        tokenized1 = word_tokenize(str(seg1))
        stemmed1 = [stemmer.stem(token) for token in tokenized1]
        
        tokenized2 = word_tokenize(str(seg2))
        stemmed2 = [stemmer.stem(token) for token in tokenized2]

        jaccard_sim = jaccard_similarity(stemmed1, stemmed2)

        jaccard_sims.append(jaccard_sim)

    df["jaccard_sim"] = jaccard_sims
    print(df)
    return df

def get_semantic_textual_similarity(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    #for df in df_list:
    sts = []
    for seg1, seg2 in zip(df["seg1_text"], df["seg2_text"]):
        embeddings1 = model.encode(str(seg1))
        embeddings2 = model.encode(str(seg2))

        sim = model.similarity(embeddings1, embeddings2)
        #print(sim[0][0].item())
        sts.append(sim[0][0].item())
    df["cosine_miniLM"] = sts

#        print(df)
    return df

#get_semantic_textual_similarity(df_list)

def get_conll_format_parsing(text):
    nlp = init_parser("en_core_web_sm",
                    "spacy",
                    ext_names={"conll_pd": "pandas"},
    )
    doc = nlp(text)
    return doc
    #print(doc._.pandas)


def get_textual_measures(df):
    #for df in df_list:
    seg1s = [str(seg1) for seg1 in df["seg1_text"]]
    seg2s = [str(seg2) for seg2 in df["seg2_text"]]


    df1 = td.extract_metrics(text=seg1s, spacy_model="en_core_web_lg", metrics=["descriptive_stats", "readability", "dependency_distance", "pos_proportions", "quality"]).drop(["text", "smog"], axis=1)
    df1.rename(columns={col:f"seg1_{col}" for col in df1.columns}, inplace=True)
    #print(df1)
    df2 = td.extract_metrics(text=seg2s, spacy_model="en_core_web_lg", metrics=["descriptive_stats", "readability", "dependency_distance", "pos_proportions", "quality"]).drop(["text", "smog"], axis=1)
    df2.rename(columns={col:f"seg2_{col}" for col in df2.columns}, inplace=True)

    #print(df2)
    joint = [seg1 + ". " + seg2 for seg1, seg2 in zip(seg1s, seg2s)]
    
    df_joint = td.extract_metrics(text=joint, spacy_model="en_core_web_lg", metrics=["coherence"]).drop(["text", "second_order_coherence"], axis=1)
    df_joint.rename(columns={col:f"joint_{col}" for col in df_joint.columns}, inplace=True)
    print(df_joint)

    df = pd.concat([df, df1, df2, df_joint], axis=1)


    #for col in df.columns:
    #    print(col)
    #print(df.columns)
    print(df)

        #out_df_list.append(df)

    
    #print(df.columns)
    return df


def write_out_results(df, out_path, corpus):
   df.to_csv(f"{out_path}{config}_0.999_withFeatures.tsv", sep="\t")

out_path = "/home/oenni/Dokumente/NLI-Argumentation-project/feature_analysis/automatic/corpus_nli_preds_threshold"

config_list = ["hard", "soft"]
for config in config_list:
    print(config)
    df = read_in_corpus(corpus_path, config)
    df = get_sentiment(df)
    df = get_semantic_textual_similarity(df)
    # df_list = get_cosine_similarity(df_list)
    df = get_jaccard_similartiy(df)
    df = get_textual_measures(df)

    write_out_results(df, out_path, config)
    print("done")