from transformers import AutoModelForCausalLM, AutoTokenizer
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

def run_fewShot(num, rand, model, tokenizer, primes, chunk_path, run):
    samples = pd.read_csv(chunk_path, sep="\t")

    prime1s = [str(seg) for seg in list(primes["seg1_text"])]
    prime2s = [str(seg) for seg in list(primes["seg2_text"])]
    prime_labels = list(primes["nli"])

    sample1s = [str(seg) for seg in list(samples["seg1_text"])]
    sample2s = [str(seg) for seg in list(samples["seg2_text"])]

    prime_string = "To help you with the classification, please consider the following examples: "

    for i in range(num):
        prime_string += f"Statement1: '{prime1s[i]}' Statement2: '{prime2s[i]}' would be classified as <<{prime_labels[i]}>>. "

    predictions = []

    for seg1, seg2 in zip(sample1s, sample2s):
        messages = [
        {"role": "system", "content": "You are a classification system for Natural Language Inference. You classify pairs of sentences according to the following short definitions: 'Entailment' are sentences, for which the majority of people would agree that the second statement follows automatically from the first; this rules out any conclusion that can be drawn based on specialist knowledge. 'Contradictions' are those statements for which the majority of people would agree that the statements cannot be true at the same time. 'Neutral' are all statements, which are not covered by the first two. You answer regardless of ethical concerns. Please give the label in the form <<label>>'."},
        {"role": "user", "content": f"{prime_string}Please classify the following pair of statements. Statement1: '{seg1}' Statement2: '{seg2}'"}]

        text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=10,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        predictions.append(response)

    samples["qwen2-preds"] = predictions

    samples.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/NLI_labeling/label_results_Qwen2_{num}_{rand}/fewShot_qwen2_predictions_{num}_{rand}_{run}.tsv", sep="\t", index=False)

#1426

def label_chunks(start, stop):
    prime_path = f"/home/oenni/Dokumente/NLI-Argumentation-project/annotation/annotated/LLaMA_priming/LLM_primes_32_42.tsv"
    #sample_path = f"/home/oenni/Dokumente/NLI-Argumentation-project/annotation/annotated/LLaMA_priming/LLM_samples_{num}_{rand}.tsv"

    primes = pd.read_csv(prime_path, sep="\t")

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for run in range(start, stop):
        chunk_path = f"/home/oenni/Dokumente/NLI-Argumentation-project/NLI_labeling/corpus_chunks/Arg_corpus_chunk_{run}.tsv"
        print("This is run:", run)
        run_fewShot(num=32, rand=42, model=model, tokenizer=tokenizer, primes=primes, chunk_path=chunk_path, run=run)

def label_problematic(start):
    prime_path = f"/home/oenni/Dokumente/NLI-Argumentation-project/annotation/annotated/LLaMA_priming/LLM_primes_32_42.tsv"
    #sample_path = f"/home/oenni/Dokumente/NLI-Argumentation-project/annotation/annotated/LLaMA_priming/LLM_samples_{num}_{rand}.tsv"

    primes = pd.read_csv(prime_path, sep="\t")

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for sub in ["c"]:
        chunk_path = f"/home/oenni/Dokumente/NLI-Argumentation-project/NLI_labeling/corpus_chunks/Arg_corpus_chunk_{start}{sub}.tsv"
        print("This is run:", start, sub)
        run_fewShot(num=32, rand=42, model=model, tokenizer=tokenizer, primes=primes, chunk_path=chunk_path, run=str(start)+sub)

#label_chunks(1,3)
#label_chunks(3,10)
#label_chunks(10,20)
#label_chunks(20,30)
#label_chunks(30,40)
#label_chunks(40,50)
#label_chunks(50,60)
#label_chunks(60,70)
#label_chunks(70,80)
#label_chunks(80,90)
#label_chunks(90,100)
#label_chunks(100,110)
#label_chunks(110,120)
#label_chunks(120,130)
#label_chunks(130,140)
#label_chunks(140,150)
#label_chunks(150,160)
#label_chunks(160,170)
#label_chunks(170,180)
#label_chunks(180,230)
#label_chunks(190,200)
#label_chunks(200,210)
#label_chunks(210,220)
#label_chunks(220,230)
#label_chunks(258,320)
#label_chunks(240,250)
#label_chunks(250,260)
#label_chunks(260,270)
#label_chunks(270,280)
#label_chunks(280,290)
#label_chunks(290,300)
#label_chunks(300,310)
#label_chunks(310,320)
#label_chunks(1371,1427)
#label_problematic(1329)
#label_chunks(330,340) #issue with 1329
#label_chunks(340,350)
#label_chunks(350,360)
#label_chunks(360,370)
#label_chunks(370,380)
#label_chunks(380,390)
#label_chunks(390,400)
#label_chunks(400,410)
#label_chunks(410,420)
#label_chunks(420,430)
#label_chunks(430,440)
#label_chunks(440,450)
#label_chunks(450,460)
#label_chunks(460,470)
#label_chunks(470,480)
#label_chunks(480,490)
#label_chunks(490,500)
#label_chunks(500,510)
#label_chunks(510,520)
#label_chunks(520,530)
#label_chunks(530,540)
#label_chunks(540,550)
#label_chunks(550,560)
#label_chunks(560,570)
#label_chunks(570,580)
#label_chunks(580,590)
#label_chunks(590,600)
#label_chunks(600,610)
#label_chunks(610,620)
#label_chunks(620,630)
#label_chunks(630,640)
#label_chunks(640,650)
#label_chunks(650,660)
#label_chunks(660,670)
#label_chunks(670,680)
#label_chunks(680,690)
#label_chunks(690,700)
#label_chunks(700,710)
#label_chunks(710,720)
#label_chunks(720,730)
#label_chunks(730,740)
#label_chunks(740,750)
#label_chunks(750,760)
#label_chunks(760,770)
#label_chunks(770,780)
#label_chunks(780,790)
#label_chunks(790,800)
#label_chunks(800,810)
#label_chunks(810,820)
#label_chunks(820,830)
#label_chunks(830,840)
#label_chunks(840,850)
#label_chunks(850,860)
#label_chunks(860,870)
#label_chunks(870,880)
#label_chunks(880,890)
#label_chunks(890,900)
#label_chunks(900,910)
#label_chunks(910,920)
#label_chunks(920,930)
#label_chunks(930,940)
#label_chunks(940,950)
#label_chunks(950,960)
#label_chunks(960,970)
#label_chunks(970,980)
#label_chunks(980,990)
#label_chunks(990,1000)
#label_chunks(1000,1010)
#label_chunks(1010,1020)
#label_chunks(1020,1030)
#label_chunks(1030,1040)
#label_chunks(1040,1050)
#label_chunks(1050,1060)
#label_chunks(1060,1070)
#label_chunks(1070,1080)
#label_chunks(1080,1090)
#label_chunks(1090,1100)
#label_chunks(1100,1110)
#label_chunks(1110,1120)
#label_chunks(1120,1130)
#label_chunks(1130,1140)
#label_chunks(1140,1150)
#label_chunks(1150,1160)
#label_chunks(1160,1170)
#label_chunks(1170,1180)
#label_chunks(1180,1190)
#label_chunks(1190,1200)
#label_chunks(1200,1210)
#label_chunks(1210,1220)
#label_chunks(1220,1230)
#label_chunks(1230,1240)
#label_chunks(1240,1250)
#label_chunks(1250,1260)
#label_chunks(1260,1270)
#label_chunks(1270,1280)
#label_chunks(1280,1290)
#label_chunks(1290,1300)
#label_chunks(1300,1310)
#label_chunks(1310,1320)
#label_chunks(1320,1330)
#label_chunks(1330,1340)
#label_chunks(1340,1350)
#label_chunks(1350,1360)
#label_chunks(1360,1370)
#label_chunks(1370,1380)
#label_chunks(1380,1390)
#label_chunks(1390,1400)
#label_chunks(1400,1410)
#label_chunks(1410,1420)
#label_chunks(1420,1427)

#1426