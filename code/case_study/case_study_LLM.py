import pandas as pd
from deepseek_pred import get_deepseek_pred
# from parse_IAM_data import load_train_data, load_test_data
from tqdm import tqdm

MODEL = "deepseek-r1:32b"
TEST_DATA = "MyDataset/train.csv"

df_test = pd.read_csv(TEST_DATA)
# print(f"""Loaded test data with:
#           - {df_test.value_counts("relation_type")["supports"]} Support pairs,
#           - {df_test.value_counts("relation_type")["attacks"]} Attack pairs,
#           - {df_test.value_counts("relation_type")["unrelated"]} Unrelated pairs""")
print(f"Running baseline with {MODEL}...")
tqdm.pandas()

df_test[['pred', 'deepseek_output']] = df_test.progress_apply(lambda row: get_deepseek_pred(row['source'], row['target']), axis=1)
df_test.to_csv("Arg_preds_myData_train_DeepseekR1_32b.csv", index=False)