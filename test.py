from rhf import RHF
import pandas as pd

dataset_name = "abalone"
df = pd.read_csv(f"../data/public/{dataset_name}.gz")
labels = df['label'].to_numpy(dtype='float32')
data = df.drop('label', axis=1).to_numpy(dtype='float32')

print("data: ", data.shape)

my_rhf = RHF(num_trees = 100, max_height = 5, split_criterion='kurtosis')
output_scores = my_rhf.fit(data)