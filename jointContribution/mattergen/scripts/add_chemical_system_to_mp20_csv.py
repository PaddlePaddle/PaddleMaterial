import pandas as pd

input_file = "datasets/mp_20/train.csv"
save_file = "datasets/mp_20_chemical_system/train.csv"

df = pd.read_csv(input_file)
# read elements
elements = df['elements']
df['chemical_system'] = elements.apply(lambda x: '-'.join(eval(x)))
df.to_csv(save_file, index=False)

