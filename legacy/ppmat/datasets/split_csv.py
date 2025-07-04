import pandas as pd

df = pd.read_csv("your-dataset.csv")

# random shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

subset1 = df.head(18000)
subset2 = df.tail(len(df) - 18000)

# save the subsets to csv files
subset1.to_csv("train.csv", index=False)
subset2.to_csv("val.csv", index=False)
