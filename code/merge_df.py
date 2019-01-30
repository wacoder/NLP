from glob import glob
import pandas as pd
import os

file_list = glob('../data/train/*')
df = pd.DataFrame()
for f in file_list:
    print(f)
    department = os.path.basename(f)
    df_temp = pd.read_csv(f)
    df_temp = df_temp[['reviewText','overall']]
    df_temp['department'] = department
    df = pd.concat([df, df_temp])

df.to_csv('../data/train/merged.csv')