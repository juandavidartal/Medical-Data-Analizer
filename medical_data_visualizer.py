import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

#Cleaning Sex Column

df['sex'] = df.sex.astype(str)
df['sex'].replace('3', '2', inplace=True) # 3 is not a valid value

# 2
def overweight(row):
    bmi = row['weight'] / (row['height'] / 100)**2
    if bmi > 25:
        return 1
    else:
        return 0


df['overweight'] = df.apply(overweight, axis=1)


# 3 Normalize data

def normalize_data(value):
    if value == 1:
        return 0
    elif value > 1:
        return 1
    else:
        return value

df["cholesterol"] = df["cholesterol"].apply(normalize_data)
df["gluc"] = df["gluc"].apply(normalize_data)





# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 6
    df_cat = df_cat.rename(columns={"variable": "feature", "value": "count"})

    # Creating the cat plot

    df_cat = df_cat.groupby(['cardio', 'feature', 'count'], as_index=False).size().rename(columns={'size': 'total'})

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x='feature', y='total', hue='count', col='cardio', data=df_cat, kind='bar')



    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (
    df['height'] >= df['height'].quantile(0.025)) & (
    df['height'] <= df['height'].quantile(0.975)) & (
    df['weight'] >= df['weight'].quantile(0.025)) & (
    df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(10, 6))
    
    
    # 15

    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", linewidths=.5, ax=ax)


    # 16
    fig.savefig('heatmap.png')
    return fig



