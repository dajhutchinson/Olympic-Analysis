import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

athlete_df=pd.read_csv("data/athlete_events.csv",index_col="ID")
noc_df=pd.read_csv("data/noc_regions.csv",index_col="NOC")

"""
CLEAN DATA
"""
# strip event string to remove reference to gender and sport
def clean_event(row):
    event=row["Event"]; sport=row["Sport"]
    event=event.replace(sport,"")
    event=event.replace("women's","")
    event=event.replace("men's","")
    event=event.strip()
    return event if event!="" else np.NaN

# drop unnecesary columns
athlete_df=athlete_df.drop(["Height","Weight","Team","Games","City"],axis=1)

# drop rows without age data
print("{} rows without age values removed.".format(athlete_df["Age"].isnull().sum()))
athlete_df=athlete_df[athlete_df["Age"].notnull()].reindex()

# make all strings lowercase
athlete_df[["Name","Season","NOC","Sport","Event","Medal"]]=athlete_df[["Name","Season","NOC","Sport","Event","Medal"]].astype(str)
for col in ["Name","Season","NOC","Sport","Event","Medal"]:
    athlete_df[col]=athlete_df[col].str.lower()
    athlete_df[col]=athlete_df[col].replace("nan",np.NaN)

# clean row data
athlete_df["Event"]=athlete_df.apply(clean_event,axis=1)

"""Sports"""
# Remove sports with little data, big teams or non-physical

# count number of olympic games each sport has appeared at
def games_per_sport(df):
    sport_years_df=pd.pivot_table(data=athlete_df,values="Year",index="Sport",aggfunc=[lambda x:len(x.unique()),lambda x:list(x.unique())])
    sport_years_df.columns=["Num_Years","Years"]
    return sport_years_df
games_per_sport_df=games_per_sport(athlete_df)
games_per_sport_df=games_per_sport_df.sort_values("Num_Years")

# remove sports which appeared in less than 5 games
common_sports=games_per_sport_df[games_per_sport_df["Num_Years"]>=5]

# remove mass team sports & non-physical sports
non_physical=["art competitions"]
team_sports=["baseball","tug-of-war","handball","basketball","ice hockey","hockey","football"]
common_sports=common_sports.drop(non_physical,axis=0).drop(team_sports,axis=0)

print("{} rows removed due to sport.".format((~athlete_df["Sport"].isin(common_sports.index)).sum()))
athlete_df=athlete_df[athlete_df["Sport"].isin(common_sports.index)]

"""
OVERVIEW ANALYSIS
"""

print("'athlete_df' contains {} rows covering:".format(athlete_df.shape[0]))
print("\t{} summer games ({}-{}) & {} winter games ({}-{}).".format(len(athlete_df[athlete_df["Season"]=="summer"]["Year"].unique()),athlete_df[athlete_df["Season"]=="summer"]["Year"].min(),athlete_df[athlete_df["Season"]=="summer"]["Year"].max(),len(athlete_df[athlete_df["Season"]=="winter"]["Year"].unique()),athlete_df[athlete_df["Season"]=="winter"]["Year"].min(),athlete_df[athlete_df["Season"]=="winter"]["Year"].max()))
print("\t{} sports ({} summer, {} winter).".format(len(athlete_df["Sport"].unique()),len(athlete_df[athlete_df["Season"]=="summer"]["Sport"].unique()),len(athlete_df[athlete_df["Season"]=="winter"]["Sport"].unique())))
both_games_sports=list(set(athlete_df[athlete_df["Season"]=="summer"]["Sport"].unique()) & set(athlete_df[athlete_df["Season"]=="winter"]["Sport"].unique()))
print("\t{} sport{} have appeared in both winter & summer games ({})".format(len(both_games_sports),"" if len(both_games_sports)==1 else "s",",".join(both_games_sports)))
print("\t{} events ({} summer, {} winter).".format(len(athlete_df["Event"].unique()),len(athlete_df[athlete_df["Season"]=="summer"]["Event"].unique()),len(athlete_df[athlete_df["Season"]=="winter"]["Event"].unique())))

# TODO
# consider each sport independetly
# split by gender
# Remove sports which appear at few games (ie less than 10?)
