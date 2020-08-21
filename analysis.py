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
"""
Age distribution of medal winners against non-winners
"""
"""
# kde plot of age distribution for medal winners against non-winners
# suggests medal winners are slightly older
xlims=(athlete_df["Age"].min(),athlete_df["Age"].max())
athlete_df[athlete_df["Medal"].notnull()]["Age"].plot.kde(xlim=xlims,label="Medal Winners",color="gold")
athlete_df[athlete_df["Medal"].isnull()]["Age"].plot.kde(xlim=xlims,label="No Medal",color="black")
plt.legend()
plt.show()
"""

"""
Remove sports with little data or with big teams
"""
print(athlete_df["Sport"].value_counts())

# count number of olympic games each sport has appeared at
def games_per_sport(df):
    pass

# TODO
# remove certain sports (big team games)
# consider each sport independetly
# split by gender
# Remove sports which appear at few games (ie less than 10?)
