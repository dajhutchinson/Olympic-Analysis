import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

athlete_df=pd.read_csv("data/athlete_events.csv",index_col="ID")
noc_df=pd.read_csv("data/noc_regions.csv",index_col="NOC")

# NOTE - age data is in years (not months) so fairly inaccurate. Cannot differentiate between people born 23 months apart and those 2 months apart

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

# remove mass team sports & non-physical sports
non_physical=["art competitions","equestrianism","polo","sailing"]
team_sports=["baseball","tug-of-war","handball","basketball","ice hockey","hockey","football","water polo"]
athlete_df=athlete_df[~athlete_df["Sport"].isin(non_physical)]
athlete_df=athlete_df[~athlete_df["Sport"].isin(team_sports)]

# remove sports which appeared in less than 5 games
games_per_sport_df=games_per_sport(athlete_df)
common_sports=games_per_sport_df[games_per_sport_df["Num_Years"]>=10]
print("{} rows removed due to sport.".format((~athlete_df["Sport"].isin(common_sports.index)).sum()))
athlete_df=athlete_df[athlete_df["Sport"].isin(common_sports.index)]

"""Add Fields"""
athlete_df["Cycle_Age"]=(athlete_df["Age"]%4).astype(int)
athlete_df["Medalist"]=athlete_df["Medal"].isin(["gold","silver","bronze"])

"""
OVERVIEW ANALYSIS
"""

print("'athlete_df' contains {} rows covering:".format(athlete_df.shape[0]))
summer_games=athlete_df[athlete_df["Season"]=="summer"]
winter_games=athlete_df[athlete_df["Season"]=="winter"]
print("\t{} summer games ({}-{}) & {} winter games ({}-{}).".format(len(summer_games["Year"].unique()),summer_games["Year"].min(),summer_games["Year"].max(),len(winter_games["Year"].unique()),winter_games["Year"].min(),winter_games["Year"].max()))
print("\t{} sports ({} summer, {} winter).".format(len(athlete_df["Sport"].unique()),len(athlete_df[athlete_df["Season"]=="summer"]["Sport"].unique()),len(athlete_df[athlete_df["Season"]=="winter"]["Sport"].unique())))

both_games_sports=list(set(athlete_df[athlete_df["Season"]=="summer"]["Sport"].unique()) & set(athlete_df[athlete_df["Season"]=="winter"]["Sport"].unique()))
print("\t{} sport{} have appeared in both winter & summer games ({})".format(len(both_games_sports),"" if len(both_games_sports)==1 else "s",",".join(both_games_sports)))
print("\t{} events ({} summer, {} winter).\n".format(len(athlete_df["Event"].unique()),len(athlete_df[athlete_df["Season"]=="summer"]["Event"].unique()),len(athlete_df[athlete_df["Season"]=="winter"]["Event"].unique())))

"""
PROBLEM ANALYSIS
"""

"""Overall Age distribution of medal winners against non-winners"""
# kde plot of age distribution for medal winners against non-winners
# suggests medal winners are slightly older
xlims=(athlete_df["Age"].min(),athlete_df["Age"].max())
athlete_df[athlete_df["Medalist"]]["Age"].plot.kde(xlim=xlims,label="Medalists",color="gold")
athlete_df["Age"].plot.kde(xlim=xlims,label="Competitors",color="black")
plt.legend()
plt.show()

"""Consider age wrt point in olympic cycle (ie mod 4)"""
all_cycle_ages_counts=athlete_df["Cycle_Age"].value_counts(normalize=True)
non_medalist_cycle_ages_counts=athlete_df[~athlete_df["Medalist"]]["Cycle_Age"].value_counts(normalize=True).sort_index()
medalist_cycle_ages_counts=athlete_df[athlete_df["Medalist"]]["Cycle_Age"].value_counts(normalize=True).sort_index()

# Props of medalists & competitors born in each year of olympiad
# print("Medal Winners")
# print(non_medalist_cycle_ages_counts)
# print("Not Winners")
# print(medalist_cycle_ages_counts)
# There is a discrepenacy (is it statistical signifcant)

y_lims=(0,1.1*max(all_cycle_ages_counts.max(),non_medalist_cycle_ages_counts.max(),medalist_cycle_ages_counts.max()))
all_cycle_ages_counts.plot(color="black",label="All",ylim=y_lims)
non_medalist_cycle_ages_counts.plot(color="gray",label="Not Winners",ylim=y_lims)
medalist_cycle_ages_counts.plot(color="gold",label="Medal Winners",ylim=y_lims)
plt.legend()
plt.show()

from scipy.stats import binom

"""Test probability of being born in olympiad"""
# MEDALISTS
print("HYPOTHEIS TEST for probability of binomial RV modelling whether a *medal winner* was born in an olympic year.")
n=athlete_df[athlete_df["Medalist"]].shape[0] # medal winner
obs=athlete_df[athlete_df["Medalist"] & (athlete_df["Cycle_Age"]==0)].shape[0] # medal winner and born in olympiad
print("Num medal winners: {:,}. Num medal winners born in olympiad: {:,} (p={:.4f}).".format(n,obs,obs/n))

expected_p=1/4
p_value=1-binom.cdf(obs,n,expected_p) # H0:p=1/4, H1:p!=1/4
print("p_value={:.8f}. {}Statistically Ssignificant\n".format(p_value,"" if p_value<=.1 else "*Not* "))

# COMPETITORS
print("HYPOTHEIS TEST for probability of binomial RV modelling whether a *competitor* was born in an olympic year.")
n=athlete_df.shape[0] # competitor
obs=(athlete_df["Cycle_Age"]==0).sum() # competitor and born in olympiad
print("Num competitors: {:,}. Num competitors born in olympiad: {:,} (p={:.4f}).".format(n,obs,obs/n))

expected_p=1/4
p_value=1-binom.cdf(obs,n,expected_p) # H0:p=1/4, H1:p!=1/4
print("p_value={:.8f}. {}Statistically Significant\n".format(p_value,"" if p_value<=.1 else "*Not* "))

"""Test overall distribution"""
# MEDALISTS
from scipy.stats import chisquare

print("CHI^2 TEST for whether distribution of *medalists* ages in olympic cycle is uniform.")
f_obs=athlete_df[athlete_df["Medalist"]]["Cycle_Age"].value_counts().sort_index().values
f_exp=np.repeat(sum(f_obs)/4,4)
print("Observed occs: {}. Expected occs: {}".format(f_obs,f_exp))
_,p_value=chisquare(f_obs,f_exp)
print("p_value={:.8f}. {}Statistically Significant\n".format(p_value,"" if p_value<=.1 else "*Not* "))

# all competitors
print("CHI^2 TEST for whether distribution of *competitors* ages in olympic cycle is uniform.")
f_obs=athlete_df["Cycle_Age"].value_counts().sort_index().values
f_exp=np.repeat(sum(f_obs)/4,4)
print("Observed occs: {}. Expected occs: {}".format(f_obs,f_exp))
_,p_value=chisquare(f_obs,f_exp)
print("p_value={:.8f}. {}Statistically Significant\n".format(p_value,"" if p_value<=.1 else "*Not* "))

"""
SPLIT BY GENDER
"""
male_athletes=athlete_df[athlete_df["Sex"]=="M"].copy()
female_athletes=athlete_df[athlete_df["Sex"]=="F"].copy()

# Distribution of ages of medalists and competitors, split by gender
xlims=(athlete_df["Age"].min(),athlete_df["Age"].max())
male_athletes[male_athletes["Medalist"]]["Age"].plot.kde(xlim=xlims,label="Male Medalists",color="blue")
male_athletes["Age"].plot.kde(xlim=xlims,label="Male Competitors",color="black")
female_athletes[female_athletes["Medalist"]]["Age"].plot.kde(xlim=xlims,label="Female Medalists",color="pink")
female_athletes["Age"].plot.kde(xlim=xlims,label="Female Competitors",color="gray")
plt.legend()
plt.show()
# Male distributions are almost identical, the womans' are not. Suggests that older female competitors are more likely to win.
# This may be due to the sports that women compete in compared to men (female gymansts are notably younger than male)

"""
AGE SPLIT BY SPORT
"""
# sort sports by mean age increasing
mean_age_per_sport=athlete_df.groupby(by=["Sport"])["Age"].mean().sort_values()
num_sports=mean_age_per_sport.shape[0]

# plot of age distribution for each sports
fig,axes=plt.subplots(ncols=3,nrows=int(np.ceil(num_sports/3)))

# ensure plots are all same scale
xlims=(athlete_df["Age"].min(),athlete_df["Age"].max())
ylims=(0,.15)

for count,sport in enumerate(list(mean_age_per_sport.index)):
    row_i=count//3; col_i=count%3
    ax=axes[row_i,col_i]
    data=athlete_df[athlete_df["Sport"]==sport]

    # plot data
    data[data["Medalist"]]["Age"].plot.kde(ax=ax,xlim=xlims,ylim=ylims,label="Medalists",color="gold")
    data["Age"].plot.kde(ax=ax,xlim=xlims,ylim=ylims,label="Competitors",color="black")

    # add horizontal line for mean of each
    ax.axvline(x=data[data["Medalist"]]["Age"].mean(),color="gold",ls="--")
    ax.axvline(x=data["Age"].mean(),color="black",ls="--")

    # style plot
    ax.set_title(str.capitalize(sport))
    ax.set_yticks([0,.05,.1,.15])
    if (col_i==0): ax.set_ylabel("Density") # LHS
    else: ax.set_ylabel(""); ax.set_yticklabels([])
    if (count+3>=num_sports): ax.set_xlabel("Age (Years)") # bottom of each column
    else: ax.set_xlabel(""); ax.set_xticklabels([])

# remove excess plots
for i in range(num_sports,3*int(np.ceil(num_sports/3))):
    row_i=i//3; col_i=i%3
    fig.delaxes(axes[row_i,col_i])

# legend data
from matplotlib.lines import Line2D
colors=['black',"black",'gold',"gold"]
style =["-","--","-","--"]
lines =[Line2D([0],[0],linewidth=3,linestyle=style[i],color=colors[i]) for i in range(4)]
labels=["Competitors","Mean","Medalists","Mean"]
fig.legend(lines,labels,title="Legend",loc="top right")

plt.show()

"""
NORMALISED AGE SPLIT BY SPORT
"""

sport_groups=athlete_df.groupby(by=["Sport"])
mean_age_per_sport=sport_groups["Age"].mean().sort_values()

# cycle age distribution per sport (Medalists)
prop_func=lambda x,n:list(x).count(n)/len(x)
competitors_sport_cycle_age_pt=pd.pivot_table(data=athlete_df,values="Cycle_Age",index="Sport",aggfunc=[lambda x:prop_func(x,0),lambda x:prop_func(x,1),lambda x:prop_func(x,2),lambda x:prop_func(x,3)])
competitors_sport_cycle_age_pt.columns=["0","1","2","3"]
competitors_cycle_age_mean=athlete_df["Cycle_Age"].value_counts(normalize=True).sort_index()

# cycle age distribution per sport (Competitors)
medalists_sport_cycle_age_pt=pd.pivot_table(data=athlete_df[athlete_df["Medalist"]],values="Cycle_Age",index="Sport",aggfunc=[lambda x:prop_func(x,0),lambda x:prop_func(x,1),lambda x:prop_func(x,2),lambda x:prop_func(x,3)])
medalists_sport_cycle_age_pt.columns=["0","1","2","3"]
medalists_cycle_age_mean=athlete_df[athlete_df["Medalist"]]["Cycle_Age"].value_counts(normalize=True).sort_index()

# legend data
from matplotlib.lines import Line2D
colors=['gray','black','red']
style =["-","-","--"]
lines =[Line2D([0],[0],linewidth=3,linestyle=style[i],color=colors[i]) for i in range(3)]
labels=["Individual Sport Proportions","Overall Proportion","Expected"]

competitors_sport_cycle_age_pt.transpose().plot.line(color="gray",ylim=(.15,.35))
plt.xlabel("Cycle Age")
plt.ylabel("Proportion")
plt.title("Cycle Age of Competitors by Sport")
plt.plot(list(competitors_cycle_age_mean.index),competitors_cycle_age_mean.values,color="black") # plot mean values
plt.plot([0,1,2,3], [.25,.25,.25,.25], '--',color="red")

plt.legend(lines,labels)

medalists_sport_cycle_age_pt.transpose().plot.line(color="gray",ylim=(.15,.35))
plt.plot(list(medalists_cycle_age_mean.index),medalists_cycle_age_mean.values,color="black") # plot mean values
plt.plot([0,1,2,3], [.25,.25,.25,.25], '--',color="red")
plt.xlabel("Cycle Age")
plt.ylabel("Proportion")
plt.title("Cycle Age of Medalists by Sport")

plt.legend(lines,labels)

plt.show()

# TODO
# consider each sport independetly
# Change in age over time
# Change by country
