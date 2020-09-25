import pandas as pd
import numpy as np

"""MAIN"""
def clean_athlete_events(df:pd.DataFrame) -> pd.DataFrame:
    """
    DESCRIPTION
    Cleans the raw data from `athlete_events.csv`. Namely:
        - Unnecessary columns are removed (weight, height team, games, city).
        - Removes rows without age data.
        - Removes large team sports, as they can skew the data. (Baseball,Tug-Of-War,Handball,Basketball,Ice Hockey,Hockey,Football & Water Polo)
        - Removes data for results before 1948 (due to disruption to games before hand and relative unprofessionalism)
        - Removes sports which appear in less than 5 games.
        - Makes all strings lowercase

    PARAMETERS
    df (pandas.DataFrame): Dataframe of data included in `athlete_events.csv`. (Note this df is not affected by this method)

    RETURNS
    pandas.DataFrame: cleaned data
    """
    clean_df=df.copy()

    # drop unnecesary columns
    clean_df=clean_df.drop(["Height","Weight","Team","Games","City"],axis=1)

    # drop rows without age data
    # print("{:,} rows without age values removed.".format(clean_df["Age"].isnull().sum()))
    clean_df=clean_df[clean_df["Age"].notnull()].reindex()

    # remove big team sports & non-physical sports
    team_sports=["Baseball","Tug-Of-War","Handball","Basketball","Ice Hockey","Hockey","Football","Water Polo"]
    clean_df=clean_df[~clean_df["Sport"].isin(team_sports)]

    # remove data from before 1948
    clean_df=clean_df[clean_df["Year"]>=1948]

    # remove sports which appeared in less than 5 games
    games_per_sport_df=games_per_sport(clean_df)
    common_sports=games_per_sport_df[games_per_sport_df["Num_Years"]>=5]
    # print("{:,} rows removed due to sport.".format((~clean_df["Sport"].isin(common_sports.index)).sum()))
    clean_df=clean_df[clean_df["Sport"].isin(common_sports.index)]

    # make all strings lowercase
    clean_df[["Name","Season","NOC","Sport","Event","Medal"]]=clean_df[["Name","Season","NOC","Sport","Event","Medal"]].astype(str)
    for col in ["Name","Season","NOC","Sport","Event","Medal"]:
        clean_df[col]=clean_df[col].str.lower()
        clean_df[col]=clean_df[col].replace("nan",np.NaN)

    # Clean `Event` column
    clean_df["Event"]=clean_df.apply(clean_event,axis=1)

    return clean_df

def clean_noc_regions(df:pd.DataFrame) -> pd.DataFrame:
    """
    DESCRIPTION
    Cleans the raw data from `noc_regions.csv`. Namely:
        - Makes all strings lowercase
        - Renames columns to "NOC","Region","Notes"

    PARAMETERS
    df (pandas.DataFrame): Dataframe of data included in `noc_regions.csv`. (Note this df is not affected by this method)

    RETURNS
    pandas.DataFrame: cleaned data
    """
    clean_df=df.copy()

    clean_df=clean_df.reset_index()
    clean_df["NOC"]=clean_df["NOC"].str.lower()
    clean_df.columns=["NOC","Region","Notes"]

    return clean_df

"""HELPERS"""

def clean_event(row:pd.Series) -> str:
    """
    DESCRIPTION
    Cleans `Event` column by removing reference to sport and gender. (i.e. removes redundant information)

    PARAMETERS
    row (pd.Series): row to clean `Event` field of

    RETURNS
    str: clean `Event` value
    """
    event=row["Event"]; sport=row["Sport"]
    event=event.replace(sport,"")
    event=event.replace("women's","")
    event=event.replace("men's","")
    event=event.strip()
    return event if (event!="") else np.NaN

# count number of olympic games each sport has appeared at
def games_per_sport(df:pd.DataFrame) -> pd.DataFrame:
    """
    DESCRIPTION
    Count the number of games each sport has appeared at.

    PARAMETERS
    df (pandas.DataFrame): DataFrame of cleaned data from `athlete_events.csv`

    RETURNS
    pandas.DataFrame: pivot table stating the number of games each sport has appeared at.
    """
    sport_years_df=pd.pivot_table(data=df,values="Year",index="Sport",aggfunc=[lambda x:len(list(x.unique()))])
    sport_years_df.columns=["Num_Years"]
    sport_years_df["Num_Years"]=sport_years_df["Num_Years"].astype(int)
    return sport_years_df

# dataframe with details grouped by athlete_id
def unique_athlete(athlete_df:pd.DataFrame) -> pd.DataFrame:
    won_a_medal=athlete_df.groupby(by="Athlete_ID")["Medalist"].sum().astype(int)
    athlete_details=athlete_df.groupby(by="Athlete_ID")["Name","Sex","NOC","Season","Sport","Birth_Year"].first()
    unique_details=athlete_df.groupby(by="Athlete_ID")["Event","Year"].agg(['unique'])
    num_events=athlete_df.groupby(by="Athlete_ID")["Event"].agg(lambda x:len(x.unique()))
    median_age=athlete_df.groupby(by="Athlete_ID")["Age"].agg(['median']).astype(int)
    unique_athlete_df=athlete_details.merge(won_a_medal,on="Athlete_ID").merge(unique_details,on="Athlete_ID").merge(num_events,on="Athlete_ID").merge(median_age,on="Athlete_ID")
    unique_athlete_df.columns=["Name","Sex","NOC","Season","Sport","Birth_Year","Num_Medals","Events","Years","Num_Events","Median_Age"]
    unique_athlete_df["Medalist"]=(unique_athlete_df["Num_Medals"]>0)

    return unique_athlete_df
