# Olympic Birth Year Analysis

**Table of Contenst**
- [The Question](#The-Question)
- [Summary of Findings](#Summary-of-Findings)
  - [Tidbit](#Tidbit)
- [Guide to Project](#Guide-to_Project)
  - [The Data](#The-Data)
    - [Limitations](#Limitations)
    - [Data Used](#Data-Used)
  - [Notebooks](#Notebooks)
- [Further Analysis](#Further-Analysis)

## The Question
Does the year you are born in an Olympic cycle affect the likelihood of you becoming an Olympic athlete or medalist?

## Summary of Findings
There is very little evidence that when you are born in an Olympic Cycle affects your chance of becoming an Olympic Athlete and almost certainly has no affect on the majority of sports.

However, I show that female gymnasts are among the youngest athletes, have some of the shortest Olympic careers and that their age distribution is statistically non-uniform. Thus, when you are born in an Olympic cycle has the greatest affect (and likely a tangible one) on becoming a female olympic gymnast.

### Tidbit
Rowing has a surprisingly high number of pre-teen athletes. This is due to the common use of young boys during the early games for their weight advantage when compared to using adults. (This is no longer common practice.)

## Guide to Project

### The Data
Data is taken from [Kaggle](https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results).
The data files are not included due to their size. My code uses the same filenames as Kaggle (`athlete_events.csv` and `noc_regions.csv`) and expects them to be stored in the `data/` directory. I changed the `ID` column in `athlete_events.csv` to `Athlete_ID` for readability. I have provided the file `data/explanation.md` to provide a summary of the fields contained in each file.

#### Limitations
The data provided only gives the athletes age in years at the time of their olympic event, it does not include their date of birth. This lack of detail means I cannot accurately determine how exactly long after an Olympics someone was born and cannot distinguish between two athletes born 1 month apart rather than 23.

#### Data Used
In `DataCleaner.py` I have defined some methods for preparing the data for analysis. I chose to exclude the following from `athlete_events.csv`:
- Any athletes without age data.
- Data before 1948 due to disruption due to world wars.
- Large team sports as I want to investigate individual efforts. (Baseball, Tug-Of-War, Handball, Basketball, Ice Hockey, Hockey, Football & Water Polo)
- Sports which appeared in less than 5 games (since 1948).
This leaves 200,969 valid entries.

`athlete_events.csv` contains an entry for each event an athlete competed in. Using this as the raw data is not ideal as it would skew the data in favour of athletes who can compete in many events (swimmers). My method `DataCleaner.unique_athlete` extracts all the unique athletes and aggregates their data. For athletes who competed in multiple years I used their median age across all their events, although other statistics have merits. There are 91,562 I have written this aggregated data to `data/unique_athletes.csv`.

I did not utilise the data from `noc_regions.csv`.

### Notebooks
1. **[Overview Analysis](Overview%20Analysis.ipynb)**. I look at the trends in the population of olympic athletes and medalists as a whole. Although fewer medalists have been born during Olympic years, and the year immediately after, this was not found to be statistically significant. The expected number of olympics attended was consistent across all the normalised ages.
2. **[Sex Analysis]("ex%20Analysis.ipynb)** I consider male and female athletes separately. Female athletes are overall a couple of years younger than male athletes, and for both groups medalists are typically a year older than the average competitor. For both groups fewer medalsits are born in Olympic years, but again this was not found to be statistically significant.
3. **[Sport Analysis](Sport%20Analysis.ipynb)** Finally, I grouped athletes by the sport they competed in. Here statistically significant results were found with gymnasts and cyclists not having uniformly distributed normalised ages.

### Techniques Used
The bulk of the work in this project is done preparing and manipulating the data for meaningful visualisation. I used *Pearson's Chi-Squared Test* to evaluate the likelihood of the normalised age distributions being uniform.

## Further Analysis
Some areas that I think would be interesting to investigate further:
 - *By country*. It is well establish for countries to introduce programs to improve their Olympic perform, just look at GB&NI before and after Bejing 2008, and it is likely

### Note
This my first *formal* data analysis project and my first project using jupyter notebooks so I have likely not followed many conventions whilst doing so. I have tried to keep my approach consistent throughout, but there are new techniques which I learnt later in the project and may not have changed earlier code to reflect.
