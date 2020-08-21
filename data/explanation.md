# athlete_events.csv

| **Column Name** | **Type** | **Description** |
|------------|----------|-----------------|
| *ID* | Int |  |
| *Name* | Str | Name of athlete |
| *Sex* | Char | Sex of athlete ('M' or 'F') |
| *Age* | Int | Age of athlete in years |
| *Height* | Int | Height of athlete in cm (Some NaNs) |
| *Weight* | Int | Weight of athlete in kg (Some NaNs) |
| *Team* | Str | Name of team competed for within the event. Could be name of football team, boat, etc. |
| *NOC* | Str | 3 character code for nation represented |
| *Games* | Str | Name of games competed at. Formated "<Year> <Season>" |
| *Year* | Int | Year of games competed at |
| *Season* | Str | Season of games competed at ('Summer' or 'Winter') |
| *City* | Str | Host city of games |
| *Sport* | Str | Sport competed in |
| *Event* | Str | Event competed in |
| *Medal* | Str | Medal won ('Gold','Silver','Bronze',NaN) |

# noc_regions.csv
| **Column Name** | **Type** | **Description** |
|------------|----------|-----------------|
| *NOC* | Str | 3 character code for nation represented |
| *Region* | Str | Full name |
| *Notes* | Str | Alternative names used (Mostly NaNs) |
