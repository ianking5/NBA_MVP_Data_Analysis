# NBA_MVP_Data_Analysis
NBA MVP Data Analysis using Python

Over the summer I knew I had to and wanted to get started in the world of data and learning the Python language became a goal of mine. To apply the programming skills I learned, I wanted to do a data analysis on a dataset of my choice, utlizing the packages Python has to offer. As a beginner, I wanted to choose a project that was interesting and appealing to me, so as a basketball fan, I chose to analyze the statistics of all the past NBA MVP winners since 1980, when the three-point line was first introduced. I obtained my dataset from basketball-reference.com, which is the leading online encyclopedia for NBA & WNBA history. This specific dataset includes data of MVP winners including their name, season won, # of first place votes, points per game, assists per game, field goal %, 3-point %, and much more.

My goals of this analysis included the following:
- To find strong correlations between First Place MVP Votes and stats such as Games Played, Age, Minutes Played, Points, Rebounds, Assists, Steals, Blocks, Field Goal %, Three-Point %, and Free Throw %
- As well as to analyze the trends in the MVP award and player performance metrics over time, seeing if criteria has evolved 
- And finally make a prediction of next season's MVP and MVP Votes

A basic summary and overview of how I conducted this analysis was pretty standard. I gathered my dataset and explored it with necessary tools such as pandas, matplotlib, and seaborn. In doing so, I cleaned the data of outliers in each column (Games Played, Age, Minutes Played, Points, Rebounds, Assists, Steals, Blocks, Field Goal %, Three-Point %, and Free Throw %), located any null values, and visualized the summary statistics of the cleaned dataframe. After that I did a correlation analysis, where I basically found the patterns and relationships all the player statistics may have had to the amount of first place MVP Votes they received that season. From this I was able to identify the key variables that influenced voting the greatest, which I was able to later apply in my MVP prediction. Next I ran a Times Series analysis where I decomposed the data to indentify features such as trend and seasonality, and then also visualize how the metrics of each statistic has changed over the years relative to MVP voting. Finally, to help with my prediction, I made a Decision Tree using Regression that displayed the most frequent statistics, predictions of the # of First Place MVP Votes for intervals relative to each statistic, and the mean-squared error(MSE) for each node. From this whole process, here are my most important findings:

- 3P% and Points-Per Game (PPG) had the strongest positive correlations to MVP Voting
- Separating years into eras (1980-1990, 1990-2000, 2000-2010, 2010-2020, 2020-Present) made it easier to analyze temporal trends in time and compare different decades
- Seeing that the game has evolved into more of a 'shooting' league, where other stats such as blocks and rebounds have become less inportant in MVP Voting
- The eras 1980-90 and 2020-Present have the most dense data distributions, indicating statistics were similar and competition was greater
- The eras 1990-2000, 2000-10, and 2010-20 did not have as dense data distributions and were more strung out, indicating less similar statistics and a more obvious MVP in those seasons

Conclusion and Prediction:
Completing this project was an amazing, rewarding feeling and it felt like I finally got my foot through the door. Seeing the power that data holds and how much one can do with it was so cool, and I am very glad I chose a fun dataset I liked as my first analysis. I definitely want to continue down this path and take on more complex and challenging problems and datasets, really testing my capacity and what I can do best when working with numbers. This has also prompted me to probably concentrate in Quantitative Analysis and minor in Computer Science, which I am taking classes for right now. 

That being said, my prediciton for the 2023-2024 NBA MVP Winner is none other than... Jayson Tatum of the Boston Celtics.  Jayson Tatum I believe will continue taking the Celtics to a top seed in the regular season and far into the playoffs, as he has done as such high levels in the past. The improvement of Kristaps Porzingis into the past role of Robert Williams III will lead to more 2nd chances for the Celtics, and most importantly Jayson Tatum in the midrange and perimeter shot areas. I think Jrue Holiday on the team will decrease his own points production simply because of all the talent in that starting 5, and Tatum will take that up. Recalling the importance of 3P% and PPG, I believe Tatum is comfortable in improving these areas because new big competition in the East, as he has improved a lot in recent years. Using the Decision Tree as reference, if Tatum has a 50/37-38 shooting season with around 30 PPG, which I believe he can, and taking into account the great competition putting up similar stats, he'll win MVP with around 83-88 FP MVP Votes
