# Import necessary modules/libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# Goals of this analysis
# Find if there is a strong correlation between First Place MVP Votes and stats such as Games Played, Age, Minutes Played, Points, Rebounds, Assists, Steals, Blocks, Field Goal %, Three-Point %, and Free Throw %
# As well as to analyze the trends in the MVP award and player performance metrics over time, see if criteria has evolved 
# And finally make a prediction of next season's MVP and MVP Votes 

# Load the MVP Data into a DataFrame
path = r'C:\Users\iank3\OneDrive\Desktop\mvpdata.csv'
df = pd.read_csv(path)
print('The dataframe for the past NBA MVPs since the addition of the three point line is...',
      '-------------------------------------------'
      , df.head)


# Data Cleaning (Check for missing values in the DataFrame)
missing_values = df.isnull()
missing_counts=df.isnull().sum()

print("Missing values:\n", missing_values)
print("\nMissing value counts:\n", missing_counts)
# no missing values recorded

# Detecting possible outliers
# Visuals to identify outliers in a specific column
# sns.boxplot will create a boxplot showing if there are any outliers beyond the min./max.
# sns.boxplot(x=df['Games'])
# plt.show()
# Using quartiles
Q1_games = df['Games'].quantile(0.25)
Q3_games = df['Games'].quantile(0.75)
IQR_games = Q3_games - Q1_games
# Identify bounds with quatiles and then identify outliers
lower_bound_games = Q1_games - 1.5 * IQR_games
upper_bound_games = Q3_games + 1.5 * IQR_games
outliers_games = df[(df['Games'] < lower_bound_games) | (df['Games'] > upper_bound_games)]
print(outliers_games)
# Outliers in Games Played include Joel Embiid(66), Giannis Antetokounmpo (63), Lebron James(62), and Karl Malone(49)

# Now repeat those steps for each variable/column

# sns.boxplot(x=df['Age'])
# plt.show()
Q1_age = df['Age'].quantile(0.25)
Q3_age = df['Age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
lower_bound_age = Q1_age - 1.5 * IQR_age
upper_bound_age = Q3_age + 1.5 * IQR_age
outliers_age = df[(df['Age'] < lower_bound_age) | (df['Age'] > upper_bound_age)]
print(outliers_age)
# Outliers in Age include Karl Malone(35) and Michael Jordan(34)

# sns.boxplot(x=df['Minutes'])
# plt.show()
Q1_minutes = df['Minutes'].quantile(0.25)
Q3_minutes = df['Minutes'].quantile(0.75)
IQR_minutes = Q3_minutes - Q1_minutes
lower_bound_minutes = Q1_minutes - 1.5 * IQR_minutes
upper_bound_minutes = Q3_minutes + 1.5 * IQR_minutes
outliers_minutes = df[(df['Minutes'] < lower_bound_minutes) | (df['Minutes'] > upper_bound_minutes)]
print(outliers_minutes)
# No outstanding outliers in Minutes Per Game

# sns.boxplot(x=df['PTS'])
# plt.show()
Q1_points = df['PTS'].quantile(0.25)
Q3_points = df['PTS'].quantile(0.75)
IQR_points = Q3_points - Q1_points
lower_bound_points = Q1_points - 1.5 * IQR_points
upper_bound_points = Q3_points + 1.5 * IQR_points
outliers_points = df[(df['PTS'] < lower_bound_points) | (df['PTS'] > upper_bound_points)]
print(outliers_points)
# Outliers in Points Per Game include Steve Nash(15.5)

# sns.boxplot(x=df['RB'])
# plt.show()
Q1_rebounds = df['RB'].quantile(0.25)
Q3_rebounds = df['RB'].quantile(0.75)
IQR_rebounds = Q3_rebounds - Q1_rebounds
lower_bound_rebounds = Q1_rebounds - 1.5 * IQR_rebounds
upper_bound_rebounds = Q3_rebounds + 1.5 * IQR_rebounds
outliers_rebounds = df[(df['RB'] < lower_bound_rebounds) | (df['RB'] > upper_bound_rebounds)]
print(outliers_rebounds)
# No outstanding outliers in Rebounds Per Game

# sns.boxplot(x=df['AST'])
# plt.show()
Q1_assists = df['AST'].quantile(0.25)
Q3_assists = df['AST'].quantile(0.75)
IQR_assists = Q3_assists - Q1_assists
lower_bound_assists = Q1_assists - 1.5 * IQR_assists
upper_bound_assists = Q3_assists + 1.5 * IQR_assists
outliers_assists = df[(df['AST'] < lower_bound_assists) | (df['AST'] > upper_bound_assists)]
print(outliers_assists)
# No outstanding outliers in Assists Per Game

# sns.boxplot(x=df['STL'])
# plt.show()
Q1_steals = df['STL'].quantile(0.25)
Q3_steals = df['STL'].quantile(0.75)
IQR_steals = Q3_steals - Q1_steals
lower_bound_steals = Q1_steals - 1.5 * IQR_steals
upper_bound_steals = Q3_steals + 1.5 * IQR_steals
outliers_steals = df[(df['STL'] < lower_bound_steals) | (df['STL'] > upper_bound_steals)]
print(outliers_steals)
# Outliers in Steals Per Game include Michael Jordan(2.7), (3.2)

# sns.boxplot(x=df['BLK'])
# plt.show()
Q1_blocks = df['BLK'].quantile(0.25)
Q3_blocks = df['BLK'].quantile(0.75)
IQR_blocks = Q3_blocks - Q1_blocks
lower_bound_blocks = Q1_blocks - 1.5 * IQR_blocks
upper_bound_blocks = Q3_blocks + 1.5 * IQR_blocks
outliers_blocks = df[(df['BLK'] < lower_bound_blocks) | (df['BLK'] > upper_bound_blocks)]
print(outliers_blocks)
# Outliers in Blocks Per Game include David Robinson(3.2) and Hakeem Olajuwon(3.7)

# sns.boxplot(x=df['FG%'])
# plt.show()
Q1_fgpct = df['FG%'].quantile(0.25)
Q3_fgpct = df['FG%'].quantile(0.75)
IQR_fgpct = Q3_fgpct - Q1_fgpct
lower_bound_fgpct = Q1_fgpct - 1.5 * IQR_fgpct
upper_bound_fgpct = Q3_fgpct + 1.5 * IQR_fgpct
outliers_fgpct = df[(df['FG%'] < lower_bound_fgpct) | (df['FG%'] > upper_bound_fgpct)]
print(outliers_fgpct)
# Outliers in Field Goal % include Russell Westbrook(42.5%) and Allen Iverson(42.0%)

# sns.boxplot(x=df['3P%'])
# plt.show()
Q1_3ppct = df['3P%'].quantile(0.25)
Q3_3ppct = df['3P%'].quantile(0.75)
IQR_3ppct = Q3_3ppct - Q1_3ppct
lower_bound_3ppct = Q1_3ppct - 1.5 * IQR_3ppct
upper_bound_3ppct = Q3_3ppct + 1.5 * IQR_3ppct
outliers_3ppct = df[(df['3P%'] < lower_bound_3ppct) | (df['3P%'] > upper_bound_3ppct)]
print(outliers_3ppct)
# Outliers in Three-Point % include Shaquille O'Neal(0%), Karl Malone(0%) twice, and Moses Malone(0%) twice

# sns.boxplot(x=df['FT%'])
# plt.show()
Q1_ftpct = df['FT%'].quantile(0.25)
Q3_ftpct = df['FT%'].quantile(0.75)
IQR_ftpct = Q3_ftpct - Q1_ftpct
lower_bound_ftpct = Q1_ftpct - 1.5 * IQR_ftpct
upper_bound_ftpct = Q3_ftpct + 1.5 * IQR_ftpct
outliers_ftpct = df[(df['FT%'] < lower_bound_ftpct) | (df['FT%'] > upper_bound_ftpct)]
print(outliers_ftpct)
# Outliers in Free Throw % include Shaquille O'Neal(52.4%)

# sns.boxplot(x=df['FP Votes'])
# plt.show()
Q1_votes = df['FP Votes'].quantile(0.25)
Q3_votes = df['FP Votes'].quantile(0.75)
IQR_votes = Q3_votes - Q1_votes
lower_bound_votes = Q1_votes - 1.5 * IQR_votes
upper_bound_votes = Q3_votes + 1.5 * IQR_votes
outliers_votes = df[(df['FP Votes'] < lower_bound_votes) | (df['FP Votes'] > upper_bound_votes)]
print(outliers_votes)
# No outstanding outliers in First Place MVP Votes

# Handling and Removing the outliers before analysis
# Define a function to remove outliers below the lower bound
def remove_outliers(value, threshold):
    if (value) >= threshold:
        return value
    else:
        return np.nan  # Basically will return 'NaN' on the df, meaning its not a value
     
# Set the outlier thresholds
outlier_threshold_games = Q1_games - 1.5 * IQR_games
outlier_threshold_points = Q1_points - 1.5 * IQR_points
outlier_threshold_fgpct = Q1_fgpct - 1.5 * IQR_fgpct
outlier_threshold_ftpct = Q1_ftpct - 1.5 * IQR_ftpct
outlier_threshold_3ppct = Q1_3ppct - 1.5 * IQR_3ppct

# Apply the remove_outliers function to each column (outliers will be replaced with 'NaN')
df['Games'] = df['Games'].apply(lambda x: remove_outliers(x, outlier_threshold_games))
df['PTS'] = df['PTS'].apply(lambda x: remove_outliers(x, outlier_threshold_points))
df['FG%'] = df['FG%'].apply(lambda x: remove_outliers(x, outlier_threshold_fgpct))
df['FT%'] = df['FT%'].apply(lambda x: remove_outliers(x, outlier_threshold_ftpct))
df['3P%'] = df['3P%'].apply(lambda x: remove_outliers(x, outlier_threshold_3ppct))

# Now define a function to remove outliers above the upper bound
def remove_outliers(value, threshold):
    if (value) <= threshold:
        return value
    else:
        return np.nan  
    
# Set the outlier thresholds
outlier_threshold_age = Q3_age + 1.5 * IQR_age
outlier_threshold_steals = Q3_steals + 1.5 * IQR_steals
outlier_threshold_blocks = Q3_blocks + 1.5 * IQR_blocks

# Apply the remove_outliers function to each column (outliers will be replaced with 'NaN')
df['Age'] = df['Age'].apply(lambda x: remove_outliers(x, outlier_threshold_age))
df['STL'] = df['STL'].apply(lambda x: remove_outliers(x, outlier_threshold_steals))
df['BLK'] = df['BLK'].apply(lambda x: remove_outliers(x, outlier_threshold_blocks))

# Print the cleaned dataframe with no outlier values
print(df)
df_clean = df


# Summary statistics and structure knowledge
print(df.describe())
print(df.info())

# Correlation Analysis using the 'corr()' method
# Goal is to find strongest correlations of variables to FP MVP Votes
# I will create scatterplots for the strongest correlations
# A value between -1 and 1 will be given where...
    # A value close to 1 indicates a strong positive correlation
    # A value close to -1 indicates a strong negative correlation
    # A value close to 0 indicates a weak or no correlation

cc_votes_games= df['FP Votes'].corr(df['Games'])     #cc means correlation coefficient
print(cc_votes_games)
# correlation coefficient = -0.0847 

cc_votes_age= df['FP Votes'].corr(df['Age'])  
print(cc_votes_age)
# correlation coefficient = -0.3278 

cc_votes_minutes= df['FP Votes'].corr(df['Minutes'])  
print(cc_votes_minutes)
# correlation coefficient = -0.0240 

cc_votes_points= df['FP Votes'].corr(df['PTS'])  
print(cc_votes_points)
# correlation coefficient = 0.3071 

cc_votes_rebounds= df['FP Votes'].corr(df['RB'])  
print(cc_votes_rebounds)
# correlation coefficient = -0.1724 

cc_votes_assists= df['FP Votes'].corr(df['AST'])  
print(cc_votes_assists)
# correlation coefficient = -0.0752 

cc_votes_steals= df['FP Votes'].corr(df['STL'])  
print(cc_votes_steals)
# correlation coefficient = 0.1178 

cc_votes_blocks= df['FP Votes'].corr(df['BLK'])  
print(cc_votes_blocks)
# correlation coefficient = -0.0470 

cc_votes_fgpct= df['FP Votes'].corr(df['FG%'])  
print(cc_votes_fgpct)
# correlation coefficient = -0.0374

cc_votes_ftpct= df['FP Votes'].corr(df['FT%'])  
print(cc_votes_ftpct)
# correlation coefficient = -0.0108

cc_votes_3ppct= df['FP Votes'].corr(df['3P%'])  
print(cc_votes_3ppct)
# correlation coefficient = 0.3631

# The two variables with the strongest positive correlations are 'Points Per Game' and '3-Points %'

# Visualizations of Data (Different Plots)

# Creating scatterplots with regression lines to visualize the relationships
# Scatterplot of Points Per Game
sns.scatterplot(x='PTS', y='FP Votes', data=df, color='black')
sns.regplot(x='PTS', y='FP Votes', data=df, color='lightgreen', scatter=False) 
plt.title('Points Per Game to First Place MVP Votes Scatter Plot')
plt.show()

# # Scatterplot of 3-Point %
sns.scatterplot(x='3P%', y='FP Votes', data=df, color='black')
sns.regplot(x='3P%', y='FP Votes', data=df, color='purple', scatter=False) 
plt.title('3-Point Percentage to First Place MVP Votes Scatter Plot')
plt.show()

# Time Series Analysis

# Decompose the time series for components plot:
# Decomposition helps separate the time series into its components: trend, seasonality, and noise

# Decomposition requires DateTime Format, so the given seasons will be represented by the end year
# Split the 'Season' column into two separate columns
df[['Start Year', 'End Year']] = df['Season'].str.split('-', expand=True)

# Convert the 'EndYear' column to integers
df['End Year'] = df['End Year'].astype(int)

# Drop the 'StartYear' column
df.drop(columns=['Start Year'], inplace=True)

# Reading the dataset
df = pd.read_csv(path, 
                 parse_dates=True, 
                 index_col='End Year')

# Remove the non-numeric columns from this DataFrame to do decomposition
non_numeric_columns = ['Player', 'Team', 'era', 'Season']
decomp_df = df.drop(columns=non_numeric_columns)

# Multiplicative decomposition (if the seasonality and trend are not constant)
result=seasonal_decompose(decomp_df['FP Votes'], model='multiplicative')

# Plot Decomposed Components
# The decomposition object contains the trend, seasonal, and residual components
result.plot()
plt.show()

# MORE HELPFUL PLOTS AFTER DECOMPOSITION

# Reading the dataset again, this time using 'Season'
df = pd.read_csv(path, 
                 parse_dates=True, 
                 index_col='Season')

# Create Plot of Multiple Line Graphs Showing the trends of player stats through time
df.plot(subplots=True, figsize=(10, 8))
plt.title('Stats Over The Years', y=21.5)
plt.legend()
plt.show()
# As expected, 3P%, FG% and PTS per game have three of the highest lines in the current era
# This shows how certain stats have evolved and strived to become higher, representative of the FP Votes "given" in those areas


# Violin Plot to see how First Place MVP Votes varied among eras
# Use pd.get_dummies to redefine columns into numerical values that'll represent one of five eras
df = pd.get_dummies(df, columns=['era'])
df['era'] = df[['era_1', 'era_2', 'era_3', 'era_4', 'era_5']].idxmax(axis=1)
df['era'] = df['era'].map({'era_1': '1980-1990', 'era_2': '1990-2000', 'era_3': '2000-2010', 'era_4': '2010-2020', 'era_5': '2020-Present'})

#Set up violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='era', y='FP Votes', data=df, xticklabels=['1980-1990', '1990-2000', '2000-2010', '2010-2020', '2020-Present'], palette='deep',rotation=45)

# Set custom x-axis labels and show the plot
ax = plt.gca()
custom_xticklabels =['1980-1990', '1990-2000', '2000-2010', '2010-2020', '2020-Present']
ax.set_xticklabels(custom_xticklabels)
plt.xlabel('Era')
plt.ylabel('FP Votes')
plt.title("Relationship of Eras to FP MVP Votes")
plt.show()

# End with a Decision Tree Predictive Model

X = df[['3P%', 'Age', 'AST', 'BLK', 'FG%', 'FT%', 'Games', 'Minutes', 'PTS', 'RB', 'STL']]
y = df['FP Votes'] # This is the target variable

# Create A Decision Tree Regressor
reg = DecisionTreeRegressor()

# Fit the model to the training data (X contains stats, y contains the FP Votes)
reg.fit(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Import these modules this visualize the Decision Tree
import graphviz
from sklearn.tree import export_graphviz

# Visualize the decision tree
dot_data = export_graphviz(
    model,  
    out_file=None,
    feature_names=['3P%', 'Age', 'AST', 'BLK', 'FG%', 'FT%', 'Games', 'Minutes', 'PTS', 'RB', 'STL'],  
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # This will create a 'decision_tree.pdf' file in my folder
graph.view("decision_tree")  # This will open the PDF file for viewing

# After viewing the decision tree (located in the folder), PTS, used as the root node, and 3P% and FG% are used frequently
# Recall that 3P% and PTS had the strongest positive correlations to FP Votes

# Based on this and the MVP candidates for next season, I have first narrowed down my top three selections
# Joel Embiid, Jayson Tatum, and a dark-horse candidate, Shai Gilgeous-Alexander 

# 1. Joel Embiid, the star of the 76ers franchise, may take more responsibility this season due to the uncertainty surrounding James Harden
    # Having less of a three point shooter like Harden at his side, Embiid I believe will likely pick that up, leading to more FP MVP Votes
    # Embiid, already averaging 33.1 PPG last season, will likely continue his high usage and success, and an increase in his 3P% will only increase his MVP chances

# 2. Jayson Tatum I believe will continue taking the Celtics to a top seed in the regular season and far into the playoffs as he usually does, as he has done as such high levels in the past
    # The improvement of Kristaps Porzingis into the past role of Robert Williams III will lead to more 2nd chances for the Celtics, and most importantly Jayson Tatum in the midrange and perimeter shot areas
    # I think Jrue Holiday on the team will decrease his points production simply because of all the talent in that starting 5, and Tatum will take that up
    # Recalling the importance of 3P% and PPG, I believe Tatum is comfortable in improving these areas because new big competition in the East, as he has improved a lot in recent years

# 3. Shai Gilgeous-Alexander stunned the league with his stats last year, and he most definitely won't fall short of that this year
    # Shai's development will continue at en elite level, and I think he has a goal in mind to take the OKC Thunder to the playoffs this year, after falling short last year
    # That being said, similar to Porzingis and Tatum, I think Chet Holmgren this year will provide a lot more opportunities for Shai, hopefully forcing him to shoot at greater efficiency
    # Having improved statistically every year he has been with OKC, I think Shai will stay solid around his 31.4 PPG last year and improve just enough from his 34.5% from 3 to put him in the talks of MVP

# AND FINALLY, my prediction for the 2023-2024 NBA Season, is.... Jayson Tatum
# I believe of these three candidates Jayson Tatum is the brighter star with more upside, and he is a more well-rounded player and can dominate the most in the stats that will earn him the most FP MVP Votes
# Using the Decision Tree as reference, if Tatum has a 50/37-38 shooting season with around 30 PPG, which I believe he can, and taking into account the great competition putting up similar stats he'll win MVP with around 83-88 FP MVP Votes

# Final Pick: Jayson Tatum (83-88 FP MVP Votes)