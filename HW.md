>  
> 
 
> **IE582 - HOMEWORK2** 
> 
>  
> **Evaluating Match Outcome Prediction and Market Efficiency Using Decision Trees** 
> 
> 
> **Fatih Berker Akyıldız** - 2019402162 
 
***2. CONTENT*** 
 
1.  TASK 1
 
2.  TASK 2
 
3.  TASK 3
 
4.  REPORTING

5.  REFERENCES & CITITIONS  
 

 
 TO ACCESS JUPYTER NOTEBOOK CLICK HERE -> [Fatih Berker Akyıldız's Jupyter Notebook](https://github.com/BU-IE-582/fall-24-fbaakyildiz/blob/main/hw2.ipynb)
 **1**
**Initialization** 
```python
# Remove rows where 'suspended' or 'stopped' is True
data = data[(data['suspended'] == False) & (data['stopped'] == False)]

# Identify columns with missing values and count them
missing_columns = data.isnull().sum()

# Filter and display columns that have more than 0 missing values
missing_columns = missing_columns[missing_columns > 0]
# Remove rows with missing values in 'current_state' column
data.dropna(subset=['current_state'], inplace=True)

# Fill missing values for other columns using the mean of 'fixture_id' groups
for column in data.columns:
    if column != 'current_state' and data[column].isnull().sum() > 0:
        data[column] = data.groupby('fixture_id')[column].transform(lambda x: x.fillna(x.mean()))

# Fill any remaining missing values with 0
data.fillna(0, inplace=True)```

We cleaned the dataset by removing unreliable rows where 'suspended' or 'stopped' was True and dropped rows with missing 'current_state' values. Missing data in other columns was imputed using the mean of each 'fixture_id' group, and any remaining missing values were filled with 0 to ensure data completeness and consistency for analysis.

**Task 1.1** 

```python
# Calculate probabilities for home win, draw, and away win
data['home_prob'] = 1 / data['1']
data['draw_chance'] = 1 / data['X']
data['away_prob'] = 1 / data['2']

# Normalize the probabilities
data['total_probability'] = data['home_prob'] + data['draw_chance'] + data['away_prob']
data['normalized_home_prob'] = data['home_prob'] / data['total_probability']
data['normalized_draw_chance'] = data['draw_chance'] / data['total_probability']
data['normalized_away_prob'] = data['away_prob'] / data['total_probability']

# Calculate the difference between home win and away win probabilities
data['home_away_diff'] = data['normalized_home_prob'] - data['normalized_away_prob']```

**Task 1.2** 
```python
# Create bin intervals between -1 and 1
bin_intervals = np.linspace(-1, 1, 21)
data['diff_bin'] = pd.cut(data['home_away_diff'], bins=bin_intervals)

# Analyze each bin for draw probability and total counts
bin_analysis = data.groupby('diff_bin').agg(
    draw_probability=('result', lambda x: (x == 'X').mean()),
    total_count=('result', 'size'),
    draw_count=('result', lambda x: (x == 'X').sum())
).reset_index()

# Display the result of the bin analysis
print(bin_analysis)```

**Task 1.3**
```python
# Separate first-half and second-half data
first_half_data = data[data['halftime'] == '1st-half']
second_half_data = data[data['halftime'] == '2nd-half']

# Analyze each bin for the first half
first_half_bin_analysis = first_half_data.groupby('diff_bin')['result'].apply(lambda x: (x == 'X').mean()).reset_index()
first_half_bin_analysis.columns = ['Bin', 'Draw_Probability']

# Analyze each bin for the second half
second_half_bin_analysis = second_half_data.groupby('diff_bin')['result'].apply(lambda x: (x == 'X').mean()).reset_index()
second_half_bin_analysis.columns = ['Bin', 'Draw_Probability']

# Plot draw probability for both halves using scatter plots
plt.figure(figsize=(12, 6))```

**Task 1.4**

It is apparent that a slight bias exists in the data caused by I believe in the frist half despite the all forecasting effort not all aspects of situation of players is known and it becomes more clear with observing the each player at the first half. During the first half, it can be observed that the probability of a draw is higher in situations where the likelihood of the home team winning is lower, compared to cases where the away team's win probability is lower. This pattern is likely due to the advantage the home team has when playing in their own stadium, which increases their chances of converting a potential loss into a draw. This advantage is also reflected in betting market dynamics, where the perceived probability of the home team turning a loss into a draw is higher due to the home advantage.

When comparing the first and second halves, the second half appears to exhibit a more consistent and structured pattern. The main reason for this is the gradual reduction in the amount of time remaining in the match. As time decreases, both the home and away teams have fewer opportunities to change the outcome, leading to a reduction in the probability of a draw. This is because, as the remaining time diminish, the chance of the leading team maintaining their advantage increases, making the final outcome more predictable. This effect results in the second half having a more stable and clear pattern compared to the first half.


**TASK 2**
```python
# Filter out matches with late goals
late_goal_games = data[(data['minute'] > 45) & (data['halftime'] == '2nd-half')]
matches_to_remove_late = pd.concat([late_goal_games['fixture_id']]).unique()
filtered_data = data[~data['fixture_id'].isin(matches_to_remove_late)]

# Count removed matches for late goals
removed_late_goal_count = len(late_goal_games)
print(f"Number of rows removed due to late goals: {removed_late_goal_count}")

# Filter out matches with early red cards
early_red_card_games = data[(data['Redcards - home'] > 0) & (data['minute'] <= 15) | (data['Redcards - away'] > 0) & (data['minute'] <= 15)]
matches_to_remove_red = pd.concat([early_red_card_games['fixture_id']]).unique()
filtered_data = filtered_data[~filtered_data['fixture_id'].isin(matches_to_remove_red)]

# Count removed matches for early red cards
removed_early_red_card_count = len(early_red_card_games)
print(f"Number of matches removed due to early red cards: {removed_early_red_card_count}")```

For speaking in the favor of the probability distribution graph the highest draw probabilities (≈0.6) occur around P(Home) - P(Away) ≈ 0, indicating that balanced matches are most likely to result in draws.
This reinforces the natural equilibrium in football when teams are evenly matched, as neither team can dominate or break the tie effectively.
The removal of late goals prevents artificially high probabilities caused by last-minute match-deciding events.
Similarly, removing early red cards avoids situations where the balance shifts too heavily in favor of one team early in the game. As a result, the central peak becomes sharper and more defined, focusing on the "pure" match dynamics.

First Half:
Low Draw Probabilities:
Draw probabilities are relatively low across all bins, peaking at around 0.12 in the central range (P(Home) - P(Away) ≈ 0).
This reflects the nature of the first half, where teams typically focus on understanding their opponent and forming strategies, rather than settling into a draw.
Extreme Values (P(Home) - P(Away) ≈ ±1):
Draw probabilities are nearly zero at extreme bins, aligning with expectations that a large imbalance in team performance early in the match rarely results in a tie.
The removal of early red cards ensures that the analysis focuses on "normal" gameplay scenarios, which is why the bins near the extremes show a minimal influence from disruptions.

Second Half:
Higher Draw Probabilities:
Draw probabilities increase significantly in the second half, peaking at approximately 0.25 in the central range (P(Home) - P(Away) ≈ 0).
As time becomes a limiting factor, teams often adopt more defensive strategies, leading to higher draw probabilities, particularly in balanced matches.
Extreme Values (P(Home) - P(Away) ≈ ±1):
Draw probabilities remain close to zero at the extremes, indicating that one-sided matches are unlikely to result in draws, even as the match progresses.
The removal of late goals refines the dataset, eliminating outliers where late-match events could have artificially increased or decreased draw probabilities. As a result, the second-half probabilities present a cleaner representation of natural gameplay tendencies.
The first half is exploratory, with lower draw probabilities across all ranges, reflecting teams' focus on strategy and testing their opponent.
The second half shows a significant increase in draw probabilities, particularly in balanced matches, as teams shift towards risk-averse strategies to preserve their standings.
Filtering disruptive events like late goals and early red cards provides a cleaner view of these tendencies, highlighting the core dynamics of football matches in their natural state.

**TASK 3**
```python
# Drop unnecessary columns and prepare for decision tree
label_encoder = LabelEncoder()
data['halftime'] = label_encoder.fit_transform(data['halftime'].astype(str))
data['result'] = label_encoder.fit_transform(data['result'].astype(str))
data['current_state'] = label_encoder.fit_transform(data['current_state'].astype(str))

columns_to_drop = ['result', 'current_time', 'half_start_datetime', 'match_start_datetime', 
                   'latest_bookmaker_update', 'suspended', 'stopped', 'ticking', 'final_score', 'name', 'P_diff_bin']
columns_to_drop = [col for col in columns_to_drop if col in data.columns]

X = data.drop(columns=columns_to_drop)
y = data['result']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42, max_depth=15, min_samples_split=20)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.show()```

```python
# Analyze misclassified samples
misclassified_indices = np.where(y_test != y_pred)[0]
misclassified_samples = X_test.iloc[misclassified_indices]
misclassified_samples['true_label'] = y_test.iloc[misclassified_indices].values
misclassified_samples['predicted_label'] = y_pred[misclassified_indices]

# Select relevant columns for time analysis
time_analysis = misclassified_samples[['minute', 'halftime', 'true_label', 'predicted_label']]

# Group by halftime, minute, and label combinations
error_summary = time_analysis.groupby(['halftime', 'minute', 'true_label', 'predicted_label']).size().reset_index(name='error_count')

# Print error summary
print("Summary of misclassifications:")
print(error_summary)```

```python
# Select relevant columns for time analysis
time_analysis = misclassified_samples[['minute', 'halftime', 'true_label', 'predicted_label']]

# Group by halftime, minute, and label combinations
error_summary = time_analysis.groupby(['halftime', 'minute', 'true_label', 'predicted_label']).size().reset_index(name='error_count')

# Print error summary
print("Summary of misclassifications:")
print(error_summary)

# Highlight key metrics from the analysis
most_common_errors = error_summary.sort_values(by='error_count', ascending=False).head(5)
print("Most common misclassifications:")
print(most_common_errors)

# Analyze the range of minutes where errors occur
minute_range = (time_analysis['minute'].min(), time_analysis['minute'].max())
print(f"Errors occur between minutes {minute_range[0]} and {minute_range[1]}.")

# Count errors per halftime
halftime_errors = time_analysis['halftime'].value_counts()
print("Error counts per halftime:")
print(halftime_errors)

# Identify specific minutes with highest errors
minute_error_summary = time_analysis['minute'].value_counts().head(10)
print("Top 10 minutes with highest errors:")
print(minute_error_summary)```

The decision tree achieved nearly 96% accuracy, but misclassifications highlight trends in specific game periods. Errors are most common in the early minutes (e.g., 3rd, 4th, and 6th minutes) due to limited early-game data, leading the model to rely on pre-game probabilities. The range of errors spans from the start to minute 56, with more errors in the first half (446 vs. 270 in the second half). Key minutes, such as the first 10, account for the highest errors, indicating that additional features or real-time data could improve model predictions during these periods.

```python
# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
})

# Sort feature importances
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print("Feature Importances:")
print(feature_importances)```

The feature importance analysis highlights P_diff (0.192) as the most critical predictor, reflecting its alignment with probability-driven match outcomes. Fixture_id (0.103) unexpectedly ranks high, potentially due to recurring patterns in specific match-ups. Away and home assists (0.065, 0.059) emphasize the role of team coordination in influencing results, underscoring soccer's collaborative nature. Meanwhile, features like minute and halftime hold negligible predictive power (~0.000), indicating their limited direct impact. Overall, the model effectively captures logical relationships while revealing biases in certain match patterns.


**CONCLUSION**

```python
y_prob = clf.predict_proba(X_test)
predicted_home_prob = y_prob[:, label_encoders['result'].transform(['1'])[0]]
predicted_away_prob = y_prob[:, label_encoders['result'].transform(['2'])[0]]
predicted_draw_prob = y_prob[:, label_encoders['result'].transform(['X'])[0]]

# Calculate the mean of actual and predicted probabilities
mean_home = data['P_home'].mean()
mean_draw = data['P_draw'].mean()
mean_away = data['P_away'].mean()

mean_pred_home = predicted_home_prob.mean()
mean_pred_draw = predicted_draw_prob.mean()
mean_pred_away = predicted_away_prob.mean()

# Calculate the error rates between actual and predicted probabilities
home_error = np.abs(mean_home - mean_pred_home)
draw_error = np.abs(mean_draw - mean_pred_draw)
away_error = np.abs(mean_away - mean_pred_away)

# Print the results
print(f"Actual Home Probability Mean: {mean_home:.4f}")
print(f"Predicted Home Probability Mean: {mean_pred_home:.4f}")
print(f"Home Error Rate: {home_error:.4f}\n")

print(f"Actual Draw Probability Mean: {mean_draw:.4f}")
print(f"Predicted Draw Probability Mean: {mean_pred_draw:.4f}")
print(f"Draw Error Rate: {draw_error:.4f}\n")

print(f"Actual Away Probability Mean: {mean_away:.4f}")
print(f"Predicted Away Probability Mean: {mean_pred_away:.4f}")
print(f"Away Error Rate: {away_error:.4f}")```

The comparison of actual and predicted probabilities shows minimal error rates for all outcomes, with the home win having an error of 0.0045, the draw at 0.0084, and the away win at 0.0129. These low error rates suggest that the decision tree model effectively predicts probabilities close to the actual averages. However, the slightly higher deviation in the away win probabilities might indicate inefficiencies in capturing specific match dynamics, possibly due to less impactful features or unmodeled randomness. Overall, the model aligns well with the actual data, reinforcing its reliability in predicting match outcomes.




**PART 5 REFERENCES & CITITIONS**
**CHATGPT 4o**
**All prompts that were given to GenAI tool to handle coding part of this assignment adittionally english grammar and vocublary of this assigment's report was corrected briefly by not GenAi but online tools:**
-Write Python code to load a dataset and perform comprehensive data cleaning, including handling missing values, removing irrelevant columns, and preparing the data for analysis. Ensure the dataset is ready for downstream modeling tasks.

-Calculate probabilities for home win, draw, and away win based on betting odds columns (1, X, 2). Normalize these probabilities and add them as new features in the dataset.

-Encode all categorical columns in the dataset into numerical values using appropriate encoding techniques like LabelEncoder, ensuring that no data integrity is lost.

-Drop columns that are irrelevant to prediction tasks or that might leak information about the target variable (result). Ensure only necessary features remain in the dataset.

-Filter the dataset to remove matches with late goals (goals after the 45th minute in the second half). Count and display the number of rows removed.

-Further filter the dataset by removing matches where a red card was issued in the first 15 minutes of the game. Display the number of rows removed and ensure the dataset is updated.

-Plot the normalized probabilities of outcomes (P_home, P_draw, P_away) against the difference in probabilities (P_home - P_away) to analyze the distribution of probabilities across matches.

-Group the data into bins based on the differences in normalized probabilities (P_home - P_away), calculate the average draw probability, and summarize the results in a clear table.

-Train a DecisionTreeClassifier to predict the match outcome (result) using the features prepared earlier. Configure the decision tree with custom parameters such as max_depth and min_samples_split to optimize for interpretability.

-Evaluate the trained decision tree model by calculating its accuracy on the test set. Display a confusion matrix to highlight the model's performance.

-Extract and rank the feature importances from the trained decision tree model. Present the results in a sorted table to interpret which features are most influential in predicting match outcomes.

-Analyze model misclassifications by identifying instances where the predicted results differ from the actual results. Summarize misclassification patterns in terms of halftime, minute, true labels, and predicted labels.

-Compare the mean predicted probabilities for home win, draw, and away win with their actual implied probabilities derived from betting odds. Calculate error rates for each outcome and evaluate the efficiency of the predictions.

-Provide insights into potential inefficiencies in the betting odds market by identifying cases where the predicted probabilities deviate significantly from implied probabilities.

-Visualize the trends in model misclassification by halftime and minute to identify specific time periods or scenarios where the model struggles to make accurate predictions.

Many revisements were made where I got angry to chatGPT sometimes. I'm not including these but here are the generalized version of me taking aid from chatGPT 4o model.

