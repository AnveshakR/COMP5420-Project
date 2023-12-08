import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# Read the CSV file into a DataFrame
file_path = 'columns.csv'
df = pd.read_csv(file_path)

# Create a new class "Unknown" and encode it as 2
df['sport'] = df['sport'].map({'Football': 0, 'Soccer': 1, 'Unknown': 2})

# Define features and target
features = ['phase', 'predicted_emotion', 'predicted_emotion_2nd']
target = 'sport'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

dump(clf, 'DT_model.joblib')