# Coding0to100
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('student_performance.csv')

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['nationality'] = le.fit_transform(df['nationality'])
df['parent_education'] = le.fit_transform(df['parent_education'])

# Split the dataset into features and target
X = df.drop('performance', axis=1)
y = df['performance']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rfc.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Identify key factors influencing academic performance
feature_importances = rfc.feature_importances_
feature_names = X.columns
print('Feature Importances:')
for feature_name, importance in zip(feature_names, feature_importances):
    print(f'{feature_name}: {importance:.2f}')
