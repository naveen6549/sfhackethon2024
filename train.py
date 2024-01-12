import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load historical case data from Salesforce
# Assume you have a CSV file with columns 'CaseText' and 'AssignedTo'
csv_file_path = 'historical_cases.csv'
df = pd.read_csv(csv_file_path)

# Preprocess data
# ... (cleaning, handling missing values, etc.)

# Vectorize text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['CaseText'])
y = df['AssignedTo']

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X, y)

while True:
    # Test with user input
    user_input = input("Enter a case description (type 'exit' to stop): ")

    if user_input.lower() == 'exit':
        break

    user_input_vec = vectorizer.transform([user_input])
    prediction = classifier.predict(user_input_vec)

    print(f'The predicted support team for "{user_input}" is: {prediction[0]}')

    # Get user feedback (correct support team)
    correct_support_team = input("Enter the correct support team (or type 'skip' to skip): ")

    if correct_support_team.lower() != 'skip':
        # Include the corrected answer in the training data
        new_data = {'CaseText': [user_input], 'AssignedTo': [correct_support_team]}
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df, new_df], ignore_index=True)

        # Retrain the model with the updated dataset
        X = vectorizer.fit_transform(df['CaseText'])
        y = df['AssignedTo']

        classifier.fit(X, y)

        # Save the updated training data to the CSV file
        df.to_csv(csv_file_path, index=False)

        # Save the updated model to a file (optional)
        joblib.dump(classifier, 'updated_model.joblib')
        print("Model updated and saved.")
