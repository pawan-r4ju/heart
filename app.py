from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('heart_disease_data.csv')

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Convert categorical columns to numerical using LabelEncoder
le = LabelEncoder()
X['sex'] = le.fit_transform(X['sex'])
# Apply this for other categorical columns as needed

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a simple model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = [request.form.get('age'), request.form.get('sex'),
                      request.form.get('cp'), request.form.get('trestbps'),
                      request.form.get('chol'), request.form.get('fbs'),
                      request.form.get('restecg'), request.form.get('thalach'),
                      request.form.get('exang'), request.form.get('oldpeak'),
                      request.form.get('slope'), request.form.get('ca'),
                      request.form.get('thal')]

        # Convert input data to a numpy array
        input_data_as_numpy_array = np.asarray(input_data, dtype=float)

        # Reshape the numpy array as we are predicting for only one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Make a prediction
        prediction = model.predict(input_data_reshaped)

        # Determine prediction result text
        result_text = 'The Person does not have Heart Disease' if prediction[0] == 0 else 'The Person has Heart Disease'

        return render_template('result.html', result=result_text)

if __name__ == '__main__':
    app.run(debug=True)
