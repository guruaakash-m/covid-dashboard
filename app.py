from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
df = pd.read_csv("covid_data.csv")

# Data Cleaning
df['contact_number'] = df['contact_number'].fillna(0)
df['confirmed_date'] = pd.to_datetime(df['confirmed_date'], errors='coerce')
df['released_date'] = pd.to_datetime(df['released_date'], errors='coerce')

# Feature Engineering
df['age'] = 2020 - df['birth_year']
df['recovery_days'] = (df['released_date'] - df['confirmed_date']).dt.days
df = df[df['recovery_days'] >= 0]

# Create static folder
if not os.path.exists("static"):
    os.makedirs("static")

# Generate Graphs
def generate_graphs():
    plt.figure()
    sns.countplot(x='sex', data=df)
    plt.savefig("static/gender.png")
    plt.close()

    plt.figure()
    sns.histplot(df['age'], bins=20)
    plt.savefig("static/age.png")
    plt.close()

    plt.figure()
    df['region'].value_counts().head(10).plot(kind='bar')
    plt.savefig("static/region.png")
    plt.close()

    plt.figure()
    sns.histplot(df['recovery_days'], bins=20)
    plt.savefig("static/recovery.png")
    plt.close()

generate_graphs()

# Dashboard Route
@app.route('/')
def home():
    selected_region = request.args.get('region')

    if selected_region:
        filtered_df = df[df['region'] == selected_region]
    else:
        filtered_df = df

    return render_template(
        'index.html',
        total_cases=len(filtered_df),
        avg_recovery=round(filtered_df['recovery_days'].mean(), 2),
        most_affected_region=filtered_df['region'].value_counts().idxmax(),
        avg_age=round(filtered_df['age'].mean(), 1),
        regions=df['region'].unique(),
        selected_region=selected_region
    )
# Prediction Page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        age = int(request.form['age'])
        contacts = int(request.form['contacts'])

        model = LinearRegression()
        ml_df = df[['age','contact_number','recovery_days']].dropna()

        X = ml_df[['age','contact_number']]
        y = ml_df['recovery_days']

        model.fit(X, y)

        result = model.predict([[age, contacts]])
        prediction = round(result[0], 2)

    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)