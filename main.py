import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def main():
    # Step 1: Load the dataset
    df = pd.read_csv('data/student_por.csv')

    # Step 2: Choose features and target
    features = ['studytime', 'G1', 'G2', 'absences']
    target = 'G3'  # Final grade

    print("Available Columns:\n", df.columns.tolist())

    # Step 3: Ensure required columns exist
    for col in features + [target]:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    X = df[features]
    y = df[target]

    # Step 4: Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 6: Evaluate model
    r2_score = model.score(X_test, y_test)
    print(f"\nModel R² Score: {r2_score:.2f}")

    # Step 7: Predict on new data (example input)
    sample_student = [[2, 12, 13, 4]]  # studytime, G1, G2, absences
    predicted_score = model.predict(sample_student)
    print(f"Predicted Final Exam Score (G3): {predicted_score[0]:.2f}")

    # Step 8: Visualization
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, color='blue')
    plt.xlabel('Actual G3 Scores')
    plt.ylabel('Predicted G3 Scores')
    plt.title('Actual vs Predicted G3 Scores')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
