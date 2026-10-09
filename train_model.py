
import json
import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset():
    iris = load_iris(as_frame=True)
    data = iris.frame.copy()

    data["species"] = data["target"].map(
        dict(enumerate(iris.target_names))
    )

    data = data.drop(columns=["target"])

    return data, list(iris.feature_names), list(iris.target_names)


def train_model():
    print("Loading Iris dataset...")

    data, features, class_names = load_dataset()
    data.to_csv("iris_dataset.csv", index=False)

    print("Dataset created successfully")
    print("Number of records:", len(data))

    X = data[features]
    y = data["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000, random_state=42
        ))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    matrix = confusion_matrix(
        y_test, predictions, labels=class_names
    )

    print("Model accuracy:", round(float(accuracy), 4))
    print("Class names:", class_names)
    print("Confusion matrix:")
    print(matrix)

    joblib.dump(model, "iris_classifier.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "class_names": class_names,
        "confusion_matrix": matrix.tolist()
    }

    with open("metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Model saved successfully")
    print("Metrics saved successfully")


if __name__ == "__main__":
    train_model()
