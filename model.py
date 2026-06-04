import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def train_model(df, target_column):

    data = df.copy()

    # Fill missing values
    for col in data.columns:

        if data[col].dtype == "object":
            data[col] = data[col].fillna("Unknown")
        else:
            data[col] = data[col].fillna(data[col].mean())

    encoders = {}

    for col in data.columns:

        if data[col].dtype == "object":

            le = LabelEncoder()

            data[col] = le.fit_transform(
                data[col].astype(str)
            )

            encoders[col] = le

    X = data.drop(columns=[target_column])

    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    return (
        model,
        accuracy,
        report,
        matrix,
        feature_importance
    )
