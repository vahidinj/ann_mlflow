# Importing the libraries
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import mlflow
import numpy as np

# Start an MLflow run
with mlflow.start_run():
    # Load the dataset
    df = pd.read_csv("Churn_Modelling.csv")
    df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

    # Identify numerical and categorical features
    num_feat = df.drop("Exited", axis=1).select_dtypes("int").columns
    cat_feat = df.drop("Exited", axis=1).select_dtypes("object").columns

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), list(cat_feat)),
            ("num", StandardScaler(), list(num_feat)),
        ],
        remainder="passthrough",
    )

    # Split the data
    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Apply preprocessing
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Convert to NumPy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Build the ANN
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
    ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
    ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    # Compile the ANN
    ann.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
        ],
    )

    # Train the model
    ann.fit(X_train, y_train, batch_size=32, epochs=150)

    # Make predictions
    y_pred = ann.predict(X_test)
    y_pred = y_pred > 0.5

    # Log metrics
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    # Log confusion matrix and classification report as artifacts
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save confusion matrix and classification report to files
    pd.DataFrame(cm).to_csv("confusion_matrix.csv", index=False)
    pd.DataFrame(report).to_csv("classification_report.csv", index=False)

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.csv")
    mlflow.log_artifact("classification_report.csv")

    # Log the TensorFlow model
    mlflow.tensorflow.log_model(ann, "model")

    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", acc)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
