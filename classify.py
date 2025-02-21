import json
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_DIR = Path("parsed_data")

X: list[list[float]] = []
y: list[str] = []

for class_dir in DATA_DIR.iterdir():
    class_name = class_dir.name

    for features_vector_persist_json_path in class_dir.iterdir():
        with open(features_vector_persist_json_path, "r") as f:
            x = json.load(f)
        X.append(x)
        y.append(class_name)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)

preds = linear_svc.predict(X_test)

print(classification_report(y_test, preds))
