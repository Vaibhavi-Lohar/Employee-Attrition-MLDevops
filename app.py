
from pathlib import Path

import logging
import math

import mlflow.sklearn
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model"

DATASET_PATH = BASE_DIR / "employee_attrition.csv"

# Load the saved model and training dataset.
model = mlflow.sklearn.load_model(str(MODEL_PATH))
df = pd.read_csv(DATASET_PATH)

# Apply the same exclusions used during training.
df = df.drop(
    columns=[
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours",
    ]
)

FEATURES = list(model.feature_names_in_)

# Verify that the dataset contains every model feature.
missing = set(FEATURES) - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing features: {missing}")

# Calculate defaults using the training dataset only.
defaults = {}

for name in FEATURES:
    if pd.api.types.is_numeric_dtype(df[name]):
        defaults[name] = df[name].median()
    else:
        modes = df[name].mode(dropna=True)
        if modes.empty:
            raise ValueError(f"No valid category found for {name}.")
        defaults[name] = modes.iloc[0]

# Only these 10 features are entered by the user.
FORM_FIELDS = [
    "Age",
    "MonthlyIncome",
    "JobRole",
    "OverTime",
    "YearsAtCompany",
    "BusinessTravel",
    "JobSatisfaction",
    "DistanceFromHome",
    "JobLevel",
    "WorkLifeBalance",
]

# Build input controls from actual training data.
fields = []

for name in FORM_FIELDS:
    numeric = pd.api.types.is_numeric_dtype(df[name])

    field = {
        "name": name,
        "numeric": numeric,
        "options": [],
        "min": None,
        "max": None,
        "step": "1" if numeric and pd.api.types.is_integer_dtype(df[name]) else "any",
    }

    if numeric:
        field["min"] = float(df[name].min())
        field["max"] = float(df[name].max())
    else:
        field["options"] = sorted(
            df[name].dropna().unique().tolist()
        )

    fields.append(field)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Start with median/mode values for all 30 features.
            employee = defaults.copy()

            for field in fields:
                name = field["name"]
                value = request.form.get(name, "").strip()

                if not value:
                    raise ValueError(f"Please fill in {name}.")

                if field["numeric"]:
                    number = float(value)

                    if not math.isfinite(number):
                        raise ValueError(
                            f"{name} must be a finite number."
                        )

                    if not field["min"] <= number <= field["max"]:
                        raise ValueError(
                            f"{name} must be between "
                            f"{field['min']:g} and {field['max']:g}."
                        )

                    if field["step"] == "1" and not number.is_integer():
                        raise ValueError(f"{name} must be a whole number.")

                    if field["step"] == "1":
                        number = int(number)

                    employee[name] = number

                else:
                    if value not in field["options"]:
                        raise ValueError(
                            f"Please select a valid value for {name}."
                        )

                    employee[name] = value

            # Preserve the exact feature order expected by the model.
            input_df = pd.DataFrame(
                [[employee[name] for name in FEATURES]],
                columns=FEATURES,
            )

            result = int(model.predict(input_df)[0])

            prediction = (
                "Likely to leave"
                if result == 1
                else "Likely to stay"
            )

            app.logger.info("Demo prediction completed.")

        except (ValueError, TypeError) as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Prediction failed")
            error = (
                "Prediction failed. Check the terminal for details."
            )

    return render_template(
        "index.html",
        fields=fields,
        prediction=prediction,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
