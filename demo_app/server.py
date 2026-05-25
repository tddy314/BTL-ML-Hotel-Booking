import argparse
import json
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_DIR = PROJECT_DIR / "model"
DATA_PATH = PROJECT_DIR / "hotel_bookings.csv"

RAW_COLUMNS = [
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

NUMERIC_COLUMNS = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

MODEL_DEFINITIONS = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "path": MODEL_DIR / "logistic_regression" / "final_model.joblib",
        "metadata": MODEL_DIR / "logistic_regression" / "final_model_metadata.json",
    },
    "svm": {
        "label": "SVM (LinearSVC)",
        "path": MODEL_DIR / "svm" / "final_model.joblib",
        "metadata": MODEL_DIR / "svm" / "final_model_metadata.json",
    },
    "random_forest": {
        "label": "Random Forest",
        "path": MODEL_DIR / "random_forest" / "final_model.joblib",
        "metadata": MODEL_DIR / "random_forest" / "final_model_metadata.json",
    },
    "neural_network": {
        "label": "Neural Network",
        "path": MODEL_DIR / "neural_network" / "final_model.joblib",
        "metadata": MODEL_DIR / "neural_network" / "final_model_metadata.json",
    },
}


def load_demo_assets():
    original_df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    original_df["agent"] = original_df["agent"].fillna("Unknown").map(normalize_agent)
    original_df["country"] = original_df["country"].fillna("Unknown")
    original_df["children"] = original_df["children"].fillna(original_df["children"].median())
    bounds = {
        column: (
            float(original_df[column].quantile(0.01)),
            float(original_df[column].quantile(0.99)),
        )
        for column in ["adr", "adults", "lead_time", "days_in_waiting_list"]
    }
    sample = original_df.loc[0, RAW_COLUMNS].to_dict()
    sample = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in sample.items()
    }

    models = {}
    public_models = []
    for model_id, definition in MODEL_DEFINITIONS.items():
        model = joblib.load(definition["path"])
        with definition["metadata"].open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        metrics = metadata.get("test_metrics", {})
        models[model_id] = model
        public_models.append(
            {
                "id": model_id,
                "label": definition["label"],
                "name": metadata.get("final_model_name", model_id),
                "metrics": metrics,
                "has_probability": hasattr(model, "predict_proba"),
            }
        )
    return models, public_models, bounds, sample


def normalize_agent(value):
    if pd.isna(value) or str(value).strip() == "":
        return "Unknown"
    text = str(value).strip()
    if text.lower() == "unknown":
        return "Unknown"
    try:
        return str(float(text))
    except ValueError:
        return text


def prepare_input(rows, clip_bounds):
    input_df = pd.DataFrame(rows)
    missing_columns = [column for column in RAW_COLUMNS if column not in input_df.columns]
    if missing_columns:
        raise ValueError("Thiếu cột đầu vào: " + ", ".join(missing_columns))

    data = input_df.loc[:, RAW_COLUMNS].replace(r"^\s*$", np.nan, regex=True).copy()
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["agent"] = data["agent"].fillna("Unknown").map(normalize_agent)
    data["country"] = data["country"].fillna("Unknown")
    data["children"] = data["children"].fillna(0)

    data["total_nights"] = data["stays_in_weekend_nights"] + data["stays_in_week_nights"]
    data["total_guests"] = data["adults"] + data["children"] + data["babies"]
    data["has_children"] = (data["children"] > 0).astype(int)
    data["is_family"] = ((data["children"] + data["babies"]) > 0).astype(int)
    data["room_changed"] = (data["reserved_room_type"] != data["assigned_room_type"]).astype(int)
    data["has_agent"] = (data["agent"] != "Unknown").astype(int)
    data["has_previous_cancellation"] = (data["previous_cancellations"] > 0).astype(int)
    data["arrival_month_number"] = data["arrival_date_month"].map(MONTH_MAP)

    for column, (lower, upper) in clip_bounds.items():
        data[column] = data[column].clip(lower=lower, upper=upper)
    return data


class PredictionHandler(SimpleHTTPRequestHandler):
    models = {}
    public_models = []
    clip_bounds = {}
    sample = {}

    def send_json(self, payload, status=HTTPStatus.OK):
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/api/models":
            self.send_json({"models": self.public_models})
            return
        if self.path == "/api/template":
            self.send_json(
                {
                    "columns": RAW_COLUMNS,
                    "numeric_columns": NUMERIC_COLUMNS,
                    "sample": self.sample,
                }
            )
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_json({"error": "Không tìm thấy API."}, HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            model_id = payload.get("model")
            mode = payload.get("mode")
            rows = payload.get("rows", [])
            if model_id not in self.models:
                raise ValueError("Mô hình được chọn không hợp lệ.")
            if not isinstance(rows, list) or not rows:
                raise ValueError("Không có dữ liệu để dự đoán.")
            max_rows = 10 if mode == "manual" else 10000
            if len(rows) > max_rows:
                raise ValueError(f"Số dòng vượt quá giới hạn {max_rows}.")

            prepared = prepare_input(rows, self.clip_bounds)
            model = self.models[model_id]
            predicted = model.predict(prepared)
            has_probability = hasattr(model, "predict_proba")
            probabilities = model.predict_proba(prepared) if has_probability else None
            scores = model.decision_function(prepared) if not has_probability else None

            results = []
            for index, original in enumerate(rows):
                output = dict(original)
                output.update(
                    {
                        "row_number": index + 1,
                        "predicted_class": int(predicted[index]),
                        "prediction": "Hủy booking" if int(predicted[index]) == 1 else "Không hủy",
                    }
                )
                if has_probability:
                    output.update(
                        {
                            "prob_not_canceled_0": float(probabilities[index][0]),
                            "prob_canceled_1": float(probabilities[index][1]),
                            "percent_not_canceled_0": float(probabilities[index][0] * 100),
                            "percent_canceled_1": float(probabilities[index][1] * 100),
                        }
                    )
                else:
                    output["decision_score"] = float(scores[index])
                results.append(output)

            selected = next(model for model in self.public_models if model["id"] == model_id)
            self.send_json(
                {
                    "model_id": model_id,
                    "model_label": selected["label"],
                    "has_probability": has_probability,
                    "results": results,
                }
            )
        except (ValueError, KeyError, TypeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self.send_json({"error": f"Lỗi dự đoán: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main():
    parser = argparse.ArgumentParser(description="Hotel booking prediction demo server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    models, public_models, clip_bounds, sample = load_demo_assets()
    PredictionHandler.models = models
    PredictionHandler.public_models = public_models
    PredictionHandler.clip_bounds = clip_bounds
    PredictionHandler.sample = sample

    handler = partial(PredictionHandler, directory=str(APP_DIR))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Demo running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
