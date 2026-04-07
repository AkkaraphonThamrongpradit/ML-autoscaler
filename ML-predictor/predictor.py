import time
from flask import Flask, Response
import numpy as np
import pandas as pd
from tensorflow import keras
import joblib
from load_data_pred import load_data
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
print(tf.__version__)

app = Flask(__name__)

# ===== ต้องตรง TRAIN =====
FEATURES = [
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max",
    "pps_rx",
    "msg_count",
    "cpu_diff",
    "pps_rx_diff",
    "cpu_acc",
    "pps_rx_acc",
    "pps_rx_ratio",
    "pps_rx_trend",
    "cpu_std",
    "pps_rx_std"
]

WINDOW = 100         # -100s
GAP_THRESHOLD = "5s"
# =========================

model = keras.models.load_model("cpu_prediction_tcn.keras", compile=False)
x_scaler = joblib.load("x_scaler.save")
y_scaler = joblib.load("y_scaler.save")

last_predict_time = 0
last_seen_data_ts = {}
latest_predictions = {}


# --------------------------------------------------
# preprocess เหมือนตอน train
# --------------------------------------------------
def preprocess(df_dep: pd.DataFrame):

    df_dep = df_dep.copy().sort_index()
    df_dep = df_dep[~df_dep.index.duplicated()]
    
    # interpolate metric
    # เพิ่ม msg_count เข้าไปในลิสต์ที่ต้องทำ interpolate ด้วย
    for f in ["cpu_avg", "cpu_max", "mem_avg", "mem_max", "msg_count"]: 
        if f in df_dep.columns:
            df_dep[f] = df_dep.groupby("deployment")[f].transform(
                lambda x: x.interpolate(method="time")
            )

    # traffic (ถ้า interpolate แล้วยังมี NaN ให้เติม 0)
    df_dep["pps_rx"] = df_dep["pps_rx"].fillna(0)
    if "msg_count" in df_dep.columns:
        df_dep["msg_count"] = df_dep["msg_count"].fillna(0)

    # ----- feature engineering -----
    df_dep["cpu_diff"] = df_dep.groupby("deployment")["cpu_avg"].diff()
    df_dep["cpu_acc"] = df_dep.groupby("deployment")["cpu_diff"].diff()

    df_dep["pps_rx_diff"] = df_dep.groupby("deployment")["pps_rx"].diff()
    df_dep["pps_rx_acc"] = df_dep.groupby("deployment")["pps_rx_diff"].diff()

    df_dep["pps_rx_ratio"] = df_dep["pps_rx"] / (df_dep.groupby("deployment")["pps_rx"].shift(1) + 1)

    df_dep["pps_rx_trend"] = (
        df_dep.groupby("deployment")["pps_rx"]
        .rolling(12, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df_dep["cpu_std"] = (
        df_dep.groupby("deployment")["cpu_avg"]
        .rolling(12, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    df_dep["pps_rx_std"] = (
        df_dep.groupby("deployment")["pps_rx"]
        .rolling(12, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    df_dep["cpu_diff"] = df_dep["cpu_diff"].interpolate(method="time")
    df_dep["pps_rx_diff"] = df_dep["pps_rx_diff"].interpolate(method="time")
    df_dep["cpu_acc"] =  df_dep["cpu_acc"].interpolate(method="time")
    df_dep["pps_rx_acc"] = df_dep["pps_rx_acc"].interpolate(method="time")
    
    df_dep["cpu_diff"] = df_dep["cpu_diff"].fillna(0)
    df_dep["pps_rx_diff"] = df_dep["pps_rx_diff"].fillna(0)
    df_dep["cpu_acc"] =  df_dep["cpu_acc"].fillna(0)
    df_dep["pps_rx_acc"] = df_dep["pps_rx_acc"].fillna(0)

    df_dep["pps_rx_ratio"] = df_dep["pps_rx_ratio"].replace([np.inf, -np.inf], np.nan)
    df_dep["pps_rx_ratio"] = df_dep["pps_rx_ratio"].fillna(1)

    df_dep["pps_rx_trend"] = df_dep["pps_rx_trend"].bfill()
    df_dep["cpu_std"] = df_dep["cpu_std"].bfill()
    df_dep["pps_rx_std"] = df_dep["pps_rx_std"].bfill()

    for f in FEATURES:
        if f not in df_dep.columns:
            df_dep[f] = 0

    return df_dep


# --------------------------------------------------
# เช็คว่าข้อมูลต่อเนื่องไหม
# --------------------------------------------------
def is_continuous(df_dep):

    if len(df_dep) < WINDOW:
        return False

    gaps = df_dep.index.to_series().diff() > pd.Timedelta(GAP_THRESHOLD)

    # ถ้ามี gap ใน WINDOW ล่าสุด → ใช้ไม่ได้
    return not gaps.iloc[-WINDOW:].any()


def predict_for_deployment(df_dep):

    df_dep = preprocess(df_dep)

    # ----- เช็คความต่อเนื่อง -----
    if not is_continuous(df_dep):
        raise ValueError("data not continuous")

    # เอา WINDOW ล่าสุด
    df_dep = df_dep.tail(WINDOW)

    df_dep = df_dep.dropna(subset=FEATURES)
    if len(df_dep) < WINDOW:
        raise ValueError("not enough data after cleaning")
    
    # ----- scale -----
    data = df_dep[FEATURES].values
    if np.isnan(data).any():
        raise ValueError("nan in features")
    scaled = x_scaler.transform(data)

    seq = scaled.reshape(1, WINDOW, len(FEATURES))

    # ----- predict -----
    pred = model(seq, training=False).numpy()[0][0]
    pred = float(pred)

    # model output = scaled cpu_peak
    cpu_pred = y_scaler.inverse_transform([[pred]])[0][0]

    cpu_pred = max(0, cpu_pred)

    return {
        "cpu_pred_peak": cpu_pred
    }


@app.route("/metrics")
def metrics():

    global last_predict_time, last_seen_data_ts, latest_predictions

    df = load_data()
    now = time.time()
    lines = []

    current_deployments = set(df["deployment"].unique())

    # ลบ cache ของ deployment ที่หายไป
    for dep in list(latest_predictions.keys()):
        if dep not in current_deployments:
            del latest_predictions[dep]

    for dep in list(last_seen_data_ts.keys()):
        if dep not in current_deployments:
            del last_seen_data_ts[dep]

    # ถ้า scrape เร็วเกิน 5 วิ → return cache
    if now - last_predict_time < 5 and latest_predictions:
        for dep, preds in latest_predictions.items():
            for metric, value in preds.items():
                lines.append(f'{metric}{{deployment="{dep}"}} {value:.4f}')
        return Response("\n".join(lines) + "\n", mimetype="text/plain")

    updated = False  # flag ว่ามี deployment ไหนมีข้อมูลใหม่ไหม

    for dep, df_dep in df.groupby("deployment"):

        if df_dep.empty:
            continue

        current_max_ts = df_dep.index.max()

        # ถ้า deployment นี้ไม่มีข้อมูลใหม่ → ข้าม
        if dep in last_seen_data_ts and current_max_ts == last_seen_data_ts[dep]:
            continue

        # มีข้อมูลใหม่
        updated = True
        last_seen_data_ts[dep] = current_max_ts

        try:
            if len(df_dep) < 2:
                continue

            p = predict_for_deployment(df_dep)


            latest_predictions[dep] = {
                "pred_cpu_peak": p["cpu_pred_peak"],
                "pred_error": 0
            }

        except Exception as e:
            print("Predict error:", dep, e)
            latest_predictions[dep] = {
                "pred_error": 1
            }
            continue

    # ถ้าไม่มี deployment ไหนมีข้อมูลใหม่ → return cache
    if not updated:
        for dep, preds in latest_predictions.items():
            for metric, value in preds.items():
                lines.append(f'{metric}{{deployment="{dep}"}} {value:.4f}')
        return Response("\n".join(lines) + "\n", mimetype="text/plain")

    # มีการ predict เกิดขึ้น
    last_predict_time = now

    # แปลง cache เป็น prometheus format
    for dep, preds in latest_predictions.items():
        for metric, value in preds.items():
            lines.append(f'{metric}{{deployment="{dep}"}} {value:.4f}')

    return Response("\n".join(lines) + "\n", mimetype="text/plain")




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
