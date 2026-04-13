import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, GlobalAveragePooling1D
from sklearn.preprocessing import RobustScaler


from load_from_tsdb import load_data


print("Loading data from TSDB...")
df = load_data()
df = df.sort_index()
df = df.sort_values(["deployment", df.index.name])

# ==================================================
# CONFIG
# ==================================================
FEATURES = [
    "cpu_avg",
    "msg_count",
    "mps",
    "mps_trend",
    "mps_std",     
    "cpu_std",
    "pps_rx",
    "pps_rx_trend"
]

WINDOW = 200         # -200s
PRED_STEP = 50        # ทำนาย +50s
N_FEATURE = len(FEATURES)

GAP_THRESHOLD = "5s"   
# ==================================================


# --------------------------------------------------
# 1) Handle missing values 
# --------------------------------------------------

# metric → interpolate
for f in ["cpu_avg", "cpu_max", "mem_avg", "mem_max"]:
    df[f] = df.groupby("deployment")[f].transform(
        lambda x: x.interpolate(method="time")
    )

# traffic → ถ้าไม่มี = 0
df["pps_rx"] = df["pps_rx"].fillna(0)

# ถ้า msg_count เป็นยอดรวมสะสมหรือยอดปัจจุบันที่หายไปบางช่วง (Interpolate)
df["msg_count"] = df.groupby("deployment")["msg_count"].transform(
    lambda x: x.interpolate(method="time")
)

# หรือถ้าหายไปเลยให้ถือว่าเป็น 0 (คล้าย pps_rx)
df["msg_count"] = df["msg_count"].fillna(0)
# replicas ไม่ใช้ train แต่เก็บไว้

df["mps"] = df.groupby("deployment")["mps"].transform(
    lambda x: x.interpolate(method="time")
)

df["mps"] = df["mps"].fillna(0)

df["pps_rx_trend"] = (
    df.groupby("deployment")["pps_rx"]
    .rolling(10, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

df["cpu_std"] = (
    df.groupby("deployment")["cpu_avg"]
    .rolling(10, min_periods=1)
    .std()
    .reset_index(level=0, drop=True)
)
# trend ของ mps
df["mps_trend"] = (
    df.groupby("deployment")["mps"]
    .rolling(10, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# volatility ของ mps
df["mps_std"] = (
    df.groupby("deployment")["mps"]
    .rolling(10, min_periods=1)
    .std()
    .reset_index(level=0, drop=True)
)

df["mps_trend"] = df["mps_trend"].bfill()
df["mps_std"] = df["mps_std"].fillna(0)
df["pps_rx_trend"] = df["pps_rx_trend"].bfill()
df["cpu_std"] = df["cpu_std"].fillna(0)

df = df.dropna(subset=FEATURES + ["cpu_max"])

# ==================================================
# FEATURE CORRELATION ANALYSIS
# ==================================================

import seaborn as sns
import matplotlib.pyplot as plt

analysis_cols = FEATURES + ["cpu_max"]

corr_matrix = df[analysis_cols].corr(method="pearson")

print("\n===== CORRELATION MATRIX =====")
print(corr_matrix)

plt.figure(figsize=(12,8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")
plt.show()

print("\n===== PER DEPLOYMENT CORRELATION =====")

for dep in df["deployment"].unique():

    df_dep = df[df["deployment"] == dep]

    corr = df_dep[analysis_cols].corr()

    print("\nDEPLOYMENT:", dep)
    print(corr["cpu_max"].drop("cpu_max").sort_values(ascending=False))

# ==================================================
# FEATURE → TARGET CORRELATION
# ==================================================

target_corr = corr_matrix["cpu_max"].drop("cpu_max")

target_corr = target_corr.sort_values(ascending=False)

print("\n===== CORRELATION WITH cpu_max =====")
print(target_corr)

target_corr.plot(
    kind="bar",
    figsize=(10,5),
    title="Feature Importance vs cpu_max"
)

plt.ylabel("Correlation")
plt.show()

# ==================================================
# HIGH CORRELATION FEATURE DETECTION
# ==================================================

print("\n===== HIGHLY CORRELATED FEATURES (>0.9) =====")

for i in range(len(FEATURES)):
    for j in range(i+1, len(FEATURES)):

        f1 = FEATURES[i]
        f2 = FEATURES[j]

        corr = corr_matrix.loc[f1, f2]

        if abs(corr) > 0.9:
            print(f1, "<->", f2, "=", corr)

# --------------------------------------------------
# 3) Model
# --------------------------------------------------
model = Sequential([
    keras.Input(shape=(WINDOW, N_FEATURE)),

    Conv1D(64, 3, padding="causal", dilation_rate=1, activation="relu"),
    Conv1D(64, 3, padding="causal", dilation_rate=2, activation="relu"),
    Conv1D(64, 3, padding="causal", dilation_rate=4, activation="relu"),

    Conv1D(32, 3, padding="causal", dilation_rate=8, activation="relu"),

    GlobalAveragePooling1D(),

    Dense(64, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss=tf.keras.losses.Huber(delta=0.1),
    metrics=["mae"]
)

# --------------------------------------------------
# 4) function แยก segment ต่อเนื่อง
# --------------------------------------------------
def split_continuous_segments(df_dep):

    df_dep = df_dep.copy()

    gaps = df_dep.index.to_series().diff() > pd.Timedelta(GAP_THRESHOLD)

    segs = []
    start = 0

    for i in range(1, len(df_dep)):
        if gaps.iloc[i]:
            segs.append(df_dep.iloc[start:i])
            start = i

    segs.append(df_dep.iloc[start:])

    return segs

# --------------------------------------------------
# 5) Build GLOBAL dataset จากทุก deployment
# --------------------------------------------------

deployments = df["deployment"].unique()

X_train_global = []
y_train_global = []

X_test_global = []
y_test_global = []

for dep, df_dep in df.groupby("deployment"):

    print("\nCollecting data from:", dep)
    print("Data points:", len(df_dep))
    print("Time range:", df_dep.index.min(), "to", df_dep.index.max())
    # แยก train/test ตามเวลา
    split_time = df_dep.index.to_series().quantile(0.8)

    train_df = df_dep[df_dep.index <= split_time]
    test_df  = df_dep[df_dep.index > split_time]

    train_segments = split_continuous_segments(train_df)
    test_segments  = split_continuous_segments(test_df)

    for seg in train_segments:

        if len(seg) < WINDOW + PRED_STEP:
            continue

        data = seg[FEATURES].values

        for i in range(len(data) - WINDOW - PRED_STEP + 1):

            x = data[i : i + WINDOW]

            future_cpu = seg["cpu_max"].values[i + WINDOW : i + WINDOW + PRED_STEP]

            y = np.max(future_cpu)

            X_train_global.append(x)
            y_train_global.append(y)
    for seg in test_segments:

        if len(seg) < WINDOW + PRED_STEP:
            continue

        data = seg[FEATURES].values

        for i in range(len(data) - WINDOW - PRED_STEP + 1):

            x = data[i : i + WINDOW]

            future_cpu = seg["cpu_max"].values[i + WINDOW : i + WINDOW + PRED_STEP]

            y = np.max(future_cpu)

            X_test_global.append(x)
            y_test_global.append(y)

# --------------------------------------------
# แปลงเป็น numpy array ครั้งเดียว
# --------------------------------------------
X_train_raw = np.asarray(X_train_global, dtype=np.float32)
y_train_raw = np.asarray(y_train_global, dtype=np.float32)

X_test_raw = np.asarray(X_test_global, dtype=np.float32)
if len(X_test_raw) == 0:
    raise ValueError("Test dataset empty after split")
y_test_raw = np.asarray(y_test_global, dtype=np.float32)

# --------------------------------------------------
# scale features (fit only on train)
# --------------------------------------------------

x_scaler = RobustScaler()
x_scaler.fit(X_train_raw.reshape(-1, N_FEATURE))

X_train = x_scaler.transform(
    X_train_raw.reshape(-1, N_FEATURE)
).reshape(-1, WINDOW, N_FEATURE)

X_test = x_scaler.transform(
    X_test_raw.reshape(-1, N_FEATURE)
).reshape(-1, WINDOW, N_FEATURE)

print("\nGLOBAL dataset shape:")
print("X_train:", X_train_raw.shape)
print("y_train:", y_train_raw.shape)
print("X_test:", X_test_raw.shape)
print("y_test:", y_test_raw.shape)

if len(X_train_raw) < 100:
    raise ValueError("Train dataset too small")

y_scaler = RobustScaler()
y_scaler.fit(y_train_raw.reshape(-1,1))

y_train = y_scaler.transform(y_train_raw.reshape(-1,1)).flatten()
y_test = y_scaler.transform(y_test_raw.reshape(-1,1)).flatten()

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# --------------------------------------------------
# 6) Train GLOBAL model
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=70,
    batch_size=64,
    shuffle=False,
    callbacks=[
        keras.callbacks.EarlyStopping(
            patience=8,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5
        )
    ]
)

# --------------------------------------------------
# training history
# --------------------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.legend(["train","validation"])
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()

loss, mae = model.evaluate(X_test, y_test)

# --------------------------------------------------
# prediction vs true
# --------------------------------------------------

y_pred = model.predict(X_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.3)

plt.xlabel("True CPU")
plt.ylabel("Predicted CPU")
plt.title("Prediction vs True CPU")

plt.show()

print("\nGLOBAL MAE:", mae)

# --------------------------------------------------
# 6) Save
# --------------------------------------------------
model.save("cpu_prediction_tcn.keras")
joblib.dump(x_scaler, "x_scaler.save")
joblib.dump(y_scaler, "y_scaler.save")

print("DONE → model train")
