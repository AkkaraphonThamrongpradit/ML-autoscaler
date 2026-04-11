import time
import math
import requests
import numpy as np
from kubernetes import client, config
import pandas as pd

# =========================
# CONFIG
# =========================

PROMETHEUS_URL = "http://prometheus:9090"

QUERY_CPU_PRED = "cpu_pred_peak"
QUERY_CPU_ACTUAL = 'max by (owner_name) (rate(container_cpu_usage_seconds_total{container!=""}[1m])* on(pod, namespace) group_left(owner_name)kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"ems-.*"}) * 1000'
QUERY_CPU_SD = 'stddev_over_time(max by (owner_name) (rate(container_cpu_usage_seconds_total{container!=""}[1m])* on(pod, namespace) group_left(owner_name)kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"ems-.*"}* 1000)[3m:10s])'
QUERY_LATENCY = "latency"

NAMESPACE = "edge-apps"


MIN_REPLICAS = 1
MAX_REPLICAS = 7
MAX_SCALE_STEP = 3

SCALE_UP_COOLDOWN = 15
SCALE_DOWN_COOLDOWN = 60
LOOP_INTERVAL = 5

K_HYSTERESIS = 1.5   # k คงที่
CPU_REQUEST_PER_POD = 350   # millicore

HISTORY_LENGTH = 10  # ใช้เก็บ CPU prediction history เพื่อ std

SLO_LATENCY = 0.5  # latency/item 

# =========================
# KUBERNETES CLIENT
# =========================

config.load_incluster_config()
apps_v1 = client.AppsV1Api()

# =========================
# STATE STORAGE
# =========================

state = {}

def get_state(dep):
    if dep not in state:
        state[dep] = {
            "last_pred": None,
            "last_scale_up": 0,
            "last_scale_down": 0,
            "cpu_history": []
        }
    return state[dep]

# =========================
# PROMETHEUS QUERY
# =========================

def query_prometheus(query):
    url = f"{PROMETHEUS_URL}/api/v1/query"
    try:
        r = requests.get(url, params={"query": query}, timeout=3)
        r.raise_for_status() # เพิ่มการเช็ค status code เพื่อความปลอดภัย
        data = r.json()
        results = {}

        # กำหนด label names ที่เราสนใจ
        target_labels = ["deployment", "pod", "owner_name"]

        for item in data["data"]["result"]:
            metric = item["metric"]
            value = float(item["value"][1])
            # วนลูปเช็คว่ามี label ที่ต้องการอยู่ใน metric หรือไม่
            for label in target_labels:
                if label in metric:
                    # ใช้ค่าของ label นั้นเป็น key ใน results
                    results[metric[label]] = value
                    # หากเจออันใดอันหนึ่งแล้วให้หยุดเช็ค label อื่นใน item นี้
                    break 
                    
        return results
    except Exception as e:
        print(f"Prometheus query error: {e}")
        return {}

def query_predicted_cpu():
    return query_prometheus(QUERY_CPU_PRED)

def query_actual_cpu():
    return query_prometheus(QUERY_CPU_ACTUAL)

def query_cpu_sd():
    return query_prometheus(QUERY_CPU_SD)

def query_latency():
    return query_prometheus(QUERY_LATENCY)

# =========================
# GET CURRENT REPLICAS
# =========================

def get_current_replicas(dep):
    try:
        deploy = apps_v1.read_namespaced_deployment(name=dep, namespace=NAMESPACE)
        return deploy.spec.replicas
    except Exception as e:
        print("get replicas error:", e)
        return None

# =========================
# SCALE DEPLOYMENT
# =========================

def scale_deployment(deployment, new_replicas):
    new_replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, int(new_replicas)))
    current = get_current_replicas(deployment)
    if new_replicas != current:
        apps_v1.patch_namespaced_deployment(
            name=deployment,
            namespace=NAMESPACE,
            body={"spec": {"replicas": new_replicas}}
        )
        print(f"[{deployment}] Scaled from {current} → {new_replicas}")
    else:
        print(f"[{deployment}] No scaling needed. Current replicas: {current}")

# =========================
# AUTOSCALER LOOP
# =========================

while True:
    try:
        # 1. Fetch Data
        df = pd.DataFrame({
            'cpu_pred': pd.Series(query_predicted_cpu()),
            'cpu_actual': pd.Series(query_actual_cpu()),
            'cpu_sd': pd.Series(query_cpu_sd()),
            'latency': pd.Series(query_latency())
        }).fillna(0)

        if df.empty:
            print("No data from Prometheus. Waiting...")
            time.sleep(LOOP_INTERVAL)
            continue

        now = time.time()

        for dep, row in df.iterrows():
            # ดึงค่าพื้นฐาน
            cpu_pred = row['cpu_pred']
            cpu_actual = row['cpu_actual']
            cpu_sd = row['cpu_sd']
            avg_latency = row['latency']
            
            s = get_state(dep)
            current_replicas = get_current_replicas(dep)
            if current_replicas is None: continue

            # 2. Compute Thresholds (Safety: ป้องกัน UT เป็น 0)
            ut_factor = (SLO_LATENCY / avg_latency) if avg_latency > 0 else 1.0
            UT = max(1.0, CPU_REQUEST_PER_POD * ut_factor) # อย่างน้อยต้องเป็น 1 เพื่อไม่ให้หาร 0
            LT = max(0, UT - K_HYSTERESIS * cpu_sd)

            target_cpu = max(cpu_pred, cpu_actual)
            
            print(f"[{dep}] CPU: {target_cpu:.2f}, UT: {UT:.2f}, LT: {LT:.2f}, Replicas: {current_replicas}")

            # 3. Scaling Decision with Cooldown
            new_replicas = current_replicas

            if target_cpu > UT:
                # --- CASE: SCALE UP ---
                if now - s["last_scale_up"] > SCALE_UP_COOLDOWN:
                    factor = target_cpu / UT
                    # ปรับเพิ่มแบบคำนวณจาก Ratio แต่ไม่เกิน MAX_SCALE_STEP
                    ideal_replicas = math.ceil(current_replicas * factor)
                    new_replicas = min(current_replicas + MAX_SCALE_STEP, ideal_replicas)
                    
                    scale_deployment(dep, new_replicas)
                    s["last_scale_up"] = now
                else:
                    print(f"[{dep}] Scale up blocked by cooldown.")

            elif target_cpu < LT:
                # --- CASE: SCALE DOWN ---
                if now - s["last_scale_down"] > SCALE_DOWN_COOLDOWN:
                    # ลดลงทีละ 1 ตามความเหมาะสมของระบบ Edge
                    new_replicas = current_replicas - 1
                    
                    scale_deployment(dep, new_replicas)
                    s["last_scale_down"] = now
                else:
                    print(f"[{dep}] Scale down blocked by cooldown.")
            
            else:
                print(f"[{dep}] Stable - no action.")

        print("-" * 40)
        time.sleep(LOOP_INTERVAL)

    except Exception as e:
        print(f"Main loop error: {e}")
        time.sleep(LOOP_INTERVAL)