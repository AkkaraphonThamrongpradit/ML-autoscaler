import time
import math
import requests
import numpy as np
from kubernetes import client, config
import pandas as pd

# =========================
# CONFIG
# =========================

PROMETHEUS_URL = "http://10.96.87.31:80"

#deployment
QUERY_CPU_PRED = "pred_cpu_peak"
QUERY_CPU_PRED_ERROR = "pred_error" 
#owner_name
QUERY_CPU_ACTUAL = 'max by (owner_name) (rate(container_cpu_usage_seconds_total{container!=""}[1m])* on(pod, namespace) group_left(owner_name)kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"ems-worker-.*"}) * 1000'
QUERY_CPU_SD = 'stddev_over_time(max by (owner_name) (rate(container_cpu_usage_seconds_total{container!=""}[1m])* on(pod, namespace) group_left(owner_name)kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"ems-worker-.*"}* 1000)[3m:10s])'
#pod_name
QUERY_LATENCY = 'sum by (pod_name) (ems_total_time_seconds{namespace="edge-apps"})'

NAMESPACE = "edge-apps"


MIN_REPLICAS = 1
MAX_REPLICAS = 7
MAX_SCALE_STEP = 3

SCALE_UP_COOLDOWN = 15
SCALE_DOWN_COOLDOWN = 60
LOOP_INTERVAL = 5

K_HYSTERESIS = 1.5   # k คงที่

UT_max = 500
UT_min = 350
L_max = 60

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
        target_labels = ["deployment", "pod_name", "owner_name"]

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

def extract_worker_deployment(owner_name):
    """
    ems-worker-edge-a-574875844c -> ems-worker-edge-a
    """
    if not owner_name:
        return None
    
    parts = owner_name.split("-")
    
    # ตัด hash (ReplicaSet suffix)
    if len(parts) >= 2:
        return "-".join(parts[:-1])
    
    return owner_name

def extract_worker_name_from_producer(pod_name):
    """
    ems-producer-edge-a-xxxxx -> ems-worker-edge-a
    """
    if not pod_name:
        return None

    parts = pod_name.split("-")

    # ems-producer-edge-a-xxxxx
    if len(parts) >= 4:
        zone = f"{parts[2]}-{parts[3]}"  # edge-a
        return f"ems-worker-{zone}"
    
    return None

def normalize_owner_metrics(raw_dict):
    result = {}

    for owner_name, value in raw_dict.items():
        dep = extract_worker_deployment(owner_name)
        if not dep:
            continue

        result[dep] = result.get(dep, 0) + value  # sum หรือ avg แล้วแต่ use case

    return result

def normalize_latency_metrics(raw_dict):
    temp = {}

    for pod_name, value in raw_dict.items():
        dep = extract_worker_name_from_producer(pod_name)
        if not dep:
            continue

        temp.setdefault(dep, []).append(value)

    # 🔥 ใช้ average แทน sum
    result = {dep: np.mean(values) for dep, values in temp.items()}
    return result


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
        cpu_pred = query_prometheus(QUERY_CPU_PRED)
        pred_error = query_prometheus(QUERY_CPU_PRED_ERROR)
        cpu_actual_raw = query_prometheus(QUERY_CPU_ACTUAL)
        cpu_sd_raw = query_prometheus(QUERY_CPU_SD)
        latency_raw = query_prometheus(QUERY_LATENCY)

        # 🔥 normalize
        cpu_actual = normalize_owner_metrics(cpu_actual_raw)
        cpu_sd = normalize_owner_metrics(cpu_sd_raw)
        latency = normalize_latency_metrics(latency_raw)

        # รวมเป็น DataFrame
        # รวม key ของ deployment ทั้งหมด
        all_deps = set(cpu_pred.keys()) | set(cpu_actual.keys())

        rows = []
        for dep in all_deps:
            rows.append({
                "deployment": dep,
                "cpu_pred": cpu_pred.get(dep, np.nan),
                "pred_error": pred_error.get(dep, 0),
                "cpu_actual": cpu_actual.get(dep, np.nan),
                "cpu_sd": cpu_sd.get(dep, 0),
                "latency": latency.get(dep, np.nan),
            })

        df = pd.DataFrame(rows).set_index("deployment")

        # 🔥 filter เฉพาะข้อมูลที่ใช้ได้
        df = df.dropna(subset=["cpu_pred", "cpu_actual"])

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
            pred_error = row['pred_error']

            if avg_latency is None or np.isnan(avg_latency):
                print(f"[{dep}] No latency → fallback")
                avg_latency = L_max  # worst case → force scale up tendency
            
            s = get_state(dep)
            current_replicas = get_current_replicas(dep)
            if current_replicas is None: continue

            # 2. Compute Thresholds (Safety: ป้องกัน UT เป็น 0)
            lat = min(avg_latency, L_max)  # clamp
            UT = UT_max - (lat / L_max) * (UT_max - UT_min)
            LT = max(UT_min * 0.5, UT - K_HYSTERESIS * cpu_sd)

            if pred_error == 1:
                target_cpu = cpu_actual
                mode = "FALLBACK"
            else:
                target_cpu = max(cpu_actual, cpu_pred)
                mode = "HYBRID"
            
            print(f"[DEBUG] {dep} | mode={mode} pred={cpu_pred:.2f} err={pred_error:.0f} actual={cpu_actual:.2f} lat={avg_latency:.3f}")

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