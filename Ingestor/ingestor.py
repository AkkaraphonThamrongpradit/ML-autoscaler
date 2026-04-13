import os
import time
import requests
import psycopg2
from psycopg2 import OperationalError, InterfaceError
from psycopg2.extras import execute_values  # สำคัญมากสำหรับการทำ batch insert
from datetime import datetime, timezone

PROM_URL = os.getenv("PROM_URL", "http://prometheus.monitoring:9090")
INTERVAL = int(os.getenv("POLL_INTERVAL", "15"))

DB = None

# ---------- DB CONNECT ----------

def connect_db():
    while True:
        try:
            print(f"Connecting to DB... {os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}")
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST"),
                port=int(os.getenv("DB_PORT", 5432)),
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASS")
            )
            conn.autocommit = False
            print("DB CONNECTED!")
            return conn
        except Exception as e:
            print("DB CONNECT ERROR:", e)
            time.sleep(5)

def get_db():
    global DB
    if DB is None or DB.closed:
        DB = connect_db()
    return DB

# ---------- Prometheus Queries ----------

QUERIES = {
    "cpu_avg": 'avg by (owner_name) (rate(container_cpu_usage_seconds_total{container!=""}[1m]) * on(pod, namespace) group_left(owner_name) kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"ems-worker-.*"}) * 1000',
    "cpu_max": 'max by (owner_name) (rate(container_cpu_usage_seconds_total{container!=""}[1m]) * on(pod, namespace) group_left(owner_name) kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"ems-worker-.*"}) * 1000',
    "mem_avg": 'avg by (owner_name) (container_memory_usage_bytes{container!=""} * on(pod, namespace) group_left(owner_name) kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"ems-worker-.*"}) / 1048576',
    "mem_max": 'max by (owner_name) (container_memory_usage_bytes{container!=""} * on(pod, namespace) group_left(owner_name) kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"ems-worker-.*"}) / 1048576',
    "replicas": 'sum by (owner_name) (kube_pod_status_ready * on(pod, namespace) group_left(owner_name) kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"ems-worker-.*"})',
    "pps_rx": 'sum by (owner_name) (rate(container_network_receive_packets_total{interface="eth0", pod=~"ems-worker-.*"}[1m]) * on(pod, namespace) group_left(owner_name) kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"ems-worker-.*"})'
}

QUERY_MSG_COUNT = 'sum by (pod_name) (ems_message_count{namespace="edge-apps"})'
QUERY_MPS = 'sum by (pod_name) (pdc_realtime_mps{namespace="edge-apps"})'

# ---------- Logic Helpers ----------

def extract_worker_deployment(owner_name):
    """ems-worker-edge-a-574875844c -> ems-worker-edge-a"""
    if not owner_name: return None
    return owner_name.rsplit("-", 1)[0]

def extract_worker_name_from_producer(pod_name):
    """ems-producer-edge-a-7db86bbf54-jgrk5 -> ems-worker-edge-a"""
    if not pod_name: return None
    parts = pod_name.split("-")
    # สมมติโครงสร้างคือ ems-producer-edge-a-...
    if len(parts) >= 4:
        zone = f"{parts[2]}-{parts[3]}" # ได้ 'edge-a'
        return f"ems-worker-{zone}"
    return None

def extract_worker_name_from_pdc(pod_name):
    """pdc-edge-a-xxxx -> ems-worker-edge-a"""
    if not pod_name:
        return None
    parts = pod_name.split("-")
    # โครงสร้าง: pdc-edge-a-...
    if len(parts) >= 3:
        zone = f"{parts[1]}-{parts[2]}"  # edge-a
        return f"ems-worker-{zone}"
    return None

# ---------- Prometheus Collector ----------

def query_prom(q):
    try:
        r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": q}, timeout=10)
        r.raise_for_status()
        return r.json()["data"]["result"]
    except Exception as e:
        print(f"Query Error: {e}")
        return []

def collect_all():
    data = {}

    # 1. ดึงข้อมูลฝั่ง Worker
    for key, q in QUERIES.items():
        result = query_prom(q)
        for row in result:
            owner = row["metric"].get("owner_name")
            dep = extract_worker_deployment(owner)
            if not dep: continue
            
            val = float(row["value"][1])
            if dep not in data: data[dep] = {}
            data[dep][key] = val

    # 2. ดึงข้อมูลฝั่ง Producer และ Join เข้าแถวของ Worker
    msg_results = query_prom(QUERY_MSG_COUNT)
    for row in msg_results:
        pod_name = row["metric"].get("pod_name")
        target_dep = extract_worker_name_from_producer(pod_name)
        
        if target_dep:
            if target_dep not in data:
                data[target_dep] = {}
            data[target_dep]["msg_count"] = float(row["value"][1])

    # 3. ดึง MPS จาก PDC และ map เข้า worker
    mps_results = query_prom(QUERY_MPS)
    for row in mps_results:
        pod_name = row["metric"].get("pod_name")
        target_dep = extract_worker_name_from_pdc(pod_name)
        
        if target_dep:
            if target_dep not in data:
                data[target_dep] = {}
            data[target_dep]["mps"] = float(row["value"][1])   

    return data

# ---------- DB UPSERT (Batch Mode) ----------

def upsert_batch(ts, metrics_data):
    if not metrics_data: return
    
    conn = get_db()
    rows = []
    for dep, m in metrics_data.items():
        rows.append((
            ts, dep,
            m.get("cpu_avg"), m.get("cpu_max"),
            m.get("mem_avg"), m.get("mem_max"),
            m.get("pps_rx"), 
            m.get("msg_count"),
            m.get("mps"),
            int(float(m.get("replicas") or 1))
        ))

    sql = """
    INSERT INTO autoscale_features 
    (time, deployment, cpu_avg, cpu_max, mem_avg, mem_max, pps_rx, msg_count, mps, replicas)
    VALUES %s
    ON CONFLICT (time, deployment) DO UPDATE SET
    cpu_avg = EXCLUDED.cpu_avg,
    cpu_max = EXCLUDED.cpu_max,
    mem_avg = EXCLUDED.mem_avg,
    mem_max = EXCLUDED.mem_max,
    pps_rx  = EXCLUDED.pps_rx,
    msg_count = EXCLUDED.msg_count,
    mps = EXCLUDED.mps,
    replicas = EXCLUDED.replicas
    """

    try:
        cur = conn.cursor()
        # ใช้ execute_values เพื่อยัด list 'rows' ลงไปใน SQL ทีเดียว
        execute_values(cur, sql, rows)
        conn.commit()
        cur.close()
    except (OperationalError, InterfaceError) as e:
        print("DB Connection Error during upsert, rolling back...", e)
        conn.rollback()
        global DB
        DB = None # บังคับให้ reconnect ครั้งหน้า
    except Exception as e:
        print("UPSERT ERROR:", e)
        conn.rollback()

# ---------- Main Loop ----------

def main():
    print("Starting ingest loop...")
    while True:
        try:
            now = datetime.now(timezone.utc)
            metrics = collect_all()
            
            if metrics:
                upsert_batch(now, metrics)
                print(f"[OK] {now} → Processed {len(metrics)} deployments")
            else:
                print(f"[WARN] {now} → No metrics found")

        except Exception as e:
            print("MAIN LOOP ERROR:", e)

        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()