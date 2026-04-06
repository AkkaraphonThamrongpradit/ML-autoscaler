import os
import time
import requests
import psycopg2
from datetime import datetime, timezone

PROM_URL = os.getenv("PROM_URL", "http://prometheus.monitoring:9090")
INTERVAL = int(os.getenv("POLL_INTERVAL", "15"))

def connect_db():
    while True:
        try:
            print("Connecting to DB...",
                  os.getenv("DB_HOST"),
                  os.getenv("DB_PORT"))

            conn = psycopg2.connect(
                host=os.getenv("DB_HOST"),
                port=int(os.getenv("DB_PORT")),
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASS")
            )

            print("DB CONNECTED!")
            return conn

        except Exception as e:
            print("DB CONNECT ERROR:", e)
            time.sleep(5)

DB = connect_db()

# ---------- Prometheus Queries ----------

QUERIES = {

"cpu_avg": """
avg by (owner_name) (
  rate(container_cpu_usage_seconds_total{container!=""}[1m])
  * on(pod, namespace) group_left(owner_name)
  kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"compute.*"}
)
""",

"cpu_max": """
max by (owner_name) (
  rate(container_cpu_usage_seconds_total{container!=""}[1m])
  * on(pod, namespace) group_left(owner_name)
  kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"compute.*"}
)
""",

"mem_avg": """
avg by (owner_name) (
  container_memory_usage_bytes{container!=""}
  * on(pod, namespace) group_left(owner_name)
  kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"compute.*"}
)
""",

"mem_max": """
max by (owner_name) (
  container_memory_usage_bytes{container!=""}
  * on(pod, namespace) group_left(owner_name)
  kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"compute.*"}
)
""",

"replicas": """
sum by (owner_name) (
  kube_pod_status_ready
  * on(pod, namespace) group_left(owner_name)
  kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"compute.*"}
)
""",

"rps": """
sum by (owner_name) (
  rate(container_network_receive_packets_total{
    interface="eth0",
    namespace="edge-apps",
    pod=~"compute-node-.*"
  }[1m])
  * on(pod, namespace) group_left(owner_name)
  kube_pod_owner{
    owner_kind="ReplicaSet",
    owner_name=~"compute.*"
  }
)
"""


}

# ---------- Helpers ----------

def query_prom(q):
    r = requests.get(
        f"{PROM_URL}/api/v1/query",
        params={"query": q},
        timeout=10
    )
    return r.json()["data"]["result"]


def collect_all():
    data = {}

    for key, q in QUERIES.items():
        result = query_prom(q)

        for row in result:
            dep = row["metric"].get("owner_name")
            if not dep:
                continue

            value = float(row["value"][1])

            if dep not in data:
                data[dep] = {}

            data[dep][key] = value

    return data


def upsert(ts, dep, m):

    payload = (
        ts,
        dep,
        m.get("cpu_avg"),
        m.get("cpu_max"),
        m.get("mem_avg"),
        m.get("mem_max"),
        m.get("rps"),
        int(float(m.get("replicas", 1)))
    )

    # ===== DEBUG ก่อนยิง DB =====
    print("\n----- BEFORE INSERT -----")
    print("time       :", payload[0])
    print("deployment :", payload[1])
    print("cpu_avg    :", payload[2])
    print("cpu_max    :", payload[3])
    print("mem_avg    :", payload[4])
    print("mem_max    :", payload[5])
    print("rps        :", payload[6])
    print("replicas   :", payload[7])
    print("-------------------------\n")
    # ============================

    sql = """
    INSERT INTO autoscale_features
    (time, deployment, cpu_avg, cpu_max,
     mem_avg, mem_max, rps, replicas)

    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)

    ON CONFLICT (time, deployment)
    DO UPDATE SET
      cpu_avg = EXCLUDED.cpu_avg,
      cpu_max = EXCLUDED.cpu_max,
      mem_avg = EXCLUDED.mem_avg,
      mem_max = EXCLUDED.mem_max,
      rps     = EXCLUDED.rps,
      replicas= EXCLUDED.replicas
    """

    cur = DB.cursor()
    cur.execute(sql, payload)
    DB.commit()
    cur.close()


# ---------- Main Loop ----------

def main():
    print("Starting ingest loop...")

    while True:
        try:
            now = datetime.now(timezone.utc)

            metrics = collect_all()

            for dep, m in metrics.items():
                upsert(now, dep, m)

            print(f"[OK] {now} → {len(metrics)} deployments")

        except Exception as e:
            print("ERROR:", e)

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
