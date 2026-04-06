import os
import time
import requests
import psycopg2

from psycopg2 import OperationalError, InterfaceError

from datetime import datetime, timezone


PROM_URL = os.getenv("PROM_URL", "http://prometheus.monitoring:9090")
INTERVAL = int(os.getenv("POLL_INTERVAL", "15"))

DB = None


# ---------- DB CONNECT ----------

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

            conn.autocommit = False

            print("DB CONNECTED!")

            return conn

        except Exception as e:

            print("DB CONNECT ERROR:", e)
            time.sleep(5)


def get_db():

    global DB

    if DB is None or DB.closed:
        print("DB connection closed. Reconnecting...")
        DB = connect_db()

    return DB


# ---------- Prometheus Queries ----------

QUERIES = {

"cpu_avg": """
avg by (owner_name) (
  rate(container_cpu_usage_seconds_total{container!=""}[1m])
  * on(pod, namespace) group_left(owner_name)
  kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"ems-worker-.*"}
)
""",

"cpu_max": """
max by (owner_name) (
  rate(container_cpu_usage_seconds_total{container!=""}[1m])
  * on(pod, namespace) group_left(owner_name)
  kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"ems-worker-.*"}
)
""",

"mem_avg": """
avg by (owner_name) (
  container_memory_usage_bytes{container!=""}
  * on(pod, namespace) group_left(owner_name)
  kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"ems-worker-.*"}
)
""",

"mem_max": """
max by (owner_name) (
  container_memory_usage_bytes{container!=""}
  * on(pod, namespace) group_left(owner_name)
  kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"ems-worker-.*"}
)
""",

"replicas": """
sum by (owner_name) (
  kube_pod_status_ready
  * on(pod, namespace) group_left(owner_name)
  kube_pod_owner{owner_kind="ReplicaSet", owner_name=~"ems-worker-.*"}
)
""",

"pps_rx": """
sum by (owner_name) (
  rate(container_network_receive_packets_total{
    interface="eth0",
    namespace="edge-apps",
    pod=~"ems-worker-.*"
  }[1m])
  * on(pod, namespace) group_left(owner_name)
  kube_pod_owner{
    owner_kind="ReplicaSet",
    owner_name=~"ems-worker-.*"
  }
)
"""

}


# ---------- Prometheus ----------

def query_prom(q):

    r = requests.get(
        f"{PROM_URL}/api/v1/query",
        params={"query": q},
        timeout=10
    )

    r.raise_for_status()

    return r.json()["data"]["result"]


def collect_all():

    data = {}

    for key, q in QUERIES.items():

        result = query_prom(q)

        for row in result:

            dep = row["metric"].get("owner_name")

            if not dep:
                continue
            
            dep = dep.rsplit("-", 1)[0]

            value = float(row["value"][1])

            if dep not in data:
                data[dep] = {}

            data[dep][key] = value

    return data


# ---------- UPSERT ----------

def upsert(ts, dep, m):

    payload = (
        ts,
        dep,
        m.get("cpu_avg"),
        m.get("cpu_max"),
        m.get("mem_avg"),
        m.get("mem_max"),
        m.get("pps_rx"),
        int(float(m.get("replicas", 1)))
    )

    sql = """
    INSERT INTO autoscale_features
    (time, deployment, cpu_avg, cpu_max,
     mem_avg, mem_max, pps_rx, replicas)

    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)

    ON CONFLICT (time, deployment)
    DO UPDATE SET
      cpu_avg = EXCLUDED.cpu_avg,
      cpu_max = EXCLUDED.cpu_max,
      mem_avg = EXCLUDED.mem_avg,
      mem_max = EXCLUDED.mem_max,
      pps_rx     = EXCLUDED.pps_rx,
      replicas= EXCLUDED.replicas
    """

    retries = 2

    for attempt in range(retries):

        try:

            conn = get_db()

            cur = conn.cursor()

            cur.execute(sql, payload)

            conn.commit()

            cur.close()

            return

        except (OperationalError, InterfaceError) as e:

            print("DB ERROR:", e)

            try:
                conn.rollback()
            except:
                pass

            print("Reconnecting DB...")

            global DB
            DB = connect_db()

        except Exception as e:

            print("QUERY ERROR:", e)

            try:
                conn.rollback()
            except:
                pass

            return


# ---------- Main Loop ----------

def main():

    global DB

    DB = connect_db()

    print("Starting ingest loop...")

    while True:

        try:

            now = datetime.now(timezone.utc)

            metrics = collect_all()

            for dep, m in metrics.items():
                upsert(now, dep, m)

            print(f"[OK] {now} → {len(metrics)} deployments")

        except Exception as e:

            print("INGEST ERROR:", e)

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()