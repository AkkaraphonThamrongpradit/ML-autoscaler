import time
import requests

from kubernetes import client, config


# =========================
# CONFIG
# =========================

NAMESPACE = "edge-apps"

PROMETHEUS_URL = "http://prometheus:9090"

QUERY_SPIKE_PROB = "cpu_spike_probability"

MIN_REPLICAS = 1
MAX_REPLICAS = 10

SPIKE_UP_THRESHOLD = 0.7
SPIKE_DOWN_THRESHOLD = 0.35

SCALE_UP_COOLDOWN = 15
SCALE_DOWN_COOLDOWN = 60

SMOOTH_ALPHA = 0.3

LOOP_INTERVAL = 5


# =========================
# K8s Client
# =========================

config.load_incluster_config()
apps_api = client.AppsV1Api()


# =========================
# Controller State
# =========================

state = {}


# =========================
# Prometheus Query
# =========================

def query_spike_prob_all():

    url = f"{PROMETHEUS_URL}/api/v1/query"

    try:
        r = requests.get(url, params={"query": QUERY_SPIKE_PROB}, timeout=3)

        data = r.json()

        results = {}

        for item in data["data"]["result"]:

            metric = item["metric"]

            if "deployment" not in metric:
                continue

            dep = metric["deployment"]

            value = float(item["value"][1])

            results[dep] = value

        return results

    except Exception as e:

        print("Prometheus query error:", e)

        return {}


# =========================
# Get replicas
# =========================

def get_current_replicas(deployment):

    try:

        dep = apps_api.read_namespaced_deployment(
            deployment,
            NAMESPACE
        )

        r = dep.spec.replicas

        if r is None:
            return 0

        return r

    except Exception as e:

        print("Deployment read error:", deployment, e)

        return None


# =========================
# Scale deployment
# =========================

def scale_deployment(deployment, replicas):

    try:

        body = {
            "spec": {
                "replicas": int(replicas)
            }
        }

        apps_api.patch_namespaced_deployment_scale(
            deployment,
            NAMESPACE,
            body
        )

        print(f"Scaling {deployment} → {replicas}")

    except Exception as e:

        print("Scaling error:", deployment, e)


# =========================
# Initialize deployment state
# =========================

def get_state(dep):

    if dep not in state:

        state[dep] = {
            "last_prob": None,
            "last_scale_up": 0,
            "last_scale_down": 0
        }

    return state[dep]


# =========================
# Autoscaler Loop
# =========================

print("ML Autoscaler started")

while True:

    try:

        probs = query_spike_prob_all()

        if len(probs) == 0:
            time.sleep(LOOP_INTERVAL)
            continue

        for dep, spike_prob in probs.items():

            s = get_state(dep)

            current = get_current_replicas(dep)

            if current is None:
                continue

            # =========================
            # Smoothing
            # =========================

            if s["last_prob"] is None:
                prob_smooth = spike_prob
            else:
                prob_smooth = (
                    (1 - SMOOTH_ALPHA) * s["last_prob"]
                    + SMOOTH_ALPHA * spike_prob
                )

            s["last_prob"] = prob_smooth

            now = time.time()

            print(
                f"{dep} | prob={prob_smooth:.3f} | replicas={current}"
            )

            # =========================
            # SCALE UP
            # =========================

            if prob_smooth > SPIKE_UP_THRESHOLD:

                if now - s["last_scale_up"] > SCALE_UP_COOLDOWN:

                    new_replica = min(current * 2, MAX_REPLICAS)

                    if new_replica > current:

                        scale_deployment(dep, new_replica)

                        s["last_scale_up"] = now

            # =========================
            # SCALE DOWN
            # =========================

            elif prob_smooth < SPIKE_DOWN_THRESHOLD:

                if now - s["last_scale_down"] > SCALE_DOWN_COOLDOWN:

                    new_replica = max(current - 1, MIN_REPLICAS)

                    if new_replica < current:

                        scale_deployment(dep, new_replica)

                        s["last_scale_down"] = now

        time.sleep(LOOP_INTERVAL)

    except Exception as e:

        print("Controller error:", e)

        time.sleep(LOOP_INTERVAL)