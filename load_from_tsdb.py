import pandas as pd
import psycopg2

def load_data():
    conn = psycopg2.connect(
        host="13.250.203.51",
        port=30086,
        dbname="autoscale-db",
        user="admin",
        password="admin123"
    )

    sql = """
    SELECT time, deployment, cpu_avg, cpu_max, mem_avg, mem_max, pps_rx, replicas
    FROM autoscale_features
    ORDER BY time DESC
    """

    df = pd.read_sql(sql, conn)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
