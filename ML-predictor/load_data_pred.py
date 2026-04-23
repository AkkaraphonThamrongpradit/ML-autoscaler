import pandas as pd
import psycopg2

def load_data():

    conn = psycopg2.connect(
        host="10.96.136.61",
        port=5432,
        dbname="autoscale-db",
        user="admin",
        password="admin123"
    )

    sql = """
    SELECT *
    FROM autoscale_features
    WHERE time >= NOW() - INTERVAL '5 minutes'
    AND deployment = 'ems-worker-edge-d'
    ORDER BY time ASC
    """

    df = pd.read_sql(sql, conn)
    conn.close()

    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    return df
