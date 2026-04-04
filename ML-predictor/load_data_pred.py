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
    SELECT *
    FROM autoscale_features
    WHERE time >= NOW() - INTERVAL '2 minutes'
    ORDER BY time DESC
    """

    df = pd.read_sql(sql, conn)
    conn.close()

    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    return df
