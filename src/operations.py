from .connection import create_conn
import json

conn = create_conn()

def write_to_table(table, data_list):
    cur = conn.cursor()
    
    for data in data_list:
        placeholders = ', '.join(['%s'] * len(data))
        columns = ', '.join(data.keys())
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        values = tuple([json.dumps(val) if isinstance(val, (dict, list)) else val for val in data.values()])

        try:
            cur.execute(sql, values)  # Pass the tuple of values directly
            conn.commit()
        except Exception as e:
            print(f"Error {e}")
            conn.rollback()

    cur.close()

# write a method to read from the table
def read_clusters_from_table(table):
    cur = conn.cursor()

    # Get unique cluster values
    cur.execute(f"SELECT DISTINCT cluster FROM {table}")
    unique_clusters = [item[0] for item in cur.fetchall()]

    clusters_data = {}  # Dictionary to store data for each cluster

    for cluster in unique_clusters:
        # Fetch question and insight for each cluster
        query = f"SELECT question, insight FROM {table} WHERE cluster = %s"
        cur.execute(query, (cluster,))
        rows = cur.fetchall()

        # Add data to dictionary
        clusters_data[cluster] = rows

    cur.close()

    return clusters_data



# write a method to read from the table

