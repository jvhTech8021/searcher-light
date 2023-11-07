from connection import create_conn, get_report

conn = create_conn()

if conn is not None:
    cur = conn.cursor()
    cur.execute("SELECT * FROM insights")
    rows = cur.fetchall()
    print(f"There are {len(rows)} in the insights table")
    cur.close()
    conn.close()

if conn is not None:
    report_id = 11  # replace with the actual report_id
    rows = get_report(conn, report_id)
    for row in rows:
        print(row)
    conn.close()