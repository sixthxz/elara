import sqlite3, json, textwrap

DB = "project-elara/elara_proxy_metrics.db"
GATE2 = 0.05

con = sqlite3.connect(DB)
cur = con.cursor()

# Per-session: último ciclo de cada sesión con d_rho_series
cur.execute("""
    SELECT r.session_id,
           r.d_rho_series,
           r.lock_frac,          -- si está en records
           r.compressed
    FROM records r
    INNER JOIN (
        SELECT session_id, MAX(cycle) AS max_cycle
        FROM records
        GROUP BY session_id
    ) last ON r.session_id = last.session_id
           AND r.cycle      = last.max_cycle
    ORDER BY r.session_id
""")
rows = cur.fetchall()
con.close()

# Construir tabla
print(r"\begin{center}")
print(r"\resizebox{\columnwidth}{!}{%")
print(r"\begin{tabular}{llp{4cm}l}")
print(r"\toprule")
print(r"\textbf{ID} & \textbf{Type} & \textbf{$\drho$ series} & "
      r"\textbf{$\max_k \drho^{(k)}$} \\")
print(r"\midrule")

for i, (sid, series_json, *_) in enumerate(rows):
    series = json.loads(series_json) if series_json else []
    if not series:
        continue
    max_val = max(series)
    label   = "COHERENT" if max_val < GATE2 else "BREAK"
    rounded = [round(v, 3) for v in series]
    # bold el spike en sesiones BREAK
    if label == "BREAK":
        spike_i = max(range(len(series)), key=lambda k: series[k])
        rounded[spike_i] = r"\mathbf{" + str(rounded[spike_i]) + "}"
    series_str = "$[" + ", ".join(str(v) for v in rounded) + "]$"
    print(f"S{i+1:02d} & {label} & {series_str} & ${max_val:.3f}$ \\\\")

print(r"\bottomrule")
print(r"\end{tabular}}")
print(r"\end{center}")