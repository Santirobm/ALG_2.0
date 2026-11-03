import pandas as pd
import numpy as np

df = pd.read_csv('ALG_2.0/results/hito3/unseen_distributions_summary.csv')

print("=" * 80)
print("HITO 3.2: RESUMEN DE 6 DISTRIBUCIONES DESCONOCIDAS (UNSEEN DISTRIBUTIONS)")
print("=" * 80)

# Check Recall across all rows
min_recall = df['IDSetRecall_Pct'].min()
avg_recall = df['IDSetRecall_Pct'].mean()
print(f"IDSetRecall: Min={min_recall:.2f}%, Avg={avg_recall:.2f}%")

# Check StrictRankMatch
min_rank = df['StrictRankMatch_Pct'].min()
avg_rank = df['StrictRankMatch_Pct'].mean()
print(f"StrictRankMatch: Min={min_rank:.2f}%, Avg={avg_rank:.2f}%")

print("\n--- RESUMEN POR DISTRIBUCIÓN (Speedup ALG vs FAISS Flat) ---")
summary = df.groupby(['Distribution', 'CacheMode']).agg(
    Mean_Speedup=('Speedup_vs_Flat', 'mean'),
    Min_Speedup=('Speedup_vs_Flat', 'min'),
    Max_Speedup=('Speedup_vs_Flat', 'max'),
    Avg_ALG_us=('Mean_ALG_us', 'mean'),
    Avg_FAISS_us=('Mean_FAISS_us', 'mean'),
    Min_Recall=('IDSetRecall_Pct', 'min')
).reset_index()

print(summary.to_string(index=False))

print("\n--- SPEEDUPS DESTACADOS POR SELECTIVIDAD Y CACHE ---")
piv = df.pivot_table(index=['Distribution', 'Selectivity'], columns=['CacheMode'], values='Speedup_vs_Flat')
print(piv.round(2).to_string())

print("\n--- ANÁLISIS DE PODA (PRUNING RATIO) ---")
pru = df.groupby('Distribution')['Pruning_Ratio'].agg(['min', 'mean', 'max'])
print(pru.to_string())
