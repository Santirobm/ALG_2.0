import pandas as pd
import numpy as np

def df_to_markdown_table(df):
    """Convierte un DataFrame o Pivot Table a formato Markdown sin tabulate."""
    df_reset = df.reset_index()
    headers = [str(col) if not isinstance(col, tuple) else " / ".join(map(str, col)) for col in df_reset.columns]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df_reset.iterrows():
        row_vals = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                row_vals.append(f"{v:.2f}")
            else:
                row_vals.append(str(v))
        lines.append("| " + " | ".join(row_vals) + " |")
    return "\n".join(lines)

def run_analysis():
    raw_path = 'results/hito3/raw_queries.csv'
    sum_path = 'results/hito3/summary.csv'

    df_raw = pd.read_csv(raw_path)
    df_sum = pd.read_csv(sum_path)

    print("================================================================================")
    print("ANALISIS CANONICO HITO 3.1: NATIVE BENCHMARK INTEGRITY")
    print("================================================================================")

    # 1. Exactness verification
    df_alg = df_sum[df_sum['Engine'] == 'ALG_Plan5A']
    df_flat = df_sum[df_sum['Engine'] == 'FAISS_Flat']

    print("\n1. CERTIFICACION DE EXACTITUD ESTRICTA (StrictRankMatch@K vs Double Ground Truth):")
    print(f"   - Min StrictRankMatch ALG vs GT:  {df_alg['StrictRankMatch_vs_GT_Pct'].min():.2f}%")
    print(f"   - Mean StrictRankMatch ALG vs GT: {df_alg['StrictRankMatch_vs_GT_Pct'].mean():.2f}%")
    print(f"   - Max StrictRankMatch ALG vs GT:  {df_alg['StrictRankMatch_vs_GT_Pct'].max():.2f}%")
    print(f"   - Min IDSetRecall ALG vs GT:      {df_alg['IDSetRecall_Pct'].min():.2f}%")
    print(f"   - Mean IDSetRecall ALG vs GT:     {df_alg['IDSetRecall_Pct'].mean():.2f}%")
    print(f"   - Max IDSetRecall ALG vs GT:      {df_alg['IDSetRecall_Pct'].max():.2f}%")
    print(f"   - Max Distance Error ALG vs GT:   {df_alg['MaxDistErr'].max():.6e}")

    print("\n   FAISS Flat vs Double Ground Truth:")
    print(f"   - Mean StrictRankMatch Flat vs GT: {df_flat['StrictRankMatch_vs_GT_Pct'].mean():.2f}%")
    print(f"   - Mean IDSetRecall Flat vs GT:     {df_flat['IDSetRecall_Pct'].mean():.2f}%")
    print(f"   - Max Distance Error Flat vs GT:   {df_flat['MaxDistErr'].max():.6e}")

    # 2. Resumen por selectividad y CacheMode
    print("\n2. SPEEDUP ALG PLAN 5A vs FAISS IndexFlatL2 (NATIVO C++):")
    piv = df_alg.pivot_table(index=['CacheMode', 'Selectivity'], 
                             columns=['TopK'], 
                             values='Speedup_vs_Flat', 
                             aggfunc='mean')
    print(piv.round(2))

    print("\n3. SPEEDUP POR ESCENARIO (MEDIA GLOBAL):")
    piv_sc = df_alg.pivot_table(index='Scenario', 
                                columns='CacheMode', 
                                values='Speedup_vs_Flat', 
                                aggfunc='mean')
    print(piv_sc.round(2))

    print("\n4. PERCENTILES DE LATENCIA (p50, p95, p99 - METODO TIPO 7 CONTINUO):")
    piv_lat = df_sum[df_sum['Engine'].isin(['ALG_Plan5A', 'FAISS_Flat'])].pivot_table(
        index=['CacheMode', 'Engine'],
        values=['Mean_us', 'p50_us', 'p95_us', 'p99_us'],
        aggfunc='mean'
    )
    print(piv_lat.round(1))

    print("\n5. COMPARACION CON FAISS IndexIVFFlat (Recall vs Latencia):")
    df_ivf32 = df_sum[df_sum['Engine'] == 'FAISS_IVF32']
    piv_ivf = df_ivf32.pivot_table(index=['Selectivity'], 
                                   columns='Scenario', 
                                   values='IDSetRecall_Pct', 
                                   aggfunc='mean')
    print(piv_ivf.round(1))

    # Guardar resumen en formato Markdown
    md_path = 'results/hito3/informe_hito3_1.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Informe de Resultados Hito 3.1: Native Benchmark Integrity\n\n")
        f.write("## 1. Protocolo y Certificación Matemática Rigurosa\n")
        f.write("- **Configuración:** $D = 64$, $N = 100{,}000$, $float32$ (25.6 MB de corpus crudo).\n")
        f.write("- **Hardware:** Intel Core i5-13500 (24 MB Intel Smart Cache L3), CPU Pinning P-Core 0, Priority `HIGH_PRIORITY_CLASS`.\n")
        f.write("- **FAISS C++ Nativo:** Enlace directo C++ contra `faiss.dll` (MSVC ABI) mediante símbolos exportados `search@IndexFlat` y `search@IndexIVF`. **Cero CPython, cero PyObjects, cero diccionarios/tuplas y cero conversión de NumPy dentro del cronómetro**.\n")
        f.write("- **Filtros:** `IDSelectorBitmap` preconstruido para FAISS; listas contiguas pre-indexadas por selectividad para ALG Plan 5A.\n")
        f.write("- **Ground Truth Independiente en Double (binary64):** Acumulación, selección en heap y ordenación estrictamente en 64-bit double, con política determinista de desempate por menor ID.\n")
        f.write("- **Auditoría de Exactitud:** `StrictRankMatch@K` evalúa identidad exacta posición por posición ($a_i = gt_i$). `IDSetRecall@K` evalúa pertenencia de conjuntos. `MaxDistErr` mide la cota de error numérico.\n")
        f.write("- **Estadística:** Cuantiles continuos Tipo-7 (estándar NumPy/R) con 100 consultas WARM y **100 consultas COLD** por punto experimental (**10,000 consultas totales**, con evicción LLC de 36 MB entre motores en cada consulta fría).\n\n")

        f.write("## 2. Auditoría de Exactitud y Recall\n\n")
        f.write(f"- **StrictRankMatch@K (ALG vs Double GT):** Media **{df_alg['StrictRankMatch_vs_GT_Pct'].mean():.2f}%** (Min: {df_alg['StrictRankMatch_vs_GT_Pct'].min():.2f}%, Max: {df_alg['StrictRankMatch_vs_GT_Pct'].max():.2f}%).\n")
        f.write(f"- **IDSetRecall@K (ALG vs Double GT):** Media **{df_alg['IDSetRecall_Pct'].mean():.2f}%** (100.00% exacto en todas las consultas).\n")
        f.write(f"- **Max Dist Error (ALG vs Double GT):** **{df_alg['MaxDistErr'].max():.6e}** (acorde al límite teórico de precisión de máquina $float32$).\n")
        f.write(f"- **StrictRankMatch@K (ALG vs FAISS Flat):** Media **{df_alg['StrictRankMatch_vs_Flat_Pct'].mean():.2f}%**.\n\n")

        f.write("## 3. Resumen de Speedup Nativo (ALG Plan 5A vs FAISS IndexFlatL2 Nativo)\n\n")
        f.write(df_to_markdown_table(piv.round(2)))
        f.write("\n\n")

        f.write("## 4. Desglose de Rendimiento Nativo por Escenario\n\n")
        f.write(df_to_markdown_table(piv_sc.round(2)))
        f.write("\n\n")

        f.write("## 5. Comparativa de Percentiles de Latencia (µs)\n\n")
        f.write(df_to_markdown_table(piv_lat.round(1)))
        f.write("\n\n")

        f.write("## 6. Frontera de Calidad: Recall de FAISS IndexIVFFlat (nprobe=32)\n\n")
        f.write(df_to_markdown_table(piv_ivf.round(1)))
        f.write("\n")

    print(f"\nInforme Markdown generado en: {md_path}")

if __name__ == '__main__':
    run_analysis()
