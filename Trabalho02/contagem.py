import csv
import re
from collections import Counter

with open("./Trabalho02/texto.txt", encoding="utf-8") as f:
    texto = f.read()


def tokenizar(t):
    return re.findall(r"\w+", t, flags=re.UNICODE)


# (c) COM normalização de case
tokens_norm = tokenizar(texto.lower())
freq_norm = Counter(tokens_norm)
N_norm, V_norm = len(tokens_norm), len(freq_norm)

# (d) SEM normalização de case
tokens_raw = tokenizar(texto)
freq_raw = Counter(tokens_raw)
N_raw, V_raw = len(tokens_raw), len(freq_raw)

# Gerar CSV (versão normalizada), em ordem alfabética
with open("./Trabalho02/frequencia.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["termo", "frequencia"])
    for termo, freq in freq_norm.most_common():
        w.writerow([termo, freq])

# Saídas para os prints
print("=== TEXTO PROCESSADO (início) ===")
print(" ".join(tokens_norm[:150]))

print("\n=== TOP 20 TERMOS (case normalizado) ===")
for termo, freq in freq_norm.most_common(20):
    print(f"{termo:15s} {freq}")

print("\n=== TOTAIS ===")
print(f"Com normalização de case: N = {N_norm} | |V| = {V_norm}")
print(f"Sem normalização de case: N = {N_raw} | |V| = {V_raw}")
