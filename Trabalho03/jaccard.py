"""Trabalho 2 - Mineração de Textos (Python puro, sem bibliotecas externas)."""

# ---------- Setup ----------
import math
import os

PASTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sinopses")
NOMES = [f"{i}.txt" for i in range(1, 11)]


# ---------- Pré-processamento ----------
PONTUACAO = [".", ",", ";", '"', "'", "“", "”", "‘", "’"]


def ler_arquivo(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        return f.read()


def tokenizar(texto):
    for ch in PONTUACAO:
        texto = texto.replace(ch, " ")
    return texto.split()


tokens = {nome: tokenizar(ler_arquivo(f"{PASTA}/{nome}")) for nome in NOMES}
for nome in NOMES:
    print(nome, "->", len(tokens[nome]), "tokens")


# ---------- Questão 1 - Jaccard ----------
# (b)
conjuntos = {nome: set(tokens[nome]) for nome in NOMES}


# (c)
def jaccard(a, b):
    uniao = a | b
    if len(uniao) == 0:
        return 0.0
    return len(a & b) / len(uniao)


pares_jaccard = []
for i in range(len(NOMES)):
    for j in range(i + 1, len(NOMES)):
        n1, n2 = NOMES[i], NOMES[j]
        pares_jaccard.append((n1, n2, jaccard(conjuntos[n1], conjuntos[n2])))

# (d)
pares_jaccard.sort(key=lambda p: p[2], reverse=True)
print("Top 4 — Jaccard")
for n1, n2, s in pares_jaccard[:4]:
    print(f"{n1} - {n2} - {s:.3f}")


# ---------- Questão 2 - TF, IDF e TF-IDF ----------
# Vocabulário (todos os termos distintos, ordenados)
vocab = sorted(set(t for nome in NOMES for t in tokens[nome]))
print("Tamanho do vocabulário:", len(vocab))


# (b)
def contar(lista):
    freq = {}
    for t in lista:
        freq[t] = freq.get(t, 0) + 1
    return freq


freqs = {nome: contar(tokens[nome]) for nome in NOMES}

TF = {}
for termo in vocab:
    linha = []
    for nome in NOMES:
        f = freqs[nome].get(termo, 0)
        linha.append(1 + math.log10(f) if f > 0 else 0.0)
    TF[termo] = linha

# (c)
N = len(NOMES)
df = {termo: sum(1 for v in TF[termo] if v > 0) for termo in vocab}
idf = {termo: math.log10(N / df[termo]) for termo in vocab}

# (d)
TFIDF = {termo: [v * idf[termo] for v in TF[termo]] for termo in vocab}


# (e)
def vetor_doc(k):
    return [TFIDF[termo][k] for termo in vocab]


def cosseno(u, v):
    prod = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return prod / (nu * nv)


vetores = [vetor_doc(k) for k in range(N)]

pares_tfidf = []
for i in range(N):
    for j in range(i + 1, N):
        pares_tfidf.append((NOMES[i], NOMES[j], cosseno(vetores[i], vetores[j])))

# (f)
pares_tfidf.sort(key=lambda p: p[2], reverse=True)
print("Top 4 — TF-IDF (cosseno)")
for n1, n2, s in pares_tfidf[:4]:
    print(f"{n1} - {n2} - {s:.3f}")


# ---------- Exportar tabelas para montar os PDFs ----------
def salvar_csv(arquivo, cabecalho, linhas):
    with open(arquivo, "w", encoding="utf-8") as f:
        f.write(",".join(cabecalho) + "\n")
        for l in linhas:
            f.write(",".join(str(x) for x in l) + "\n")


salvar_csv(
    "./Trabalho03/tabela_df_idf.csv",
    ["Termo", "df", "idf"],
    [[t, df[t], f"{idf[t]:.3f}"] for t in vocab],
)

salvar_csv(
    "./Trabalho03/matriz_tf.csv",
    ["Termo"] + NOMES,
    [[t] + [f"{v:.3f}" for v in TF[t]] for t in vocab],
)

salvar_csv(
    "./Trabalho03/matriz_tfidf.csv",
    ["Termo"] + NOMES,
    [[t] + [f"{v:.3f}" for v in TFIDF[t]] for t in vocab],
)

print("Gerados: tabela_df_idf.csv, matriz_tf.csv, matriz_tfidf.csv")
