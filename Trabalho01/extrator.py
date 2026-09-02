import re

import pandas as pd
import tabulate as tab

WORD_PATTERN = r"\b[a-zA-ZáàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ]+\b"


def parse_conversas_xml(filepath: str) -> str:
    """
    Extrai todas as mensagens de um arquivo do dataset PREDADORES-BR
    (c_pr.xml, c_pu.xml, c_pu_mod.xml, i_pu.xml):

        <banco>
          <conversa id="...">
            <linha num="...">
              <autor>hash md5</autor>
              <mensagem>texto</mensagem>
            </linha>
            ...
          </conversa>
          ...
        </banco>

    Não usa um parser XML estrito (xml.etree) porque os marcadores de
    anonimização do dataset (ex: ">nome<", ">local<", ">emoticon<") contêm
    "<" e ">" literais dentro do texto da mensagem, o que invalida o
    documento como XML bem formado. Em vez disso, extraí o conteúdo de
    cada tag <mensagem> via regex, o que é seguro aqui porque as mensagens
    nunca contêm a sequência "</mensagem>" literal.

    O arquivo é lido como UTF-8 e, se falhar, cai para latin-1 automaticamente.

    Retorna o texto com uma mensagem por linha, na ordem em que aparecem no
    arquivo.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            raw = f.read()

    messages = re.findall(r"<mensagem>(.*?)</mensagem>", raw, flags=re.DOTALL)
    messages = [m.strip() for m in messages if m.strip()]
    return "\n".join(messages)


def build_corpus_from_xml(filepaths: list[str], filename_out: str) -> str:
    """
    Junta uma ou mais fontes do dataset (ex.: c_pr.xml, c_pu.xml, c_pu_mod.xml,
    i_pu.xml) em um único corpus de texto (uma mensagem por linha) e salva em
    disco.
    """
    all_messages = [parse_conversas_xml(fp) for fp in filepaths]
    full_text = "\n".join(all_messages)
    with open(filename_out, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"Corpus salvo em {filename_out} ({len(all_messages)} arquivo(s) combinados)")
    return full_text


def load_affect_lexicon(
    filepath: str,
) -> tuple[dict[int, str], list[tuple[str, list[int]]]]:
    """
    Carrega o léxico AffectPT-BR, que segue o formato abaixo:

        %
        30    affect (Affect)
        31    posemo (Positive Emotions)
        32    negemo (Negative Emotions)
        ...
        %
        abaix*    30    32    35
        aberração 30    32
        ...

    - Bloco entre a 1ª e a 2ª linha "%": definição das categorias
      (id <TAB> nome).
    - Depois da 2ª linha "%": termos, onde um "*" no final indica prefixo
      (ex: "abaix*" casa com "abaixo", "abaixar" etc.), seguidos dos ids das
      categorias às quais pertencem.

    Retorna (categorias, termos), onde `categorias` é {id: nome} e `termos`
    é uma lista de (termo, [ids_categoria]).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    percent_idx = [i for i, line in enumerate(lines) if line.strip() == "%"]
    if len(percent_idx) >= 2:
        header_lines = lines[percent_idx[0] + 1 : percent_idx[1]]
        term_lines = lines[percent_idx[1] + 1 :]
    else:
        header_lines, term_lines = [], lines

    categories: dict[int, str] = {}
    for line in header_lines:
        parts = line.strip().split("\t")
        if len(parts) >= 2 and parts[0].isdigit():
            categories[int(parts[0])] = parts[1]

    terms: list[tuple[str, list[int]]] = []
    for line in term_lines:
        if not line.strip() or line.strip() == "%":
            continue
        parts = line.split("\t")
        term = parts[0].strip()
        if not term:
            continue
        cat_ids = [int(p) for p in parts[1:] if p.strip().isdigit()]
        terms.append((term, cat_ids))

    return categories, terms


def lexicon_frequency(
    text: str, terms: list[tuple[str, list[int]]], top_n: int = 20
) -> pd.DataFrame:
    """
    Passo 3: conta, para cada termo do léxico AffectPT-BR, quantas vezes ele
    ocorre no corpus e devolve os top_n termos em ordem decrescente de
    frequência.

    Termos terminados em "*" são tratados como prefixo: a contagem soma a
    frequência de todas as palavras do vocabulário do corpus que começam com
    aquele prefixo (ex: "abaix*" soma "abaixo", "abaixar", "abaixado", ...).
    Termos sem "*" são contados por correspondência exata.
    """
    words = re.findall(WORD_PATTERN, text.lower())
    freq_all = pd.Series(words).value_counts()
    vocab = freq_all.index.to_series()

    rows = []
    for term, _cat_ids in terms:
        term_norm = term.lower()
        if term_norm.endswith("*"):
            prefix = term_norm[:-1]
            count = int(freq_all[vocab.str.startswith(prefix)].sum())
        elif term_norm in freq_all.index:
            count = int(freq_all[term_norm])
        else:
            count = 0
        if count > 0:
            rows.append((term, count))

    df = pd.DataFrame(rows, columns=["termo_lexico", "frequencia"])
    df = df.sort_values("frequencia", ascending=False).reset_index(drop=True)
    return df.head(top_n)


def category_frequency(
    text: str,
    categories: dict[int, str],
    terms: list[tuple[str, list[int]]],
) -> pd.DataFrame:
    """
    Extra (útil para a Discussão): agrega a frequência de ocorrência por
    categoria afetiva (affect, posemo, negemo, anx, anger, sad, ...), somando
    a contagem de todos os termos do léxico que pertencem a cada categoria.
    """
    words = re.findall(WORD_PATTERN, text.lower())
    freq_all = pd.Series(words).value_counts()
    vocab = freq_all.index.to_series()

    cat_counts = {cat_id: 0 for cat_id in categories}
    for term, cat_ids in terms:
        term_norm = term.lower()
        if term_norm.endswith("*"):
            prefix = term_norm[:-1]
            count = int(freq_all[vocab.str.startswith(prefix)].sum())
        elif term_norm in freq_all.index:
            count = int(freq_all[term_norm])
        else:
            count = 0
        for cat_id in cat_ids:
            cat_counts[cat_id] = cat_counts.get(cat_id, 0) + count

    df = pd.DataFrame(
        [(categories.get(cid, str(cid)), c) for cid, c in cat_counts.items()],
        columns=["categoria", "frequencia"],
    )
    return df.sort_values("frequencia", ascending=False).reset_index(drop=True)


def word_frequency(text: str) -> pd.DataFrame:
    """
    Passo 1: conta a frequência de cada palavra no texto, sem remover
    stopwords (conforme a observação do enunciado: "Não remover stopwords").
    """
    words = re.findall(WORD_PATTERN, text.lower())
    freq = pd.Series(words).value_counts()
    df = freq.reset_index()
    df.columns = ["palavra", "frequencia"]
    return df


def descriptive_statistics(text: str) -> pd.DataFrame:
    """
    Calcula as estatísticas descritivas obrigatórias do corpus:
    tokens, vocabulário, mensagens, médias, dispersão, hapax/dis legomena,
    TTR, comprimento médio de palavras e de mensagens, e razão de hapax.

    Aqui, cada linha não vazia do texto extraído é considerada uma "mensagem"
    (equivalente a documento/linha do dataset, conforme o enunciado).
    """
    messages = [line.strip() for line in text.split("\n") if line.strip()]

    # tokens: todas as palavras (com repetição) do corpus inteiro
    all_words = re.findall(WORD_PATTERN, text.lower())

    freq = pd.Series(all_words).value_counts()

    tokens = len(all_words)
    vocabulario = freq.shape[0]

    words_per_message = pd.Series(
        [len(re.findall(WORD_PATTERN, msg.lower())) for msg in messages]
    )

    hapax = int((freq == 1).sum())
    dis_legomena = int((freq == 2).sum())

    ttr = vocabulario / tokens if tokens > 0 else 0
    razao_hapax = hapax / vocabulario if vocabulario > 0 else 0

    comprimento_medio_palavras = pd.Series([len(w) for w in all_words]).mean()
    comprimento_medio_mensagens = pd.Series([len(msg) for msg in messages]).mean()

    stats = {
        "Palavras (tokens)": tokens,
        "Vocabulário (types)": vocabulario,
        "Mensagens": len(messages),
        "Palavras por mensagem (µ)": round(words_per_message.mean(), 2),
        "Palavras por mensagem (σ)": round(words_per_message.std(), 2),
        "Hapax Legomena": hapax,
        "Dis Legomena": dis_legomena,
        "Razão Tipo/Token (TTR)": round(ttr, 4),
        "Comprimento médio das palavras (chars)": round(comprimento_medio_palavras, 2),
        "Comprimento médio das mensagens (chars)": round(
            comprimento_medio_mensagens, 2
        ),
        "Razão de Hapax": round(razao_hapax, 4),
    }

    return pd.DataFrame(list(stats.items()), columns=["Métrica", "Valor"])


def file_statistic(
    filename_txt: str, top_n: int = 100
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imprime a frequência de palavras (Passo 1) e as estatísticas descritivas (Passo 2) do corpus."""
    with open(filename_txt, "r", encoding="utf-8") as f:
        text = f.read()

    # --- Passo 1: Top N palavras mais frequentes ---
    df_freq = word_frequency(text)

    print(f"\nTop {top_n} palavras mais frequentes:\n")
    print(
        tab.tabulate(
            df_freq.head(top_n),
            headers="keys",
            tablefmt="grid",
            showindex=False,
        )
    )

    # --- Passo 2: Estatísticas descritivas do corpus ---
    df_stats = descriptive_statistics(text)

    print("\nEstatísticas Descritivas do Dataset:\n")
    print(
        tab.tabulate(
            df_stats,
            headers="keys",
            tablefmt="grid",
            showindex=False,
        )
    )

    return df_freq, df_stats


def main():
    # --- Dataset: PREDADORES-BR (c_pu_mod.xml) ---
    corpus = build_corpus_from_xml(
        ["./Trabalho01/c_pu_mod.xml"], "./Trabalho01/corpus_c_pu_mod.txt"
    )

    # Passo 1 e Passo 2
    df_freq, df_stats = file_statistic("./Trabalho01/corpus_c_pu_mod.txt")
    df_freq.to_csv("./Trabalho01/passo1_top100_palavras.csv", index=False)
    df_stats.to_csv("./Trabalho01/passo2_estatisticas_descritivas.csv", index=False)

    # Passo 3: cruzamento com o léxico AffectPT-BR
    categories, terms = load_affect_lexicon("./Trabalho01/AffectPT-br")
    df_lexico = lexicon_frequency(corpus, terms, top_n=20)
    df_lexico.to_csv("./Trabalho01/passo3_top20_lexico.csv", index=False)
    print("\nTop 20 termos do léxico AffectPT-BR encontrados no corpus:\n")
    print(tab.tabulate(df_lexico, headers="keys", tablefmt="grid", showindex=False))

    # Extra: frequência agregada por categoria afetiva (útil para a Discussão)
    df_categorias = category_frequency(corpus, categories, terms)
    df_categorias.to_csv("./Trabalho01/extra_frequencia_categorias.csv", index=False)
    print("\nFrequência por categoria afetiva:\n")
    print(tab.tabulate(df_categorias, headers="keys", tablefmt="grid", showindex=False))


if __name__ == "__main__":
    main()
