import io
import re
import unicodedata

import nltk
import pandas as pd
import pymupdf
import pytesseract
import tabulate as tab
from nltk.corpus import stopwords
from PIL import Image
from spellchecker import SpellChecker

spell = SpellChecker(language="pt")

WORD_PATTERN = r"\b[a-zA-ZáàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ]+\b"


def remove_accents(text: str) -> str:
    """Remove acentos mantendo as letras base (á -> a, ç -> c, etc.)"""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# garante que os pacotes de stopwords estão disponíveis (baixa só na primeira vez)
try:
    STOPWORDS_PT = set(stopwords.words("portuguese"))
    STOPWORDS_EN = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS_PT = set(stopwords.words("portuguese"))
    STOPWORDS_EN = set(stopwords.words("english"))

# normaliza as stopwords em português (sem acento, já que o texto salvo também
# não tem) e junta com as stopwords em inglês (artigo tem abstract/referências)
STOPWORDS_ALL = {remove_accents(w) for w in STOPWORDS_PT} | STOPWORDS_EN


def fix_hyphenation(text: str) -> str:
    """
    Junta palavras quebradas por hifenização de fim de linha.
    Decide entre juntar ('al-\nternativa' -> 'alternativa') ou manter o hífen
    ('sócio-\neconômico' -> 'sócio-econômico') verificando qual forma é uma
    palavra válida no dicionário.
    """
    pattern = re.compile(r"(\w+)-\s*\n\s*(\w+)")

    def replace(match: re.Match) -> str:
        part1, part2 = match.group(1), match.group(2)
        joined = part1 + part2  # ex: "alternativa"
        hyphenated = f"{part1}-{part2}"  # ex: "sócio-economico"
        joined_valid = joined.lower() not in spell.unknown([joined.lower()])
        hyphenated_valid = part2.lower() not in spell.unknown([part2.lower()])
        if joined_valid:
            return joined
        elif hyphenated_valid:
            return hyphenated
        else:
            return joined

    return pattern.sub(replace, text)


def extract_text(filename_in: str, filename_out: str, dpi: int = 300) -> str:
    """Extrai o texto de um PDF via OCR e salva em um arquivo de texto."""
    extracted_text = []
    with pymupdf.open(filename_in) as input_file:
        num_pages = len(input_file)
        for i in range(num_pages):
            print(f"Processing Page {i + 1}/{num_pages}...")
            page = input_file[i]
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang="por")
            if text:
                extracted_text.append(text)
    full_text = "\n".join(extracted_text)
    full_text = fix_hyphenation(full_text)
    full_text = remove_accents(full_text)
    print(f"Extracted text from file {filename_in} via OCR and removed accents")
    with open(filename_out, "w", encoding="utf-8") as output_file:
        output_file.write(full_text)
    print(f"Saved to file {filename_out}")
    return full_text


def word_frequency(text: str, remove_stopwords: bool = True) -> pd.DataFrame:
    """Conta a frequência de cada palavra no texto, opcionalmente removendo stopwords."""
    words = re.findall(WORD_PATTERN, text.lower())

    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS_ALL]

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

    # frequência "crua" (sem remover stopwords) para as métricas do corpus
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
    filename_txt: str, top_n: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imprime a frequência de palavras (sem stopwords) e as estatísticas descritivas do corpus."""
    with open(filename_txt, "r", encoding="utf-8") as f:
        text = f.read()

    # --- Top N palavras mais frequentes (sem stopwords) ---
    df_freq = word_frequency(text, remove_stopwords=True)

    print(f"\nTop {top_n} palavras mais frequentes (sem stopwords):\n")
    print(
        tab.tabulate(
            df_freq.head(top_n),
            headers="keys",
            tablefmt="grid",
            showindex=False,
        )
    )

    # --- Estatísticas descritivas do corpus ---
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
    extract_text("./Trabalho01/Artigo01.pdf", "./Trabalho01/saida01.txt")
    file_statistic("./Trabalho01/saida01.txt")


if __name__ == "__main__":
    main()
