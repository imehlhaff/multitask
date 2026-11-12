"""
Combine the CAP party manifesto/platform files into one sentence-level dataset.
See notes.md for the issues handled here.
"""
import re
from pathlib import Path

import pandas as pd
from ftfy import fix_text

RAW_DIR = Path(__file__).parent / "raw"
OUT_DIR = Path(__file__).parent / "clean"

# CAP master major topics; 0 = no policy content
VALID_MAJORTOPICS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23}
COLUMNS = ["doc_id", "country", "party", "year", "text", "majortopic", "subtopic"]

# Irish-specific subtopics -> master codebook subtopics ('Subtopics' tab of codebooks/IPAP CAP codebook v1.2.xlsx)
IRL_TO_MASTER_SUBTOPIC = {
    203: 202, 211: 208, 350: 399, 406: 405, 702: 701, 706: 705, 720: 705,
    930: 900, 931: 900, 932: 900, 933: 900, 999: 900, 1011: 1002, 1411: 1401, 1402: 1401, 1420: 1499,
    1527: 1526, 1907: 1921, 1908: 1921, 1909: 1921, 1911: 1921, 1912: 1921, 1913: 1910, 1914: 1921,
    1916: 1921, 1919: 1921, 1920: 1921, 1922: 1910, 1923: 1910, 2301: 2300, 2399: 2300,
}

# Layered mojibake in the raw US files, which ftfy only half repairs (e.g. "√≠_" -> "í_").
# Meanings are inferred from context, see notes.md.
US_MOJIBAKE = {
    "√≠_": " ",
    "√≠√Ø": "—", "¬Ñ": "—", "√ë": "—",
    "√≠¬ê": "–",
    "√≠¬ä": "...",
    "√≠¬¥": "'", "√î": "'",
    "√≠√ç": '"', "√≠√±": '"', "√≠√Æ": '"', "√í": '"', "√ì": '"',
    "¬å¬´": "•",
    "¬á": "á", "¬é": "é",
    # One-offs
    "communiqu¬Å_s": "communiqués", "Mart√§√≥¬ª": "Martí", "Mart√≠¬Ä¬å_": "Martí", "√≠¬¢√§¬â√•¬å_": "", "¬É": "",
}
US_MOJIBAKE_RE = "|".join(re.escape(k) for k in sorted(US_MOJIBAKE, key=len, reverse=True))

# The 2010 Lib Dem PDF split "fi"/"fl" ligatures from the rest of the word ("offi ce", "fl awed")
SPLIT_LIGATURE_RE = r"(?<![\w-])([^\W\d_]*(?:fi|fl))\s+([a-z]+)"

# Wrong-encoding damage in individual Irish documents, see notes.md. irl_2_2002_1 is handled in fix_irl_encoding.
IRL_DOC_FIXES = {
    # Mac Roman read as Windows-1252, then partly lowercased
    "irl_2_1997_1": {
        "‡": "á", "œ": "ú", "ž": "é", "ƒ": "É", "ç": "á", "ê": "í", "î": "ó", "ÃŽ": "Ó", "\x8f": "è", "ð": "–", "ò": '"',
    },
    "irl_2_1987_1": {"à": "á", "GARDAI\x8d": "GARDAÍ"},
    "irl_6_2002_1": {"¨": "€", "¤": "€", "ñ": "–"},
    "irl_1_1997_1": {"¥": "•"},
    "irl_1_1992_1": {"FÃ…S": "FÁS"},
    "irl_3_1992_1": {"FÃ¥il": "Fáil"},
    "irl_4_2002_1": {"FçS": "FÁS"},
    "irl_5_2007_1": {"GardaÌ": "Gardaí"},
}
IRL_SHARED_FIXES = {
    # "´" is an apostrophe, except where it stands in for a fada
    "Fa´il": "Fáil", "Si´ocha´na": "Síochána", "Gardai´": "Gardaí", "´": "'",
    # Unicode tags left as literal text
    "<U+2208>": "€", "<U+2666>": "•", "<U+FB01>": "fi",
}


def load_uk():
    df = pd.read_csv(RAW_DIR / "uk_manifestos_v1__1.csv", encoding="cp1252")
    party_cols = {"filter_con": "con", "filter_lab": "lab", "filter_ld": "ld", "filter_ukip": "ukip"}
    df["party"] = df[list(party_cols)].idxmax(axis=1).map(party_cols)
    df["country"] = "uk"
    df["text"] = df["description"]
    df["doc_id"] = "uk_" + df["party"] + "_" + df["year"].astype(str)
    is_ld_2010 = df["doc_id"] == "uk_ld_2010"
    df.loc[is_ld_2010, "text"] = df.loc[is_ld_2010, "text"].str.replace(SPLIT_LIGATURE_RE, r"\1\2", regex=True)
    return df


def fix_irl_encoding(doc_id, text):
    text = fix_text(text)
    if doc_id == "irl_2_2002_1":
        # Windows-1252 read as Mac Roman. The doc's few correct "€" signs come back as "Û".
        text = text.replace("ÃŒ", "Ì").replace("<U+02D9>", "˙")
        text = text.encode("mac_roman").decode("cp1252").replace("Û", "€")
    elif doc_id == "irl_2_1997_1":
        # Single quotes became an opening "ô" and a closing "i"
        text = re.sub(r"ô([^ô]*?)i\b", r"'\1'", text)
    for old, new in {**IRL_DOC_FIXES.get(doc_id, {}), **IRL_SHARED_FIXES}.items():
        text = text.replace(old, new)
    # Leftover of a closing double quote
    return re.sub(r'["€]?\x9d', '"', text)


def load_irl():
    df = pd.read_csv(RAW_DIR / "irl_manifestos_data_1981_2020_utf8.csv")
    # 99 = no policy content in the Irish codebook; 0 elsewhere
    df[["majortopic", "subtopic"]] = df[["majortopic", "subtopic"]].replace(99, 0)
    df["subtopic"] = df["subtopic"].replace(IRL_TO_MASTER_SUBTOPIC)
    # Text was double-encoded: drop the stray "Â" so ftfy can repair the rest (e.g. "Â’" -> "’", "Â€" -> "€")
    df["text"] = df["text"].astype(str).str.replace(r"Â(?=[^\x00-\x7f])", "", regex=True)
    df["party"] = df["party"].astype(str)
    df["country"] = "irl"
    # election_in_year separates the Feb and Nov 1982 elections
    df["doc_id"] = (
        "irl_" + df["party"] + "_" + df["year"].astype(str) + "_" + df["election_in_year"].astype(str)
    )
    df["text"] = [fix_irl_encoding(doc_id, text) for doc_id, text in zip(df["doc_id"], df["text"])]
    return df


def load_us(file_name, party):
    df = pd.read_csv(RAW_DIR / file_name)
    # Use the CAP master codes, not pap_majortopic/pap_subtopic (US Policy Agendas codes)
    df["party"] = party
    df["country"] = "us"
    df["text"] = df["description"].astype(str).str.replace(
        US_MOJIBAKE_RE, lambda m: US_MOJIBAKE[m.group(0)], regex=True
    )
    # -555 subtopics only occur with majortopic 0 (no policy content)
    df["subtopic"] = df["subtopic"].replace(-555, 0)
    df["doc_id"] = "us_" + party + "_" + df["year"].astype(str)
    return df


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.concat(
        [
            load_uk()[COLUMNS],
            load_irl()[COLUMNS],
            load_us("Dem_partyplatform_20.2.csv", "dem")[COLUMNS],
            load_us("US-Party-Rep_Platform_19.3_0.csv", "rep")[COLUMNS],
        ],
        ignore_index=True,
    )

    # Fix mojibake (Irish and US files) and normalise whitespace
    df["text"] = df["text"].astype(str).map(fix_text)

    # Rejoin words split by soft hyphens from PDF line breaks ("ecologi\xad cally") and the Irish "¬" variant
    # ("relat¬¬ed"). "Gardai\xad" is an OCR misread of "Gardaí", not a split word.
    df["text"] = df["text"].str.replace("Gardai\xad", "Gardaí", regex=False).str.replace(r"[\xad¬]+\s*", "", regex=True)
    df["text"] = df["text"].str.split().str.join(" ")

    # Drop empty text, Excel formula errors, and -555 / blank / out-of-scheme topic codes
    df["majortopic"] = pd.to_numeric(df["majortopic"], errors="coerce")
    df = df[~df["text"].isin(["", "nan", "#NAME?"]) & df["majortopic"].isin(VALID_MAJORTOPICS)]
    df = df.astype({"year": int, "majortopic": int, "subtopic": int})

    # Drop subtopics that don't belong to their major topic
    df = df[(df["subtopic"] == 0) | (df["subtopic"] // 100 == df["majortopic"])]

    # Contiguous 0..K-1 class indices for multiclass tasks
    for col in ["majortopic", "subtopic"]:
        codes = sorted(df[col].unique())
        df[f"{col}_idx"] = df[col].map({code: i for i, code in enumerate(codes)})

    df.to_csv(OUT_DIR / "cap_manifestos.csv", index=False)
