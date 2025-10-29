import pandas as pd
import glob
import numpy as np
import urllib.parse

def extract_getty_uri(close_match_val):
    if pd.isna(close_match_val) or not isinstance(close_match_val, str):
        return np.nan
    lines = close_match_val.split('\n')
    for line in lines:
        if line.strip().startswith("http://vocab.getty.edu/page/"):
            return line.strip()
    return np.nan

def normalize_uri(uri):
    """Normalize URI values for fair comparison."""
    if pd.isna(uri):
        return np.nan
    if not isinstance(uri, str):
        return uri
    uri = uri.strip()
    uri = urllib.parse.unquote(uri)
    uri = uri.lower()
    return uri

mappings = glob.glob("../mappings/*.csv")

excluded = ["../mappings\\Konservierungsbegriffe-csv_3.csv", "../mappings\\Restaurierungsthesaurus.csv", "Restaurierungsthesaurus_normalized.csv"]
included = [x for x in mappings if x not in excluded]

restaurierungsthesaurus = pd.read_csv("../mappings\\Restaurierungsthesaurus.csv", encoding='cp1252')
restaurierungsthesaurus['exactMatch'] = restaurierungsthesaurus['closeMatch'].apply(extract_getty_uri)
restaurierungsthesaurus['AAT URI'] = restaurierungsthesaurus['closeMatch'].apply(extract_getty_uri)
restaurierungsthesaurus['closeMatch'] = np.nan
restaurierungsthesaurus['relatedMatch'] = np.nan
restaurierungsthesaurus = restaurierungsthesaurus[["notation", "AAT URI", "exactMatch", "closeMatch", "relatedMatch"]]
restaurierungsthesaurus = restaurierungsthesaurus[restaurierungsthesaurus["notation"].isin(pd.read_csv(included[0], encoding='cp1252')["notation"])]

for col in ["AAT URI", "exactMatch", "closeMatch", "relatedMatch"]:
    restaurierungsthesaurus[col] = restaurierungsthesaurus[col].astype(str).apply(normalize_uri)

restaurierungsthesaurus.to_csv("../mappings\\Restaurierungsthesaurus_normalized.csv", index=False)

included_dfs = {
    mapping: pd.read_csv(mapping, encoding='cp1252')[["notation", "AAT URI", "exactMatch", "closeMatch", "relatedMatch"]]
    for mapping in included
}

for mapping, df in included_dfs.items():
    for col in ["AAT URI", "exactMatch", "closeMatch", "relatedMatch"]:
        df[col] = df[col].astype(str).apply(normalize_uri)

# === Hilfsfunktionen ===

def agreement_rate(df1, df2, col, ignore_missing=True):
    merged = df1.merge(df2, on="notation", suffixes=("_1", "_2"))
    if ignore_missing:
        merged = merged.dropna(subset=[f"{col}_1", f"{col}_2"])
    if merged.empty:
        return np.nan
    return (merged[f"{col}_1"] == merged[f"{col}_2"]).mean() * 100

def match_property_agreement(df1, df2):
    merged = df1.merge(df2, on="notation", suffixes=("_1", "_2"))
    match_cols = ["exactMatch", "closeMatch", "relatedMatch"]

    # Require both to have AAT URI
    merged = merged.dropna(subset=["AAT URI_1", "AAT URI_2"])
    if merged.empty:
        return np.nan

    # Compute per-column agreements, ignoring pairs of NaN
    agreements = []
    for c in match_cols:
        sub = merged[[f"{c}_1", f"{c}_2"]].dropna(how="all")  # drop rows where both are NaN
        if sub.empty:
            continue
        agreements.append((sub[f"{c}_1"] == sub[f"{c}_2"]).mean())

    if not agreements:
        return np.nan

    return np.mean(agreements) * 100


# === Ergebnisse ===

results = []

# 1 & 2: Vergleich Restaurierungsthesaurus mit jedem DF
for name, df in included_dfs.items():
    aat_agree = agreement_rate(restaurierungsthesaurus, df, "AAT URI")
    match_agree = match_property_agreement(restaurierungsthesaurus, df)
    results.append({
        "Comparison": f"Restaurierungsthesaurus vs {name.split('/')[-1]}",
        "AAT URI %": round(aat_agree, 2) if not np.isnan(aat_agree) else "n/a",
        "Match %": round(match_agree, 2) if not np.isnan(match_agree) else "n/a"
    })

# 3 & 4: Vergleich zwischen allen included DFs
for i, (name1, df1) in enumerate(included_dfs.items()):
    for j, (name2, df2) in enumerate(included_dfs.items()):
        if j <= i:
            continue
        aat_agree = agreement_rate(df1, df2, "AAT URI")
        match_agree = match_property_agreement(df1, df2)
        results.append({
            "Comparison": f"{name1.split('/')[-1]} vs {name2.split('/')[-1]}",
            "AAT URI %": round(aat_agree, 2) if not np.isnan(aat_agree) else "n/a",
            "Match %": round(match_agree, 2) if not np.isnan(match_agree) else "n/a"
        })

# === Ausgabe ===
results_df = pd.DataFrame(results)
print("\n=== Agreement Summary ===")
print(results_df.to_string(index=False))

# optional speichern
#results_df.to_csv("../mappings/agreement_summary.csv", index=False)
