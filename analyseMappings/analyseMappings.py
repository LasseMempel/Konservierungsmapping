import pandas as pd
import glob
import numpy as np
import urllib.parse
import re
import matplotlib.pyplot as plt
import seaborn as sns

# === Helper functions ===

def extract_getty_uri(close_match_val):
    """Extract Getty AAT URI from multiline SKOS match fields."""
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
    if not uri: # Check for empty string after strip
        return np.nan

    # Decode percent-encoded characters (e.g., %2F -> /)
    decoded_uri = urllib.parse.unquote(uri)

    # Handle the 'vow' URL format
    if decoded_uri.startswith("https://www.getty.edu/vow/AATFullDisplay") and 'subjectid=' in decoded_uri:
        subject_id_match = re.search(r'subjectid=(\d+)', decoded_uri)
        if subject_id_match:
            # Extract the ID and construct the standard vocab URI
            aat_id = subject_id_match.group(1)
            return f"http://vocab.getty.edu/page/aat/{aat_id}"

    # Normalize the 'vocab' URL format
    if decoded_uri.startswith("http://vocab.getty.edu/page/"):
        # Standardize the path format to 'aat/{id}' and remove trailing slashes
        # Handles both 'aat/{id}' and 'aat%2F{id}' paths
        parsed = urllib.parse.urlparse(decoded_uri)
        path = parsed.path
        # Ensure path starts with /page/ and extract the part after
        if path.startswith('/page/'):
            core_part = path[6:] # Remove '/page/'
            # Split by '/' or '%2F' to separate type and ID, assuming last part is the ID
            parts = re.split(r'[/]', core_part) # Split by '/'
            if len(parts) >= 2 and parts[-1].isdigit(): # Check if last part is numeric ID
                aat_id = parts[-1]
                normalized_path = f"/page/aat/{aat_id}"
                # Reconstruct the URI with standard format
                return urllib.parse.urlunparse(
                    (parsed.scheme, parsed.netloc, normalized_path, parsed.params, parsed.query, parsed.fragment)
                )
            elif len(parts) == 1 and parts[0].isdigit(): # If only ID is present after /page/
                 aat_id = parts[0]
                 normalized_path = f"/page/aat/{aat_id}"
                 return urllib.parse.urlunparse(
                     (parsed.scheme, parsed.netloc, normalized_path, parsed.params, parsed.query, parsed.fragment)
                 )
            # If format doesn't match expected patterns, return original or handle differently
            # For now, return the original decoded URI if it looks like vocab but doesn't parse correctly
            return decoded_uri

    # If it's not a recognized Getty URI, return the lowercased, stripped, decoded version
    # This maintains the original logic for non-Getty URIs if any exist
    return decoded_uri.lower().replace(" ", "")

def safe_split(value, sep='\n'):
    """Safely split a string value, handling NaN and non-string types."""
    if pd.isna(value) or not isinstance(value, str):
        return set() # Return empty set for easier comparison
    return {v.strip() for v in value.split(sep) if v.strip()} # Use set directly

def compute_agreement_focused_on_uri(df1, df2, on="notation"):
    """
    Compute agreement focusing on match property consistency *only* when the AAT URI matches
    AND the URI is present in a match property in both datasets.
    1. Find rows with matching 'notation'.
    2. Among these, find rows where the 'AAT URI' column value matches between datasets.
    3. Among these, find rows where the matching URI is present in ANY match property (exact, close, related) in BOTH datasets.
    4. Among these, check if the matching URI is in the same property type (exact, close, related) in both.
    Returns:
    - aat_uri_agreement: % of notation rows where both have non-NaN AAT URI and they match.
    - match_property_agreement: % of rows meeting criteria 1-3 where the URI was in the same property type.
    - rows_with_both_aat_uri: Total number of rows where both datasets had a non-NaN AAT URI (subset used for aat_uri calc).
    - rows_for_match_calc: Total number of rows meeting criteria 1-3 (subset used for match property calc).
    """
    merged = pd.merge(df1, df2, on=on, suffixes=("_1", "_2"), how="inner")
    # total_compared_notation = len(merged) # This was the old, incorrect return value for AAT calc denominator

    # --- 1 & 2: AAT URI Agreement (rows where both have non-NaN AAT URI and they match) ---
    aat_1_non_empty = merged["AAT URI_1"].notna()
    aat_2_non_empty = merged["AAT URI_2"].notna()
    both_aat_non_empty = aat_1_non_empty & aat_2_non_empty
    aat_matches = merged.loc[both_aat_non_empty, "AAT URI_1"] == merged.loc[both_aat_non_empty, "AAT URI_2"]
    
    rows_with_both_aat_uri = both_aat_non_empty.sum() # Correct denominator for AAT URI % calc
    aat_uri_agreement = (100 * aat_matches.sum() / rows_with_both_aat_uri) if rows_with_both_aat_uri > 0 else np.nan
    # Filter to rows where AAT URI matches (this defines the subset for step 3)
    uri_matched_rows = merged[both_aat_non_empty & aat_matches]

    # --- 3 & 4: Match Property Agreement (only on rows where AAT URI matched AND is in a property in both) ---
    if uri_matched_rows.empty:
        match_property_agreement = np.nan
        rows_for_match_calc = 0
    else:
        match_property_agreements = []
        rows_for_match_calc = 0 # Count rows that meet criteria 1-3
        
        # Iterate through each row where AAT URI matched
        for idx, row in uri_matched_rows.iterrows():
            matched_uri = row["AAT URI_1"] # Since they match, we can use either
            
            # Get sets of URIs for each property in both datasets for this row
            exact_1_uris = safe_split(row["exactMatch_1"])
            close_1_uris = safe_split(row["closeMatch_1"])
            related_1_uris = safe_split(row["relatedMatch_1"])
            
            exact_2_uris = safe_split(row["exactMatch_2"])
            close_2_uris = safe_split(row["closeMatch_2"])
            related_2_uris = safe_split(row["relatedMatch_2"])

            # Check if the matched URI is present in ANY match property in df1
            uri_in_df1_props = matched_uri in (exact_1_uris | close_1_uris | related_1_uris)
            # Check if the matched URI is present in ANY match property in df2
            uri_in_df2_props = matched_uri in (exact_2_uris | close_2_uris | related_2_uris)

            # Only proceed to check property *type* agreement if URI is in a property in BOTH datasets
            if uri_in_df1_props and uri_in_df2_props:
                rows_for_match_calc += 1 # Increment count for rows meeting criteria 1-3

                # Determine which property type(s) the matched URI belongs to in df1
                uri_prop_1 = set()
                if matched_uri in exact_1_uris:
                    uri_prop_1.add("exact")
                if matched_uri in close_1_uris:
                    uri_prop_1.add("close")
                if matched_uri in related_1_uris:
                    uri_prop_1.add("related")

                # Determine which property type(s) the matched URI belongs to in df2
                uri_prop_2 = set()
                if matched_uri in exact_2_uris:
                    uri_prop_2.add("exact")
                if matched_uri in close_2_uris:
                    uri_prop_2.add("close")
                if matched_uri in related_2_uris:
                    uri_prop_2.add("related")

                # Check if the URI appears in the same property type in both datasets
                # This means the intersection of property sets for this URI is non-empty
                if uri_prop_1 & uri_prop_2: # If there's overlap in property types
                     match_property_agreements.append(1)
                else: # URI is in different property types in both datasets that have the URI
                     match_property_agreements.append(0)
            # If URI is not in a property in either df1 or df2, skip this row for match calc

        # Calculate the final percentage for match property agreement based on filtered rows
        match_property_agreement = (100 * sum(match_property_agreements) / rows_for_match_calc) if rows_for_match_calc > 0 else np.nan

    # Return the correct count for rows where AAT URI comparison was possible
    # and the count for rows where match property comparison was possible (after new filter)
    return aat_uri_agreement, match_property_agreement, rows_with_both_aat_uri, rows_for_match_calc


# === Load and normalize data ===
mappings = glob.glob("../mappings/*.csv")
# testing
#mappings = glob.glob("../testmappings/*.csv")

excluded = [
    "mappings\Konservierungsbegriffe-csv_3.csv",
    #"../mappings\Restaurierungsthesaurus.csv",
    "mappings\Restaurierungsthesaurus_normalized.csv"
]
included = [x for x in mappings if x not in excluded]

# Load and prepare Restaurierungsthesaurus
restaurierungsthesaurus = pd.read_csv("../mappings/Restaurierungsthesaurus.csv", encoding='cp1252')
restaurierungsthesaurus['exactMatch'] = restaurierungsthesaurus['closeMatch'].apply(extract_getty_uri)
restaurierungsthesaurus['AAT URI'] = restaurierungsthesaurus['closeMatch'].apply(extract_getty_uri)
restaurierungsthesaurus['closeMatch'] = np.nan
restaurierungsthesaurus['relatedMatch'] = np.nan
restaurierungsthesaurus = restaurierungsthesaurus[["notation", "AAT URI", "exactMatch", "closeMatch", "relatedMatch"]]
# Filter based on first included dataset's notations
first_df_notations = pd.read_csv(included[0], encoding='cp1252')["notation"]
restaurierungsthesaurus = restaurierungsthesaurus[
    restaurierungsthesaurus["notation"].isin(first_df_notations)
]
for col in ["AAT URI", "exactMatch", "closeMatch", "relatedMatch"]:
    restaurierungsthesaurus[col] = restaurierungsthesaurus[col].apply(normalize_uri)

#restaurierungsthesaurus.to_csv("../mappings/Restaurierungsthesaurus_normalized.csv", index=False)

# Load included datasets
included_dfs = {}
required_columns = ["notation", "AAT URI", "exactMatch", "closeMatch", "relatedMatch"]

for mapping in included:
    print(f"Loading {mapping}...")
    df_temp = pd.read_csv(mapping, encoding='cp1252')
    print(f"  Columns found: {list(df_temp.columns)}")
    
    # Check if all required columns exist
    missing_cols = [col for col in required_columns if col not in df_temp.columns]
    if missing_cols:
        print(f"  WARNING: Missing columns {missing_cols} in {mapping}. Skipping this file.")
        continue # Skip this file if required columns are missing
    
    # Select only the required columns
    df_temp = df_temp[required_columns]
    
    # Normalize columns
    for col in ["AAT URI", "exactMatch", "closeMatch", "relatedMatch"]:
        df_temp[col] = df_temp[col].apply(normalize_uri)
    
    included_dfs[mapping] = df_temp
    print(f"  Successfully loaded and processed {mapping}.")

print(f"\nSuccessfully loaded {len(included_dfs)} datasets for comparison.")

# === Compute agreement results ===
if not included_dfs:
    print("No valid datasets found for comparison after filtering. Exiting.")
else:
    results = []

    # Compare Restaurierungsthesaurus with each included DF
    for name, df in included_dfs.items():
        print(f"Computing agreement: Restaurierungsthesaurus vs {name.split('/')[-1]}")
        aat_agree, match_agree, rows_both_aat, rows_for_match = compute_agreement_focused_on_uri(restaurierungsthesaurus, df)
        # rows_both_aat is now the correct count for AAT URI % denominator
        # rows_for_match is the count for Match % denominator (after new filter)
        results.append({
            "Comparison": f"Restaurierungsthesaurus vs {name.split('/')[-1]}",
            "AAT URI %": round(aat_agree, 2) if not np.isnan(aat_agree) else "n/a",
            "Rows with Both AAT URIs": rows_both_aat,
            "Match % (URI+Prop Subset)": round(match_agree, 2) if not np.isnan(match_agree) else "n/a", # Renamed label
            "Rows for Match % Calc": rows_for_match # This is the key denominator for the Match % metric (after new filter)
        })

    # Compare between all included DFs
    for i, (name1, df1) in enumerate(included_dfs.items()):
        for j, (name2, df2) in enumerate(included_dfs.items()):
            if j <= i:
                continue
            print(f"Computing agreement: {name1.split('/')[-1]} vs {name2.split('/')[-1]}")
            aat_agree, match_agree, rows_both_aat, rows_for_match = compute_agreement_focused_on_uri(df1, df2)
            # rows_both_aat is now the correct count for AAT URI % denominator
            # rows_for_match is the count for Match % denominator (after new filter)
            results.append({
                "Comparison": f"{name1.split('/')[-1]} vs {name2.split('/')[-1]}",
                "AAT URI %": round(aat_agree, 2) if not np.isnan(aat_agree) else "n/a",
                "Rows with Both AAT URIs": rows_both_aat,
                "Match % (URI+Prop Subset)": round(match_agree, 2) if not np.isnan(match_agree) else "n/a", # Renamed label
                "Rows for Match % Calc": rows_for_match # This is the key denominator for the Match % metric (after new filter)
            })

    # === Output ===
    if results:
        results_df = pd.DataFrame(results)
        print("\n=== Agreement Summary (URI-Match Focused) ===")
        print(results_df.to_string(index=False))

        # Optional: save to CSV
        # results_df.to_csv("../mappings/agreement_summary_uri_focused.csv", index=False)
    else:
        print("No comparison results to display.")

# === Generate Heatmap with Custom Color Scheme ===
if results and len(results_df) > 0:
    # Extract unique dataset names
    all_datasets = set()
    for comparison in results_df["Comparison"]:
        parts = comparison.split(" vs ")
        for part in parts:
            all_datasets.add(part)
    
    all_datasets = sorted(list(all_datasets))
    
    # Create a mapping for more readable labels
    dataset_labels = {}
    annotator_count = 1
    for ds in all_datasets:
        if "Restaurierungsthesaurus" in ds:
            dataset_labels[ds] = "Fachthesaurus"
        else:
            dataset_labels[ds] = f"annotator{annotator_count}"
            annotator_count += 1
    
    # Create a matrix for the heatmap
    n_datasets = len(all_datasets)
    matrix = np.zeros((n_datasets, n_datasets))
    
    # Fill the matrix with agreement values (using AAT URI % for this example)
    for idx, row in results_df.iterrows():
        comparison = row["Comparison"]
        datasets = comparison.split(" vs ")
        
        if len(datasets) == 2:
            ds1, ds2 = datasets[0], datasets[1]
            
            # Get indices
            try:
                i = all_datasets.index(ds1)
                j = all_datasets.index(ds2)
                
                # Get the agreement value (convert "n/a" back to NaN for processing)
                agreement = row["AAT URI %"]
                if agreement == "n/a":
                    agreement = np.nan
                
                # Fill both sides of the matrix (symmetric)
                matrix[i][j] = agreement if not pd.isna(agreement) else np.nan
                matrix[j][i] = agreement if not pd.isna(agreement) else np.nan
                
            except ValueError:
                continue  # Skip if dataset not found in all_datasets
    
    # Fill diagonal with 100 (perfect agreement with itself)
    for i in range(n_datasets):
        matrix[i][i] = 100.0
    
    # Create labels for the heatmap
    labels = [dataset_labels.get(ds, ds) for ds in all_datasets]
    
    # Define custom color palette matching your webpage
    # From light beige → white → dark navy blue
    colors = ["#F5F3EE", "#FFFFFF", "#1A2B4C"]  # Light beige, White, Dark Navy
    cmap = sns.blend_palette(colors, as_cmap=True)
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    mask = np.isnan(matrix)
    sns.heatmap(
        matrix, 
        mask=mask,
        xticklabels=labels, 
        yticklabels=labels,
        annot=True, 
        fmt='.1f',
        cmap=cmap, 
        center=50,  # Center at 50% for balanced visual
        cbar_kws={'label': 'AAT URI Agreement (%)'},
        square=True,
        linewidths=0.5,
        linecolor='lightgray'
    )
    plt.title("Getty AAT URI Mapping Agreement Heatmap: Fachthesaurus vs Annotators", fontsize=16, pad=20, fontweight='bold', color='#1A2B4C')
    plt.xticks(rotation=45, ha='right', color='#1A2B4C')
    plt.yticks(rotation=0, color='#1A2B4C')
    plt.tight_layout()
    plt.savefig('heatmap_AAT_URI_agreement.png')
    plt.show()
    """
    # Optionally, create a second heatmap for Match % (URI+Prop Subset)
    matrix_match = np.zeros((n_datasets, n_datasets))
    
    for idx, row in results_df.iterrows():
        comparison = row["Comparison"]
        datasets = comparison.split(" vs ")
        
        if len(datasets) == 2:
            ds1, ds2 = datasets[0], datasets[1]
            
            # Get indices
            try:
                i = all_datasets.index(ds1)
                j = all_datasets.index(ds2)
                
                # Get the match agreement value
                agreement = row["Match % (URI+Prop Subset)"]
                if agreement == "n/a":
                    agreement = np.nan
                
                # Fill both sides of the matrix (symmetric)
                matrix_match[i][j] = agreement if not pd.isna(agreement) else np.nan
                matrix_match[j][i] = agreement if not pd.isna(agreement) else np.nan
                
            except ValueError:
                continue  # Skip if dataset not found in all_datasets
    
    # Fill diagonal with 100 (perfect agreement with itself)
    for i in range(n_datasets):
        matrix_match[i][i] = 100.0
    
    # Create the second heatmap
    plt.figure(figsize=(10, 8))
    mask_match = np.isnan(matrix_match)
    sns.heatmap(
        matrix_match, 
        mask=mask_match,
        xticklabels=labels, 
        yticklabels=labels,
        annot=True, 
        fmt='.1f',
        cmap="RdYlGn", 
        center=50,
        cbar_kws={'label': 'Match Agreement (URI+Prop Subset) (%)'},
        square=True
    )
    plt.title("Match Agreement Heatmap: Fachthesaurus vs Annotators", fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    """