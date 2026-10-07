"""
PubChem property extractor + verifier for functional_group_dataset.csv.

Runs in two explicit, separable steps (as requested):

  STEP 1 - extract_pubchem_data():
      Pulls fresh property data from PubChem for all 44 molecules and saves
      it to its own CSV (pubchem_extracted.csv). This file is a snapshot of
      what PubChem currently reports - nothing from the workbook is touched
      or compared yet.

  STEP 2 - compare_with_dataset():
      Loads pubchem_extracted.csv and functional_group_dataset.csv side by
      side and writes a comparison report (comparison_report.csv).

Two different PubChem services are used, because PubChem stores these two
kinds of data differently:

  A. PUG REST 'property' table (fast, batched, near-universal coverage):
       MolecularWeight, TPSA, HBondDonorCount, HBondAcceptorCount
       -> These map directly to columns 'mw', 'tpsa', 'hbd', 'hba' and are
          the only columns this script auto-flags as match/mismatch.

  B. PUG View experimental section tree (patchy coverage, free-text values):
       Boiling Point, Solubility, Dissociation Constants
       -> These map loosely to 'boiling_point_c', 'water_solubility',
          'pka'/'pkah'. They are fetched as RAW TEXT for manual review only
          - never auto-compared - because:
            - not every molecule has these sections at all
            - solubility in the workbook is a qualitative class, PubChem's
              text is a free-form experimental description
            - a single "Dissociation Constants" entry from PubChem doesn't
              say whether it's the acid's own pKa or the pKa of a
              protonated conjugate acid (pKaH), so it can't be safely
              auto-assigned to either 'pka' or 'pkah'

PubChem usage policy: no more than 5 requests/second. This script sleeps
between requests to stay well under that limit.

Docs:
  PUG REST:  https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
  PUG View:  https://pubchem.ncbi.nlm.nih.gov/docs/pug-view
"""

import time
import csv
import requests
import pandas as pd

DATASET_CSV_PATH = "functional_group_dataset.csv"
EXTRACTED_CSV_PATH = "pubchem_extracted.csv"
COMPARISON_CSV_PATH = "comparison_report.csv"

PUG_REST_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUG_VIEW_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
REQUEST_DELAY_SECONDS = 0.25  # stays under the 5 requests/second limit

COMPUTED_PROPERTIES = ["MolecularWeight", "TPSA", "HBondDonorCount", "HBondAcceptorCount"]

# Experimental section headings to pull as raw text for manual review.
EXPERIMENTAL_HEADINGS = {
    "boiling_point_pubchem_raw": "Boiling Point",
    "water_solubility_pubchem_raw": "Solubility",
    "dissociation_constants_pubchem_raw": "Dissociation Constants",
}

NUMERIC_MATCH_TOLERANCE = {
    "mw": 0.05,
    "tpsa": 0.5,
    "hbd": 0,
    "hba": 0,
}


# ---------------------------------------------------------------------------
# STEP 1: extraction
# ---------------------------------------------------------------------------

def get_cid_by_name(molecule_name):
    """Look up a PubChem CID from a molecule name. Returns None if not found."""
    url = f"{PUG_REST_BASE}/compound/name/{molecule_name}/cids/JSON"
    response = requests.get(url)
    time.sleep(REQUEST_DELAY_SECONDS)
    if response.status_code != 200:
        return None
    data = response.json()
    return data.get("IdentifierList", {}).get("CID", [None])[0]


def get_computed_properties(cid):
    """Fetch the PUG REST computed property table for one CID."""
    prop_list = ",".join(COMPUTED_PROPERTIES)
    url = f"{PUG_REST_BASE}/compound/cid/{cid}/property/{prop_list}/JSON"
    response = requests.get(url)
    time.sleep(REQUEST_DELAY_SECONDS)
    if response.status_code != 200:
        return {}
    rows = response.json().get("PropertyTable", {}).get("Properties", [])
    return rows[0] if rows else {}


def _find_all_sections_by_heading(sections, target_heading):
    """Recursively find every section in the PUG View tree matching target_heading."""
    matches = []
    for section in sections:
        if section.get("TOCHeading") == target_heading:
            matches.append(section)
        nested = section.get("Section")
        if nested:
            matches.extend(_find_all_sections_by_heading(nested, target_heading))
    return matches


def _extract_all_value_strings(section):
    """Pull every plain-text value out of a PUG View section's Information list."""
    texts = []
    for info in section.get("Information", []):
        value = info.get("Value", {})
        for item in value.get("StringWithMarkup", []):
            text = item.get("String")
            if text:
                texts.append(text)
    return texts


def get_experimental_properties(cid):
    """
    Fetch experimental section text (boiling point, solubility, dissociation
    constants) for one CID via PUG View. Each result is the raw text found,
    semicolon-joined if multiple entries exist. Missing sections -> None.
    """
    url = f"{PUG_VIEW_BASE}/data/compound/{cid}/JSON"
    response = requests.get(url)
    time.sleep(REQUEST_DELAY_SECONDS)
    results = {key: None for key in EXPERIMENTAL_HEADINGS}
    if response.status_code != 200:
        return results

    record_sections = response.json().get("Record", {}).get("Section", [])
    for column_name, heading in EXPERIMENTAL_HEADINGS.items():
        sections = _find_all_sections_by_heading(record_sections, heading)
        all_texts = []
        for section in sections:
            all_texts.extend(_extract_all_value_strings(section))
        if all_texts:
            results[column_name] = "; ".join(all_texts)
    return results


def extract_pubchem_data(molecule_names, output_csv_path):
    """
    STEP 1. Fetch fresh PubChem data for the given molecule names and save
    it as its own CSV. No comparison happens here.
    """
    fieldnames = (
        ["iupac_name", "cid"]
        + [f"{prop.lower()}_pubchem" for prop in COMPUTED_PROPERTIES]
        + list(EXPERIMENTAL_HEADINGS.keys())
    )
    rows = []

    for name in molecule_names:
        cid = get_cid_by_name(name)
        if cid is None:
            print(f"WARNING: no PubChem CID found for '{name}', skipping.")
            rows.append({"iupac_name": name, "cid": None})
            continue

        computed = get_computed_properties(cid)
        experimental = get_experimental_properties(cid)

        row = {"iupac_name": name, "cid": cid}
        for prop in COMPUTED_PROPERTIES:
            row[f"{prop.lower()}_pubchem"] = computed.get(prop)
        row.update(experimental)
        rows.append(row)

        print(f"Extracted {name} (CID {cid})")

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"STEP 1 done: saved {len(rows)} rows to {output_csv_path}")


# ---------------------------------------------------------------------------
# STEP 2: comparison
# ---------------------------------------------------------------------------

def compare_with_dataset(extracted_csv_path, dataset_csv_path, output_csv_path):
    """
    STEP 2. Load the PubChem extraction from step 1 and the existing
    dataset CSV, merge them on iupac_name, and write a comparison report.
    Only mw/tpsa/hbd/hba get an automatic match flag (see module docstring
    for why the other columns are left as raw text for manual review).
    """
    extracted_df = pd.read_csv(extracted_csv_path)
    dataset_df = pd.read_csv(dataset_csv_path)

    merged = dataset_df.merge(
        extracted_df, on="iupac_name", how="left", suffixes=("_dataset", "_pubchem_lookup")
    )

    column_pairs = {
        "mw": "molecularweight_pubchem",
        "tpsa": "tpsa_pubchem",
        "hbd": "hbonddonorcount_pubchem",
        "hba": "hbondacceptorcount_pubchem",
    }

    report_rows = []
    for _, row in merged.iterrows():
        report_row = {
            "iupac_name": row["iupac_name"],
            "cid": row.get("cid"),
        }
        for dataset_col, pubchem_col in column_pairs.items():
            dataset_val = row.get(dataset_col)
            pubchem_val = row.get(pubchem_col)
            match = None
            if pd.notna(dataset_val) and pd.notna(pubchem_val):
                tolerance = NUMERIC_MATCH_TOLERANCE[dataset_col]
                match = abs(float(dataset_val) - float(pubchem_val)) <= tolerance
            report_row[f"{dataset_col}_dataset"] = dataset_val
            report_row[f"{dataset_col}_pubchem"] = pubchem_val
            report_row[f"{dataset_col}_match"] = match

        # Raw text fields for manual review, not auto-compared.
        report_row["boiling_point_dataset_c"] = row.get("boiling_point_c")
        report_row["boiling_point_pubchem_raw"] = row.get("boiling_point_pubchem_raw")
        report_row["water_solubility_dataset"] = row.get("water_solubility")
        report_row["water_solubility_pubchem_raw"] = row.get("water_solubility_pubchem_raw")
        report_row["pka_dataset"] = row.get("pka")
        report_row["pkah_dataset"] = row.get("pkah")
        report_row["dissociation_constants_pubchem_raw"] = row.get(
            "dissociation_constants_pubchem_raw"
        )

        report_rows.append(report_row)

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(output_csv_path, index=False)

    mismatches = report_df[
        (report_df["mw_match"] == False)
        | (report_df["tpsa_match"] == False)
        | (report_df["hbd_match"] == False)
        | (report_df["hba_match"] == False)
    ]
    print(f"STEP 2 done: saved comparison report to {output_csv_path}")
    print(f"{len(mismatches)} of {len(report_df)} molecules have at least one mismatch flagged.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset_df = pd.read_csv(DATASET_CSV_PATH)
    molecule_names = dataset_df["iupac_name"].tolist()

    print("=== STEP 1: extracting fresh data from PubChem ===")
    extract_pubchem_data(molecule_names, EXTRACTED_CSV_PATH)

    print("\n=== STEP 2: comparing against functional_group_dataset.csv ===")
    compare_with_dataset(EXTRACTED_CSV_PATH, DATASET_CSV_PATH, COMPARISON_CSV_PATH)
