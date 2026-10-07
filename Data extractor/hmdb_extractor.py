"""
HMDB local extractor for functional_group_dataset.csv.

HMDB has no self-serve public API (access requires emailing the HMDB team
directly - see https://www.hmdb.ca/simple/api - with no published turnaround
time). The practical alternative is their full bulk XML download, free for
academic/non-commercial use with citation (CC BY-NC 4.0):

    https://hmdb.ca/downloads  ->  download "All Metabolites" (hmdb_metabolites.xml)

That file covers ~220,000 human-relevant metabolites and can be several GB,
so it must be downloaded and run on your own machine - this sandbox has no
network access to hmdb.ca.

Two-phase workflow (run in order):

  PHASE 1 - inspect_schema():
      Streams just the FIRST record in the XML and prints its field names
      and values. This project has never actually parsed this file, so the
      exact tag names for boiling point / solubility / pKa are unconfirmed -
      run this first and read the printed tags before trusting phase 2.

  PHASE 2 - extract_matches():
      Streams through the whole file (using iterparse so multi-GB files
      don't need to fit in memory), keeps only records whose <name> or
      <synonym> matches one of your 44 molecules, and writes whatever
      physical/predicted property fields it finds to a CSV.

Coverage caveat: HMDB is scoped to compounds observed or expected in the
human body (including some dietary/environmental exposures), not a general
industrial-chemical database like PubChem. Not all 44 molecules in your
dataset are guaranteed to have an HMDB entry.

pKa caveat: if present, HMDB's pKa values are typically ChemAxon-predicted,
same site-ambiguity issue already flagged for PubChem/CompTox - useful as a
third cross-check point, not a replacement for the Evans/Reich lookup.
"""

import csv
import xml.etree.ElementTree as ET

HMDB_XML_PATH = "hmdb_metabolites.xml"  # path to the file you downloaded
DATASET_CSV_PATH = "functional_group_dataset.csv"
OUTPUT_CSV_PATH = "hmdb_matches_report.csv"

# HMDB XML uses a default namespace; this must match the file's root tag.
# inspect_schema() will confirm the real one - update if it differs.
NAMESPACE = {"hmdb": "http://www.hmdb.ca"}


def inspect_schema(xml_path, max_records=1):
    """
    PHASE 1. Print the raw field names/values of the first record so we can
    confirm the real tag names before writing extraction logic against them.
    """
    count = 0
    for event, elem in ET.iterparse(xml_path, events=("end",)):
        tag = elem.tag.split("}")[-1]  # strip namespace prefix for readability
        if tag == "metabolite":
            print(f"=== Record {count + 1} ===")
            for child in elem:
                child_tag = child.tag.split("}")[-1]
                if len(child) == 0:
                    print(f"{child_tag}: {child.text}")
                else:
                    print(f"{child_tag}: <nested, {len(child)} sub-elements>")
            count += 1
            elem.clear()
            if count >= max_records:
                break


def inspect_properties_detail(xml_path, max_records=1):
    """
    PHASE 1b. inspect_schema() only prints one level deep, so nested blocks
    like experimental_properties/predicted_properties showed up as
    "<nested, N sub-elements>" with no visible tag names. This drills one
    level further into just those two blocks so we can see the real
    property/kind/value (or whatever they're actually called) structure
    before extract_matches() is written against them.
    """
    count = 0
    for event, elem in ET.iterparse(xml_path, events=("end",)):
        tag = elem.tag.split("}")[-1]
        if tag != "metabolite":
            continue

        print(f"=== Record {count + 1}: {elem.findtext('hmdb:name', namespaces=NAMESPACE)} ===")
        for section_name in ("experimental_properties", "predicted_properties"):
            section = elem.find(f"hmdb:{section_name}", NAMESPACE)
            print(f"\n--- {section_name} ---")
            if section is None:
                print("  (section not found)")
                continue
            for entry in section:
                entry_tag = entry.tag.split("}")[-1]
                print(f"  <{entry_tag}>")
                for field in entry:
                    field_tag = field.tag.split("}")[-1]
                    print(f"    {field_tag}: {field.text}")

        count += 1
        elem.clear()
        if count >= max_records:
            break


def extract_matches(xml_path, dataset_csv_path, output_csv_path):
    """
    PHASE 2. Confirmed field structure (from actual inspect_properties_detail
    output, not guessed): each <experimental_properties> or
    <predicted_properties> block holds repeated <property> elements, each
    with <kind>, <value>, <source> children. Relevant kind values seen:

      experimental_properties (literature-sourced, coverage varies a lot
      per molecule - boiling_point was ABSENT on the sample record):
        - boiling_point
        - melting_point
        - water_solubility  (quantitative, e.g. "200 g/kg" - different
          format from your qualitative miscible/high/moderate/low/very low)

      predicted_properties (ChemAxon/ALOGPS models, present for nearly
      every record):
        - pka_strongest_acidic   -> conceptually aligns with your 'pka'
        - pka_strongest_basic    -> conceptually aligns with your 'pkah'
        - polar_surface_area     -> cross-check for 'tpsa'
        - donor_count            -> cross-check for 'hbd'
        - acceptor_count         -> cross-check for 'hba'
        - average_mass           -> cross-check for 'mw'

    Even though pka_strongest_acidic/basic map more sensibly here than in
    CompTox's generic fields, they are still a ChemAxon ALGORITHM's choice
    of site, not a hand-picked site matching your Data Dictionary's
    per-family definition. Treat as a third cross-check, not ground truth.
    """
    import pandas as pd

    dataset_df = pd.read_csv(dataset_csv_path)
    target_names = set(dataset_df["iupac_name"].str.lower())
    target_names |= set(dataset_df["common_name"].str.lower())

    EXPERIMENTAL_KINDS = ["boiling_point", "melting_point", "water_solubility"]
    PREDICTED_KINDS = [
        "pka_strongest_acidic",
        "pka_strongest_basic",
        "polar_surface_area",
        "donor_count",
        "acceptor_count",
        "average_mass",
    ]

    matches = []
    for event, elem in ET.iterparse(xml_path, events=("end",)):
        tag = elem.tag.split("}")[-1]
        if tag != "metabolite":
            continue

        name_elem = elem.find("hmdb:name", NAMESPACE)
        name = name_elem.text if name_elem is not None else None

        synonyms = [
            s.text for s in elem.findall("hmdb:synonyms/hmdb:synonym", NAMESPACE)
        ]
        all_names = {n.lower() for n in ([name] + synonyms) if n}

        if all_names & target_names:
            accession_elem = elem.find("hmdb:accession", NAMESPACE)
            record = {
                "hmdb_matched_name": name,
                "hmdb_accession": accession_elem.text if accession_elem is not None else None,
            }
            for kind in EXPERIMENTAL_KINDS + PREDICTED_KINDS:
                record[f"hmdb_{kind}"] = None

            for section_name in ("experimental_properties", "predicted_properties"):
                section = elem.find(f"hmdb:{section_name}", NAMESPACE)
                if section is None:
                    continue
                for prop in section.findall("hmdb:property", NAMESPACE):
                    kind_elem = prop.find("hmdb:kind", NAMESPACE)
                    value_elem = prop.find("hmdb:value", NAMESPACE)
                    if kind_elem is None or value_elem is None:
                        continue
                    if kind_elem.text in EXPERIMENTAL_KINDS + PREDICTED_KINDS:
                        record[f"hmdb_{kind_elem.text}"] = value_elem.text

            matches.append(record)

        elem.clear()

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        if matches:
            fieldnames = list(matches[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matches)

    print(f"Found {len(matches)} matching HMDB records out of 44 target molecules.")
    print(f"Saved to {output_csv_path}")


if __name__ == "__main__":
    extract_matches(HMDB_XML_PATH, DATASET_CSV_PATH, OUTPUT_CSV_PATH)
