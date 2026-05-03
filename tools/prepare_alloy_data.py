# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Prepare AlloyGAN dataset from the Collected Alloy Dataset PDF.

Downloads the PDF from the AlloyGAN repository, extracts the alloy table,
parses compositions, temperatures, computes 23 GFA criteria, and outputs
Alloy_train.csv in the format expected by AlloyGAN training.

Usage:
    python tools/prepare_alloy_data.py --output_dir ./data/alloy/

Dependencies:
    pip install pdfplumber requests
"""

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd

PDF_URL = (
    "https://raw.githubusercontent.com/photon-git/AlloyGAN/"
    "main/Collected%20Alloy%20Dataset.pdf"
)

# Top 40 elements by frequency in the dataset (order matters — matches original CSV)
TOP_40_ELEMENTS = [
    "Cu", "Zr", "Al", "Ni", "Ti", "Ag", "Fe", "Mg", "B", "Si",
    "Nb", "Y", "Ca", "La", "Co", "Be", "C", "Mo", "Pd", "P",
    "Sn", "Cr", "Hf", "Zn", "Gd", "Ce", "Er", "Ga", "Au", "Nd",
    "Dy", "W", "Pr", "Ta", "Sc", "Li", "Sm", "S", "Pt", "Mn",
]


def parse_composition(comp_str):
    """Parse a composition string like 'Cu40Zr40Ag10Al10' into element fractions.

    Also handles parenthesized forms like '(Cu50Zr42.5Al7.5)0.99Si1'.
    Returns a dict mapping element symbols to atomic percentages.
    """
    # Handle parenthesized groups: (Group)multiplier + remainder
    paren_match = re.match(
        r'\(([^)]+)\)(\d+\.?\d*)(.*)', comp_str
    )
    if paren_match:
        inner = paren_match.group(1)
        multiplier = float(paren_match.group(2)) if paren_match.group(2) else 1.0
        remainder = paren_match.group(3)
        inner_comp = _parse_elem_pairs(inner)
        if inner_comp is None:
            return None
        inner_sum = sum(inner_comp.values())
        # Two conventions in alloy notation:
        # Type A: inner sums to ~100 (e.g. Cu50Zr43Al7), mult is at% of group
        #   → (Cu50Zr43Al7)98Si2 means Cu=50*98/100=49, Si=2
        # Type B: inner sums to ~1 (e.g. Cu0.6Hf0.25Ti0.15), mult is at% of group
        #   → (Cu0.6Hf0.25Ti0.15)92Nb8 means Cu=0.6*92=55.2, Nb=8
        if inner_sum > 2.0:
            # Type A: inner values are percentages, need to rescale
            result = {k: v * multiplier / inner_sum for k, v in inner_comp.items()}
        else:
            # Type B: inner values are fractions, multiply directly
            result = {k: v * multiplier for k, v in inner_comp.items()}
        if remainder:
            rem_comp = _parse_elem_pairs(remainder)
            if rem_comp:
                for k, v in rem_comp.items():
                    result[k] = result.get(k, 0.0) + v
        return result if result else None

    return _parse_elem_pairs(comp_str)


def _parse_elem_pairs(s):
    """Parse 'Cu40Zr40Ag10Al10' → {'Cu': 40.0, 'Zr': 40.0, ...}."""
    pattern = r"([A-Z][a-z]?)(\d+\.?\d*)"
    matches = re.findall(pattern, s)
    if not matches:
        return None
    result = {}
    for elem, pct in matches:
        result[elem] = result.get(elem, 0.0) + float(pct)
    return result


def compute_gfa_criteria(Tg, Tx, Tl):
    """Compute 23 GFA criteria from Tg, Tx, Tl (all in Kelvin).

    Returns a list of 23 float values in the order matching the original CSV.

    References for each criterion are listed in the AlloyGAN paper Table S1.
    """
    # Guard against division by zero
    eps = 1e-10

    # 1: ΔTx = Tx - Tg  [Inoue 1991]
    delta_Tx = Tx - Tg

    # 2: Trg = Tg / Tl  [Lu 2000]
    Trg = Tg / (Tl + eps)

    # 3: γ = Tx / (Tg + Tl)  [Lu & Liu 2002]
    gamma = Tx / (Tg + Tl + eps)

    # 4: α = Tx / Tl  [Xiao 2004]
    alpha = Tx / (Tl + eps)

    # 5: β = Tg·Tx / Tl²  [Mondal 2005]
    beta = (Tg * Tx) / (Tl**2 + eps)

    # 6: δ = Tx / (Tl - Tg)  [Mondal 2005]
    delta = Tx / (Tl - Tg + eps)

    # 7: φ = Trg·(ΔTx/Tg)^0.143  [Chen 2006]
    phi = Trg * (abs(delta_Tx) / (Tg + eps)) ** 0.143

    # 8: γ_m = (2Tx - Tg) / Tl  [Du 2007]
    gamma_m = (2 * Tx - Tg) / (Tl + eps)

    # 9: ω = Tg/Tx - 2Tg/(Tg+Tl)  [Fan 2007]
    omega = Tg / (Tx + eps) - 2 * Tg / (Tg + Tl + eps)

    # 10: ξ = Tg / (2Tl - Tx)  [Du 2008]
    xi = Tg / (2 * Tl - Tx + eps)

    # 11: Kgl = Tg·Tx / ((Tl-Tx)·(Tl-Tg))  [Yuan 2008]
    Kgl = (Tg * Tx) / ((Tl - Tx) * (Tl - Tg) + eps)

    # 12: θ = (Tg + Tx) / Tl  [Long 2009]
    theta = (Tg + Tx) / (Tl + eps)

    # 13: α₁ = Tx·Tg / (Tl·(Tl-Tg))  [Ji 2009]
    alpha1 = (Tx * Tg) / (Tl * (Tl - Tg) + eps)

    # 14: σ = (Tx - Tg) / (Tl - Tx)  [Zhang 2009]
    sigma = (Tx - Tg) / (Tl - Tx + eps)

    # 15: ω₃ = [Tg/(Tg+Tl)]·[Tx/(Tg+Tl)]  [An/Hongqing 2009]
    omega3 = (Tg / (Tg + Tl + eps)) * (Tx / (Tg + Tl + eps))

    # 16: η = (Tg + Tx) / (Tl + Tx)  [Guo 2010]
    eta = (Tg + Tx) / (Tl + Tx + eps)

    # 17: χ = [(ΔTx)·Tg / (Tl-Tg)²]^0.143  [Dong 2011]
    chi = (abs(delta_Tx) * Tg / ((Tl - Tg) ** 2 + eps)) ** 0.143

    # 18: ε = ΔTx/(Tl−Tg)·Tg/Tl  [Błyskun 2015]
    epsilon = (delta_Tx / (Tl - Tg + eps)) * (Tg / (Tl + eps))

    # 19: D = Tx/(Tl-Tg) + Tg/Tl  [Tripathi 2016 — GP-derived]
    D = Tx / (Tl - Tg + eps) + Tg / (Tl + eps)

    # 20: ψ = (Tx-Tg)·Tg / (Tl-Tx)² + Tg/Tl  [Long 2018]
    psi = (Tx - Tg) * Tg / ((Tl - Tx) ** 2 + eps) + Tg / (Tl + eps)

    # 21: Xiong 2019 — ML-derived: Tg²/(Tg+Tl)·1/Tl
    xiong = (Tg**2) / ((Tg + Tl) * (Tl + eps) + eps)

    # 22: Deng 2020 — Tx·Tg²/(Tl²·(Tl-Tg))
    deng = (Tx * Tg**2) / (Tl**2 * (Tl - Tg) + eps)

    # 23: Ren 2021 — (Tx-Tg)/(Tl-Tg) + Tg·Tx/Tl²
    ren = (Tx - Tg) / (Tl - Tg + eps) + (Tg * Tx) / (Tl**2 + eps)

    return [
        delta_Tx, Trg, gamma, alpha, beta, delta, phi, gamma_m,
        omega, xi, Kgl, theta, alpha1, sigma, omega3, eta,
        chi, epsilon, D, psi, xiong, deng, ren,
    ]


def extract_table_from_pdf(pdf_path):
    """Extract alloy composition table from the PDF using pdfplumber.

    The PDF has a two-column text layout where each data line looks like:
        Ca40Cu30Mg30 395 430 694 Ca60Al30Ag10 483 531 868
    i.e. [Comp1 Tg1 Tx1 Tl1] [Comp2 Tg2 Tx2 Tl2]

    We parse the raw text line-by-line using regex.

    Returns a list of dicts with keys: composition, Tg, Tx, Tl.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF extraction. "
            "Install with: pip install pdfplumber"
        )

    # Pattern: composition followed by 3 numbers (Tg, Tx, Tl)
    # Composition: starts with uppercase letter, contains element-number pairs
    # Handle parenthesized compositions like (Cu50Zr42.5Al7.5)0.99Si1
    entry_pattern = re.compile(
        r'((?:\([A-Z][A-Za-z0-9.]+\)\d*\.?\d*)?'  # optional parenthesized part
        r'[A-Z][a-z]?\d[\w.]*)'                     # main composition
        r'\s+'
        r'(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)'  # Tg Tx Tl
    )

    entries = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split('\n'):
                # Skip headers and category labels
                if 'Composition' in line or 'TABLE' in line:
                    continue
                if 'Supplementary' in line or 'based' in line:
                    continue
                if line.strip() in ('g x l', ''):
                    continue

                # Find all composition+temperature matches in the line
                for m in entry_pattern.finditer(line):
                    comp_str = m.group(1)
                    tg = float(m.group(2))
                    tx = float(m.group(3))
                    tl = float(m.group(4))

                    # Basic sanity: temperatures should be > 0
                    if tg > 0 and tx > 0 and tl > 0:
                        entries.append({
                            "composition": comp_str,
                            "Tg": tg,
                            "Tx": tx,
                            "Tl": tl,
                        })

    print(f"Extracted {len(entries)} entries from PDF")
    return entries


def build_csv(entries, output_path):
    """Build Alloy_train.csv from extracted entries.

    Columns: 40 element fractions + Tg + Tx + Tl + 23 GFA criteria + source
    """
    rows = []
    skipped = 0

    for entry in entries:
        comp = parse_composition(entry["composition"])
        if comp is None:
            skipped += 1
            continue

        Tg = float(entry["Tg"])
        Tx = float(entry["Tx"])
        Tl = float(entry["Tl"])

        # Sanity checks
        comp_sum = sum(comp.values())
        if comp_sum > 105 or comp_sum < 10:
            skipped += 1
            continue
        if Tg <= 0 or Tx <= 0 or Tl <= 0:
            skipped += 1
            continue
        if Tg >= Tl or Tx >= Tl:
            warnings.warn(
                f"Skipping entry with Tg={Tg} >= Tl={Tl} or Tx={Tx} >= Tl={Tl}: "
                f"{entry['composition']}"
            )
            skipped += 1
            continue

        # Element fractions (40 columns)
        elem_fracs = [comp.get(e, 0.0) for e in TOP_40_ELEMENTS]

        # GFA criteria (23 columns)
        gfa = compute_gfa_criteria(Tg, Tx, Tl)

        row = elem_fracs + [Tg, Tx, Tl] + gfa + ["literature"]
        rows.append(row)

    # Column names
    col_names = (
        TOP_40_ELEMENTS
        + ["Tg", "Tx", "Tl"]
        + [f"GFA_{i + 1}" for i in range(23)]
        + ["source"]
    )

    df = pd.DataFrame(rows, columns=col_names)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} entries to {output_path} (skipped {skipped})")
    print(f"Shape: {df.shape}")

    # Print summary statistics
    print("\nElement frequency (non-zero entries):")
    for elem in TOP_40_ELEMENTS[:10]:
        count = (df[elem] > 0).sum()
        print(f"  {elem}: {count} ({100 * count / len(df):.1f}%)")

    return df


def download_pdf(output_path):
    """Download the Collected Alloy Dataset PDF from GitHub."""
    import requests

    print(f"Downloading PDF from {PDF_URL}...")
    resp = requests.get(PDF_URL, timeout=60)
    resp.raise_for_status()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(resp.content)
    print(f"Downloaded PDF to {output_path} ({len(resp.content)} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare AlloyGAN dataset from PDF"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/alloy/",
        help="Output directory for Alloy_train.csv",
    )
    parser.add_argument(
        "--pdf_path",
        type=str,
        default=None,
        help="Path to existing PDF (downloads if not specified)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Filter by alloy categories (e.g., Cu Fe Ti Zr). "
             "Default: use all entries.",
    )
    args = parser.parse_args()

    pdf_path = args.pdf_path
    if pdf_path is None:
        pdf_path = os.path.join(args.output_dir, "Collected_Alloy_Dataset.pdf")
        if not os.path.exists(pdf_path):
            download_pdf(pdf_path)
        else:
            print(f"Using existing PDF: {pdf_path}")

    entries = extract_table_from_pdf(pdf_path)

    if len(entries) == 0:
        print(
            "\nERROR: No entries extracted from PDF. "
            "The PDF table format may have changed.\n"
            "Please manually create Alloy_train.csv with columns:\n"
            f"  {', '.join(TOP_40_ELEMENTS[:5])}... (40 elements), "
            "Tg, Tx, Tl, GFA_1..GFA_23, source"
        )
        return

    output_path = os.path.join(args.output_dir, "Alloy_train.csv")
    df = build_csv(entries, output_path)

    if args.categories:
        # Filter to specified categories based on dominant element
        def get_dominant_element(row):
            elem_vals = {e: row[e] for e in TOP_40_ELEMENTS}
            return max(elem_vals, key=elem_vals.get)

        df["category"] = df.apply(get_dominant_element, axis=1)
        mask = df["category"].isin(args.categories)
        df_filtered = df[mask].drop(columns=["category"])

        filtered_path = os.path.join(
            args.output_dir,
            f"Alloy_train_{'_'.join(args.categories)}.csv",
        )
        df_filtered.to_csv(filtered_path, index=False)
        print(
            f"\nFiltered to {args.categories}: {len(df_filtered)} entries"
            f" → {filtered_path}"
        )


if __name__ == "__main__":
    main()
