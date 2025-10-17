import pandas as pd
import numpy as np


def diagnose(path):
    df = pd.read_csv(path)
    report = []

    # Basic info
    report.append(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Duplicates
    dup_rows = df.duplicated().sum()
    report.append(f"Duplicate rows (exact duplicates): {dup_rows}")

    # Check duplicates between features and label
    if 'label' in df.columns:
        # Find columns that equal the label (string-wise)
        equal_label_cols = []
        for col in df.columns:
            if col == 'label':
                continue
            try:
                if df[col].astype(str).equals(df['label'].astype(str)):
                    equal_label_cols.append(col)
            except Exception:
                pass
        report.append(f"Columns exactly equal to label: {equal_label_cols}")

    # Columns with unique values equal to number of rows (likely IDs)
    id_like = [col for col in df.columns if df[col].nunique() == df.shape[0]]
    report.append(f"ID-like columns (unique values == rows): {id_like}")

    # Columns with very low cardinality (e.g., constant)
    low_card = [col for col in df.columns if df[col].nunique() <= 2]
    report.append(f"Low-cardinality columns (<=2 unique values): {low_card}")

    # Numerical perfect correlation with label
    if 'label' in df.columns:
        # create numeric label if possible
        try:
            label_num = df['label'].map({'legitimate': 0, 'phishing': 1})
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            perfect_corr = []
            for col in numeric_cols:
                if df[col].isna().any():
                    continue
                corr = np.corrcoef(df[col].astype(float), label_num.astype(float))[0,1]
                if np.isclose(corr, 1.0) or np.isclose(corr, -1.0):
                    perfect_corr.append((col, corr))
            report.append(f"Numerical columns perfectly correlated with label: {perfect_corr}")
        except Exception as e:
            report.append(f"Could not compute numeric correlations: {e}")

    # Check for features created from URL (if 'url' exists) - exact matches
    if 'url' in df.columns and 'label' in df.columns:
        # If URL string encodes label (e.g., contains 'phish' or similar), check simplistic heuristics
        url_contains_label = df['url'].astype(str).str.contains('phish|phishing|malicious', case=False, na=False).sum()
        report.append(f"URLs containing 'phish'/'phishing'/'malicious' (heuristic): {url_contains_label}")

    # Print report
    print('\n'.join(report))


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python src/diagnose_leakage.py <path_to_csv>')
    else:
        diagnose(sys.argv[1])
