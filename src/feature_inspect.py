import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier


def inspect_features(path, cv_splits=5):
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        print('No label column found')
        return

    y = df['label'].map({'legitimate': 0, 'phishing': 1})
    report = []

    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    for col in df.columns:
        if col == 'label':
            continue
        series = df[col]
        nunique = series.nunique(dropna=False)
        dtype = series.dtype
        print('\n---')
        print(f"Feature: {col} (dtype={dtype}, unique={nunique})")

        # Show value counts for categorical/binary features
        try:
            vc = series.value_counts(dropna=False).head(10)
            print('Top value counts:')
            print(vc.to_string())
        except Exception as e:
            print(f'Could not compute value_counts: {e}')

        # Compute simple single-feature predictive power using a decision stump
        try:
            X_col = series.copy()
            # Convert non-numeric to category codes
            if not np.issubdtype(X_col.dtype, np.number):
                X_col = X_col.astype('category').cat.codes.replace({-1: np.nan})
            # Fill na with -9999
            X_col = X_col.fillna(-9999).values.reshape(-1, 1)
            clf = DecisionTreeClassifier(max_depth=1, random_state=42)
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X_col, y, cv=cv, scoring='f1', n_jobs=-1)
            print(f'Single-feature CV F1 (decision stump): mean={scores.mean():.4f}, std={scores.std():.4f}')
        except Exception as e:
            print(f'Error computing single-feature score: {e}')

        # For binary/categorical features, show label distribution per value
        if nunique <= 10:
            try:
                pivot = pd.crosstab(series.fillna('NA').astype(str), df['label'])
                pivot['total'] = pivot.sum(axis=1)
                pivot['pct_phishing'] = pivot.get('phishing', 0) / pivot['total']
                print('Label distribution per feature value:')
                print(pivot.sort_values('total', ascending=False).to_string())
            except Exception as e:
                print(f'Error computing pivot table: {e}')

    print('\nInspection complete')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python src/feature_inspect.py <path_to_csv> [cv_splits]')
    else:
        splits = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        inspect_features(sys.argv[1], cv_splits=splits)
