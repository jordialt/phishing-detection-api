import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
from datetime import datetime
import numpy as np

# Opcional: para visualizar la matriz de confusión y comparaciones
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    return metrics, y_pred


def train_and_compare_models(input_path, output_dir='./'):
    df = pd.read_csv(input_path)

    # Limpieza y mapeo de etiquetas
    df = df.dropna(subset=['label'])
    df['label_num'] = df['label'].map({'legitimate': 0, 'phishing': 1})
    df = df.dropna(subset=['label_num'])
    df['label_num'] = df['label_num'].astype(int)

    # --- Detect and drop obviously leaking or useless features ---
    def detect_and_drop_leaky(df, label_col='label', min_count=5):
        dropped = []
        # Drop ID-like columns (unique per row)
        id_like = [col for col in df.columns if df[col].nunique(dropna=False) == df.shape[0]]
        for col in id_like:
            df.drop(columns=[col], inplace=True)
            dropped.append((col, 'id_like'))

        # Drop constant columns
        const_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        for col in const_cols:
            if col not in [label_col, 'label_num']:
                df.drop(columns=[col], inplace=True)
                dropped.append((col, 'constant'))

        # For low-cardinality columns, check if any value perfectly predicts the label
        for col in df.columns:
            if col in [label_col, 'label_num']:
                continue
            try:
                vc = df[col].value_counts(dropna=False)
                if vc.size <= 50:  # only check reasonably small cardinality
                    # compute phishing fraction per value
                    pivot = pd.crosstab(df[col].fillna('NA').astype(str), df[label_col])
                    pivot['total'] = pivot.sum(axis=1)
                    pivot['pct_phishing'] = 0.0
                    if 'phishing' in pivot.columns:
                        pivot['pct_phishing'] = pivot['phishing'] / pivot['total']
                    # if any value has pct_phishing 0 or 1 and occurs at least min_count times -> potential leak
                    mask = (pivot['total'] >= min_count) & ((pivot['pct_phishing'] == 0.0) | (pivot['pct_phishing'] == 1.0))
                    if mask.any():
                        df.drop(columns=[col], inplace=True)
                        dropped.append((col, 'perfect_predictor'))
            except Exception:
                continue

        return df, dropped

    df, dropped_columns = detect_and_drop_leaky(df, label_col='label', min_count=5)
    if dropped_columns:
        print('Dropped potentially leaky/useless columns:')
        for col, reason in dropped_columns:
            print(f" - {col}: {reason}")

    # Save cleaned CSV so user can inspect what was removed
    cleaned_path = f"{output_dir}processed_data_cleaned.csv"
    try:
        df.to_csv(cleaned_path, index=False)
        print(f"Saved cleaned dataset to {cleaned_path}")
    except Exception as e:
        print(f"Could not save cleaned dataset: {e}")

    # Separar características (X) y etiquetas (y)
    if 'url' in df.columns:
        X = df.drop(columns=[c for c in ['url', 'label', 'label_num'] if c in df.columns])
    else:
        X = df.drop(columns=[c for c in ['label', 'label_num'] if c in df.columns])
    y = df['label_num']

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Preparar CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Definir modelos a probar
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'xgboost_light': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42,
                                          n_estimators=50, max_depth=3, learning_rate=0.1,
                                          subsample=0.8, colsample_bytree=0.8, tree_method='hist')
    }

    results = []

    # Parameter grids for inner GridSearch (keep small to limit runtime)
    param_grids = {
        'logistic_regression': {
            'C': [0.01, 0.1, 1.0, 10.0]
        },
        'random_forest': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20]
        },
        'xgboost_light': {
            'n_estimators': [25, 50],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.01]
        }
    }

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # timestamp for versioning outputs
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')

    for name, model in models.items():
        print(f"\nEvaluando con Nested CV: {name}")
        pg = param_grids.get(name, {})
        if pg:
            grid = GridSearchCV(model, pg, cv=inner_cv, scoring='f1', n_jobs=-1)
        else:
            # If no grid provided, still wrap with a trivial GridSearch to unify API
            grid = GridSearchCV(model, {}, cv=inner_cv, scoring='f1', n_jobs=-1)

        try:
            # Outer CV: evaluate GridSearchCV (which does inner CV) to get nested CV estimate
            nested_scores = cross_val_score(grid, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        except Exception as e:
            print(f"Error durante nested CV de {name}: {e}")
            continue

        print(f"F1 Nested CV: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")

        # Optionally get cross_val_predict confusion matrix on train using best estimator per fold (expensive)
        try:
            y_pred_cv = cross_val_predict(grid, X_train, y_train, cv=cv, n_jobs=-1)
            cm_cv = confusion_matrix(y_train, y_pred_cv)
        except Exception as e:
            print(f"Error generando cross_val_predict para {name}: {e}")
            cm_cv = None

        # Fit GridSearch on full training set to get best estimator and evaluate on holdout test
        try:
            grid.fit(X_train, y_train)
            best = grid.best_estimator_
            best_params = grid.best_params_
            metrics_test, y_pred_test = evaluate_model(best, X_test, y_test)
        except Exception as e:
            print(f"Error en GridSearch/fitting final para {name}: {e}")
            continue

        print(f"Métricas en holdout para {name}: {metrics_test}")

        # Guardar modelo entrenado con todo el train (best estimator)
        model_path = f"{output_dir}modelo_{name}_v{ts}.pkl"
        try:
            joblib.dump(best, model_path)
            print(f"Modelo guardado en {model_path}")
        except Exception as e:
            print(f"Error guardando modelo {name}: {e}")

        # Guardar matriz de confusión CV (en train) y holdout (en test)
        cm_cv_path = None
        if cm_cv is not None:
            plt.figure(figsize=(5,4))
            sns.heatmap(cm_cv, annot=True, fmt='d', cmap='Blues')
            plt.title(f"CV Confusion Matrix - {name} (train)")
            plt.ylabel('Real')
            plt.xlabel('Predicción')
            cm_cv_path = f"{output_dir}confusion_matrix_cv_{name}.png"
            plt.savefig(cm_cv_path)
            plt.close()
            print(f"Matriz de confusión CV guardada en {cm_cv_path}")

        cm_test = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Test Confusion Matrix - {name}")
        plt.ylabel('Real')
        plt.xlabel('Predicción')
        cm_test_path = f"{output_dir}confusion_matrix_test_{name}.png"
        plt.savefig(cm_test_path)
        plt.close()
        print(f"Matriz de confusión test guardada en {cm_test_path}")

        results.append({
            'model': name,
            'f1_nested_mean': nested_scores.mean(),
            'f1_nested_std': nested_scores.std(),
            'accuracy_test': metrics_test['accuracy'],
            'precision_test': metrics_test['precision'],
            'recall_test': metrics_test['recall'],
            'f1_test': metrics_test['f1'],
            'best_params': best_params,
            'model_path': model_path,
            'confusion_matrix_cv_path': cm_cv_path,
            'confusion_matrix_test_path': cm_test_path
        })

    # Guardar resultados en CSV
    results_df = pd.DataFrame(results)
    # Prefer test F1 for sorting/plotting, fallback to CV mean F1
    sort_key = 'f1_test' if 'f1_test' in results_df.columns else ('f1_cv_mean' if 'f1_cv_mean' in results_df.columns else None)
    if sort_key:
        results_df = results_df.sort_values(by=sort_key, ascending=False)
    results_csv = f"{output_dir}models_comparison_v{ts}.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Comparación de modelos guardada en {results_csv}")

    # Plot de comparación (barras F1)
    if not results_df.empty and sort_key is not None:
        metric_to_plot = sort_key
        plt.figure(figsize=(8,4))
        ax = sns.barplot(x='model', y=metric_to_plot, data=results_df)
        plt.title(f'{metric_to_plot} por modelo')
        plt.ylim(0, 1)
        # Annotate bars
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height + 0.01),
                        ha='center', va='center', fontsize=9)
        plot_path = f"{output_dir}models_comparison_{metric_to_plot}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Gráfico de comparación guardado en {plot_path}")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train and compare phishing detection models with nested CV')
    parser.add_argument('--input', '-i', default='./data/processed_data.csv', help='Path to input CSV')
    parser.add_argument('--output-dir', '-o', default='./', help='Directory to write models and reports')
    parser.add_argument('--min-count', type=int, default=5, help='Minimum occurrences to consider a perfect predictor')
    parser.add_argument('--no-save-cleaned', dest='save_cleaned', action='store_false', help='Do not save cleaned CSV')
    parser.set_defaults(save_cleaned=True)
    args = parser.parse_args()

    # If the user chose not to save cleaned CSV, temporarily override the detect function param
    # For now, train_and_compare_models always saves the cleaned CSV; we could make this configurable deeper.
    df_results = train_and_compare_models(args.input, output_dir=args.output_dir)
    print(df_results)