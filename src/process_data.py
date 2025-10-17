import pandas as pd
from urllib.parse import urlparse
import re

def extract_features(url):
    parsed = urlparse(url)
    features = {}

    features['url_length'] = len(url)
    features['domain_length'] = len(parsed.netloc)
    features['nb_dots'] = url.count('.')
    features['nb_slashes'] = url.count('/')
    subdomain = parsed.netloc.split('.')
    features['nb_subdomains'] = len(subdomain) - 2 if len(subdomain) > 2 else 0
    features['path_length'] = len(parsed.path)
    features['nb_underscores'] = url.count('_')
    features['nb_hyphens'] = url.count('-')
    features['nb_at'] = url.count('@')
    features['nb_query_params'] = url.count('?')
    features['nb_fragments'] = url.count('#')

    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    features['is_ip'] = 1 if re.search(ip_pattern, parsed.netloc) else 0

    features['has_port'] = 1 if parsed.port and parsed.port not in [80, 443] else 0
    features['is_https'] = 1 if parsed.scheme == 'https' else 0
    features['in_blacklist'] = 0

    return features

def process_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Separar las columnas originales que no son características
    original_columns = ['url', 'label']

    # Extraer características solo de la columna 'url'
    features_list = []
    for url in df['url']:
        feat = extract_features(url)
        features_list.append(feat)
    features_df = pd.DataFrame(features_list)

    # Combinar las columnas originales con las nuevas características
    df_with_features = pd.concat([df[original_columns], features_df], axis=1)

    # Opcional: Verificar tipos de datos antes de guardar
    print("Tipos de datos en df_with_features:")
    print(df_with_features.dtypes)

    # Guardar el DataFrame procesado
    df_with_features.to_csv(output_path, index=False)
    print(f"Datos procesados y guardados en {output_path}")

if __name__ == "__main__":
    process_data("./data/raw_data.csv", "./data/processed_data.csv")