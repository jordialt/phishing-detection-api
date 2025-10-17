from flask import Flask, request, jsonify
import joblib
import pandas as pd
from urllib.parse import urlparse
import re

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load("modelo_phishing_detector.pkl")

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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data['url']

        # Extraer características
        features = extract_features(url)
        df = pd.DataFrame([features])

        # Hacer la predicción
        prediction = model.predict(df)[0]
        result = 'phishing' if prediction == 1 else 'legitimate'

        return jsonify({'url': url, 'prediction': result})

    except KeyError:
        return jsonify({'error': 'Falta el campo "url" en la solicitud JSON.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ruta de ejemplo para verificar que la API esté corriendo
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'API de detección de phishing está corriendo.'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)