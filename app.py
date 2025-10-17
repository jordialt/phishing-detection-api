from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import joblib
import glob
import pandas as pd
from urllib.parse import urlparse
import re
import os

app = Flask(__name__)

app = Flask(__name__)
CORS(app)

# Load models (auto-detect latest versioned files)
MODELS = {}
MODEL_FILES = {
    'logistic_regression': 'modelo_logistic_regression.pkl',
    'random_forest': 'modelo_random_forest.pkl',
    'xgboost_light': 'modelo_xgboost_light.pkl'
}

def _latest_model(base_name):
    # Look for versioned files like modelo_name_vYYYYMMDDTHHMMSS.pkl
    stem = base_name.replace('.pkl', '')
    pattern = f"{stem}_v*.pkl"
    files = glob.glob(pattern)
    if not files:
        # fallback to non-versioned filename if exists
        if os.path.exists(base_name):
            return base_name
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

for name, base in MODEL_FILES.items():
    candidate = _latest_model(base)
    if candidate:
        try:
            MODELS[name] = joblib.load(candidate)
            print(f"Modelo '{name}' cargado desde {candidate}")
        except Exception as e:
            print(f"Error cargando modelo {name} desde {candidate}: {e}")
            MODELS[name] = None
    else:
        print(f"Aviso: archivo de modelo {base} no encontrado (ni versionado). {name} no estará disponible.")
        MODELS[name] = None

# Determine expected feature columns from cleaned dataset if available
EXPECTED_FEATURES = None
cleaned_path = 'processed_data_cleaned.csv'
fallback_path = 'data/processed_data.csv'
features_source = None
for p in (cleaned_path, fallback_path):
    if os.path.exists(p):
        try:
            df_sample = pd.read_csv(p, nrows=5)
            # Drop label columns if present
            cols = [c for c in df_sample.columns if c not in ('label', 'label_num')]
            EXPECTED_FEATURES = cols
            features_source = p
            print(f"Using feature columns from {p}: {EXPECTED_FEATURES}")
            break
        except Exception as e:
            print(f"Could not read {p} to infer features: {e}")

if EXPECTED_FEATURES is None:
    print('Warning: Could not infer expected feature columns. Predictions may fail if model expects specific features.')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'Falta el campo "url" en la solicitud JSON.'}), 400

        url = data['url']
        model_name = data.get('model', 'xgboost_light')

        if model_name not in MODELS:
            return jsonify({'error': f"Modelo '{model_name}' no disponible."}), 400

        # Extraer características
        features = extract_features(url)
        df = pd.DataFrame([features])
        # Reindex to expected features (fill missing with 0)
        if EXPECTED_FEATURES is not None:
            df = df.reindex(columns=EXPECTED_FEATURES, fill_value=0)

        # Hacer la predicción
        prediction = MODELS[model_name].predict(df)[0]
        result = 'phishing' if int(prediction) == 1 else 'legitimate'

        return jsonify({'url': url, 'model': model_name, 'prediction': result})

    except KeyError:
        return jsonify({'error': 'Falta el campo "url" en la solicitud JSON.'}), 400
    except Exception as e:
        print(f"ERROR en /predict: {e}")
        return jsonify({'error': f'Error interno al procesar la predicción: {str(e)}'}), 500


@app.route('/predict_proba', methods=['POST'])
def predict_proba():
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'Falta el campo "url" en la solicitud JSON.'}), 400

        url = data['url']
        model_name = data.get('model', 'xgboost_light')

        if model_name not in MODELS:
            return jsonify({'error': f"Modelo '{model_name}' no disponible."}), 400

        model = MODELS[model_name]

        # Extraer características
        features = extract_features(url)
        df = pd.DataFrame([features])
        # Reindex to expected features (fill missing with 0)
        if EXPECTED_FEATURES is not None:
            df = df.reindex(columns=EXPECTED_FEATURES, fill_value=0)

        # Comprobar si el modelo soporta predict_proba
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(df)[0]
            # assuming binary labels [0,1]
            return jsonify({'url': url, 'model': model_name, 'probabilities': {'legitimate': float(probs[0]), 'phishing': float(probs[1])}})
        else:
            return jsonify({'error': 'El modelo no soporta predict_proba.'}), 400

    except Exception as e:
        print(f"ERROR en /predict_proba: {e}")
        return jsonify({'error': f'Error interno al procesar la predicción de probabilidades: {str(e)}'}), 500


@app.route('/explain', methods=['POST'])
def explain():
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'Falta el campo "url" en la solicitud JSON.'}), 400

        url = data['url']
        model_name = data.get('model', 'xgboost_light')
        if model_name not in MODELS:
            return jsonify({'error': f"Modelo '{model_name}' no disponible."}), 400

        model = MODELS[model_name]
        features = extract_features(url)
        df = pd.DataFrame([features])
        if EXPECTED_FEATURES is not None:
            df = df.reindex(columns=EXPECTED_FEATURES, fill_value=0)

        explanation = {}
        # For tree-based models, show feature_importances_
        try:
            if hasattr(model, 'feature_importances_'):
                fi = model.feature_importances_
                idx = fi.argsort()[::-1][:10]
                items = []
                for i in idx:
                    if fi[i] > 0:
                        items.append({'feature': EXPECTED_FEATURES[i], 'importance': float(fi[i]), 'value': float(df.iloc[0, i])})
                explanation['top_features'] = items
            elif hasattr(model, 'coef_'):
                coef = model.coef_[0]
                idx = coef.argsort()[::-1][:10]
                items = []
                for i in idx:
                    items.append({'feature': EXPECTED_FEATURES[i], 'coefficient': float(coef[i]), 'value': float(df.iloc[0, i])})
                explanation['top_features'] = items
            else:
                explanation['message'] = 'Modelo no soporta explicación por importancia de características.'
        except Exception as e:
            explanation['error'] = f'No se pudo generar explicación: {e}'

        return jsonify({'url': url, 'model': model_name, 'explanation': explanation})

    except Exception as e:
        print(f"ERROR en /explain: {e}")
        return jsonify({'error': f'Error interno al generar explicación: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def home():
        # Simple HTML UI to test the models
        html = '''
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Phishing Detector — Test</title>
        </head>
        <body>
            <h2>Phishing Detector — Test UI</h2>
            <label for="url">URL:</label>
            <input type="text" id="url" size="80" placeholder="https://example.com/..." />
            <br/>
            <label for="model">Model:</label>
            <select id="model">
                <option value="xgboost_light">xgboost_light</option>
                <option value="random_forest">random_forest</option>
                <option value="logistic_regression">logistic_regression</option>
            </select>
                    <button id="predict">Predict</button>
                    <button id="predict_proba">Predict Probabilities</button>
                    <button id="explain">Explain</button>
                    <div id="result" style="white-space:pre-wrap; border:1px solid #ddd; padding:8px; margin-top:8px;"></div>
                    <div id="probs" style="margin-top:8px"></div>
                    <div id="explanation" style="margin-top:12px; border:1px solid #eee; padding:8px;"></div>
            <script>
            document.getElementById('predict').onclick = async function(){
                const url = document.getElementById('url').value;
                const model = document.getElementById('model').value;
                const res = await fetch('/predict', {
                    method: 'POST', headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({url: url, model: model})
                });
                const data = await res.json();
                document.getElementById('result').textContent = JSON.stringify(data, null, 2);
            }
                    document.getElementById('predict_proba').onclick = async function(){
                        const url = document.getElementById('url').value;
                        const model = document.getElementById('model').value;
                        const res = await fetch('/predict_proba', {
                            method: 'POST', headers: {'Content-Type':'application/json'},
                            body: JSON.stringify({url: url, model: model})
                        });
                        const data = await res.json();
                                document.getElementById('result').textContent = JSON.stringify(data, null, 2);
                                // show nice probability bars if available
                                if (data.probabilities) {
                                    const p = data.probabilities;
                                    document.getElementById('probs').innerHTML = `
                                        <div>Legitimate: <div style="background:#e6f7ff; width:300px;"><div style="background:#4caf50; width:${(p.legitimate*100)}%; color:white; padding:4px;">${(p.legitimate*100).toFixed(1)}%</div></div></div>
                                        <div>Phishing: <div style="background:#fff0f0; width:300px;"><div style="background:#f44336; width:${(p.phishing*100)}%; color:white; padding:4px;">${(p.phishing*100).toFixed(1)}%</div></div></div>
                                    `;
                                }
                    }
                            document.getElementById('explain').onclick = async function(){
                                const url = document.getElementById('url').value;
                                const model = document.getElementById('model').value;
                                const res = await fetch('/explain', {
                                    method: 'POST', headers: {'Content-Type':'application/json'},
                                    body: JSON.stringify({url: url, model: model})
                                });
                                const data = await res.json();
                                if (data.explanation && data.explanation.top_features) {
                                    let html = '<h4>Top features</h4><ul>';
                                    data.explanation.top_features.slice(0,5).forEach(f => {
                                        html += `<li>${f.feature}: value=${f.value} importance=${(f.importance||f.coefficient||0).toFixed(4)}</li>`;
                                    });
                                    html += '</ul>';
                                    document.getElementById('explanation').innerHTML = html;
                                } else {
                                    document.getElementById('explanation').textContent = JSON.stringify(data, null, 2);
                                }
                            }
            </script>
        </body>
        </html>
        '''
        return render_template_string(html)

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

if __name__ == '__main__':
    # En Render, Gunicorn se encarga de ejecutar la app, no esta parte
    # pero es útil para pruebas locales
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
