from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle et le scaler sauvegardés
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'No JSON body provided'}), 400

    # Attendu: clefs 'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'
    try:
        features = [
            float(data['SepalLength']),
            float(data['SepalWidth']),
            float(data['PetalLength']),
            float(data['PetalWidth'])
        ]
    except Exception as e:
        return jsonify({'error': 'Bad input format', 'message': str(e)}), 400

    arr = np.array([features])
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)

    return jsonify({'prediction': str(pred[0])})
@app.route('/', methods=['GET'])
def index():
    html = '''
    <html>
      <head><title>Iris classifier API</title></head>
      <body>
        <h1>Iris classifier API</h1>
        <p>Health endpoint: <a href="/health">/health</a></p>
        <p>Predict endpoint: POST JSON to <code>/predict</code> with keys
           <code>SepalLength, SepalWidth, PetalLength, PetalWidth</code>.</p>
      </body>
    </html>
    '''
    return html, 200


@app.route('/favicon.ico')
def favicon():
    return ('', 204)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
