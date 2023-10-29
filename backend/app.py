from flask import Flask, request, jsonify
import torch
import numpy as np
from PIL import Image
import nn_utils as mu

print("Loading ...")
app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Neural Network Explainer from HHI-XAI!"

@app.route('/predict', methods=['POST'])
def predict():
    # Assuming image is passed as a file in the request
    img = Image.open(request.files.get('image')).convert('L')
    img = mu.transform(img)
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        outputs = mu.model(img)
        _, predicted = torch.max(outputs.data, 1)
    return jsonify({"predicted_class": int(predicted[0])})

@app.route('/heatmap', methods=['POST'])
def heatmap():
    idx = request.form.get('idx', type=int)
    hidden_neuron_idx = request.form.get('hidden_neuron_idx', type=int)
    output_neuron_idx = request.form.get('output_neuron_idx', type=int)
    
    heatmap_array = mu.generate_crp_heatmap(mu.model, idx, hidden_neuron_idx, output_neuron_idx)
    
    return jsonify({"heatmap": heatmap_array.tolist()})

if __name__ == '__main__':
    app.run(debug=True)