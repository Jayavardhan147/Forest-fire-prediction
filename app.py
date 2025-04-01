from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("savedmodel.sav", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[f'feature{i}']) for i in range(54)]
    features = np.array(features).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    cover_types = {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas-fir",
        7: "Krummholz"
    }
    result = cover_types.get(prediction, "Unknown")
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)