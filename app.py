from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)
CORS(app)

df = pd.read_csv(r"D:\Documents\archive\Cleaned_Indian_Food_Dataset.csv")

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','), token_pattern=None)
vectorizer.fit(df['Cleaned-Ingredients'])
label_encoder = LabelEncoder()
label_encoder.fit(df['TranslatedRecipeName'])

class RecipeModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RecipeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

input_size = len(vectorizer.get_feature_names_out())
hidden_size = 128
num_classes = len(label_encoder.classes_)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RecipeModel(input_size, hidden_size, num_classes).to(device)
model.load_state_dict(torch.load('recipe_model.pth', map_location=device, weights_only=True))
model.eval()

def predict_recipes(ingredients_str):
    ingredients_vec = vectorizer.transform([ingredients_str]).toarray()
    ingredients_tensor = torch.tensor(ingredients_vec, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(ingredients_tensor)
    
    probabilities = nn.functional.softmax(output, dim=1)
    top_predictions = torch.topk(probabilities, k=5)  
    predicted_indices = top_predictions.indices.cpu().numpy().flatten()
    
    return label_encoder.inverse_transform(predicted_indices)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ingredients = data.get('ingredients', '')
    if not ingredients:
        return jsonify({'error': 'No ingredients provided'}), 400
    
    predicted_recipes = predict_recipes(ingredients)
    return jsonify({'predicted_recipes': predicted_recipes.tolist()})

if __name__ == '__main__':
    app.run(debug=True,port=5001)
