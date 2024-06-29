from flask import Flask, request, render_template, jsonify
import random
import numpy as np
from annoy import AnnoyIndex

app = Flask(__name__)

# Simulated song database
songs = [
    {"id": 1, "title": "Happy", "artist": "Pharrell Williams", "mood_vector": [0.9, 0.1, 0.8, 0.2, 0.1]},
    {"id": 2, "title": "Someone Like You", "artist": "Adele", "mood_vector": [0.1, 0.9, 0.2, 0.7, 0.3]},
    {"id": 3, "title": "Eye of the Tiger", "artist": "Survivor", "mood_vector": [0.7, 0.2, 0.9, 0.1, 0.6]},
    {"id": 4, "title": "Weightless", "artist": "Marconi Union", "mood_vector": [0.3, 0.2, 0.1, 0.9, 0.1]},
    {"id": 5, "title": "Break Stuff", "artist": "Limp Bizkit", "mood_vector": [0.5, 0.1, 0.7, 0.1, 0.9]},
    # Add more songs here...
]

# Create and populate the ANNOY index
def create_annoy_index():
    annoy_index = AnnoyIndex(5, 'angular')  # 5 is the length of our mood vectors
    for song in songs:
        annoy_index.add_item(song['id'] - 1, song['mood_vector'])
    annoy_index.build(10)  # 10 trees
    return annoy_index

annoy_index = create_annoy_index()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    mood_vector = request.json['moodVector']
    nearest_ids = annoy_index.get_nns_by_vector(mood_vector, 3)  # Get top 3 recommendations
    recommendations = [songs[i] for i in nearest_ids]
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')