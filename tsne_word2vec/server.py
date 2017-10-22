from flask import Flask, json, render_template
import numpy as np


EMBEDDING_DIM = 128

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_embedding/<word>')
def get_embedding(word):
    return json.jsonify({
        'word': word,
        'vec': list(np.random.rand(EMBEDDING_DIM))
    })


if __name__ == '__main__':
    app.run(debug=True)
