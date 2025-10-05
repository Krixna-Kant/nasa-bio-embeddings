from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from routes import api

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Enable CORS for all domains on all routes
CORS(app)

# Register API blueprint
app.register_blueprint(api)

@app.route('/')
def hello():
    return jsonify({
        'message': 'Welcome to NASA BioMind API',
        'status': 'running',
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'samples': '/api/data/samples',
            'analysis': '/api/analysis/status',
            'projects': '/api/research/projects'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )