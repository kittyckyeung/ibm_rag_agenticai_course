from flask import Flask, request, jsonify, render_template
from model import invoke_with_prompts
import time

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    user_message = data.get('message')
    
    system_prompt = "You are an AI assistant helping with customer inquiries. Provide a helpful and concise response."
    
    start_time = time.time()
    
    try:
        result = invoke_with_prompts(system_prompt, user_message)
        result['duration'] = time.time() - start_time
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

