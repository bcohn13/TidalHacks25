from flask import Flask, request, jsonify, render_template
import google.generativeai as genai

# ✅ Replace with your actual API key
genai.configure(api_key="AIzaSyAIelnAg8MwqCDQ1LwYwvcW_aJon29vE2g")

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "")

        # ✅ Explicitly use the right model name
        model = genai.GenerativeModel(model_name="models/gemini-pro")

        # ✅ Use a valid generation method
        response = model.generate_content(
            user_input,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 256,
                "top_p": 1
            }
        )

        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
