from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from src.predict import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    filename = None
    if request.method == "POST":
        if "image" not in request.files:
            return "No file uploaded", 400
        file = request.files["image"]
        if file.filename == "":
            return "No file selected", 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        result = predict_image(file_path)
    
    return render_template("index.html", result=result, filename=filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)