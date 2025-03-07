# src/app.py
import os
import sys
from flask import Flask, render_template, request, send_file, url_for, abort
from werkzeug.utils import secure_filename
from predict import predict_crop
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
from datetime import datetime
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a')
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = Flask(__name__, static_url_path='/static', static_folder='static')
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Created upload folder: {UPLOAD_FOLDER}")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    filename = None
    error = None

    if request.method == "POST":
        if 'image' not in request.files:
            error = "No file uploaded"
            logger.error("No file part in request")
            return render_template("index.html", error=error)
        
        file = request.files['image']
        if file.filename == '':
            error = "No file selected"
            logger.error("No file selected")
            return render_template("index.html", error=error)
        
        if not allowed_file(file.filename):
            error = "Invalid file type. Please upload a PNG, JPG, or JPEG image."
            logger.error(f"Invalid file type: {file.filename}")
            return render_template("index.html", error=error)

        # Get rainfall value from form (default to 0.0 if not provided)
        try:
            rainfall_value = float(request.form.get('rainfall', 0.0))
        except ValueError:
            rainfall_value = 0.0
            logger.warning("Invalid rainfall value provided, defaulting to 0.0")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        absolute_path = os.path.abspath(file_path)
        logger.info(f"Attempting to save file to: {absolute_path}")

        try:
            counter = 1
            base_name = os.path.splitext(filename)[0]
            extension = os.path.splitext(filename)[1]
            while os.path.exists(file_path):
                filename = f"{base_name}_{counter}{extension}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                counter += 1
            file.save(file_path)
            if os.path.exists(file_path):
                logger.info(f"File saved successfully: {absolute_path}")
            else:
                logger.error(f"File not found after save: {absolute_path}")
                error = "Failed to save the uploaded image"
                return render_template("index.html", error=error)
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            error = f"Error saving file: {str(e)}"
            return render_template("index.html", error=error)

        try:
            result = predict_crop(file_path, rainfall_value)
            if 'error' in result:
                error = result['error']
                result = None
                logger.error(f"Prediction error: {error}")
        except Exception as e:
            error = f"Error analyzing image: {str(e)}"
            logger.error(f"Prediction error: {str(e)}")
            result = None

    return render_template("index.html", result=result, filename=filename, error=error)

@app.route("/download-report/<filename>")
def download_report(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found for report: {file_path}")
            abort(404)
        
        # Get rainfall value from query parameter (default to 0.0 if not provided)
        try:
            rainfall_value = float(request.args.get('rainfall', 0.0))
        except ValueError:
            rainfall_value = 0.0
            logger.warning("Invalid rainfall value in report request, defaulting to 0.0")

        result = predict_crop(file_path, rainfall_value)
        if 'error' in result:
            logger.error(f"Prediction error in report: {result['error']}")
            abort(500, description=result['error'])

        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        styles = getSampleStyleSheet()
        elements = []

        title_style = ParagraphStyle(
            name='TitleStyle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.darkgreen
        )
        elements.append(Paragraph("Crop Leaf Analysis Report", title_style))
        elements.append(Spacer(1, 12))

        try:
            img = Image(file_path, width=200, height=200)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Spacer(1, 12))
            logger.info(f"Image added to report: {file_path}")
        except Exception as e:
            logger.error(f"Failed to add image to report: {str(e)}")
            elements.append(Paragraph(f"Image not available: {str(e)}", styles['Normal']))

        normal_style = styles['Normal']
        elements.append(Paragraph(f"Filename: {filename}", normal_style))
        elements.append(Paragraph(f"Crop Type: {result['crop_type']}", normal_style))
        elements.append(Paragraph(f"Fertilization Required: {result['fertilization_required']}", normal_style))
        elements.append(Paragraph(f"Destruction Percentage: {result['destruction_percentage']}", normal_style))
        elements.append(Paragraph(f"Yield Prediction: {result['yield_prediction']} tons/ha", normal_style))
        elements.append(Spacer(1, 12))

        footer_style = ParagraphStyle(
            name='FooterStyle',
            parent=styles['Normal'],
            textColor=colors.gray
        )
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:M:S')}", footer_style))

        doc.build(elements)
        pdf_buffer.seek(0)

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"{filename}_report.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        abort(500, description=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)