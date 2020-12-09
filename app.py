import os

from flask_ngrok import run_with_ngrok
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
from local_settings import SECRET_KEY

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = SECRET_KEY

run_with_ngrok(app)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part!')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file!')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))

    return render_template("index.html")


def get_prediction(image_path):
    return "Landmark"


@app.route('/result/<filename>')
def uploaded_file(filename):
    landmark = get_prediction(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template("display.html", filename=filename, landmark=landmark)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/back')
def back():
    return redirect('/')


if __name__ == "__main__":
    app.run()
