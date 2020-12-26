from commons import get_tensor
from inference import get_cat_name
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import os
import io
from PIL import Image

# save upload image to folder
UPLOAD_FOLDER = 'static/tmp'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'tiff'])


app = Flask(__name__)

# config upload image
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def index() -> 'html':
    return render_template('home.html')


@app.route('/profil')
def profil():
    return render_template('profil.html')


@app.route('/data')
def data():
    return render_template('data.html')


@app.route('/prosedur')
def prosedur():
    return render_template('prosedur.html')


@app.route('/model')
def model():
    return render_template('model.html')


@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():

    if request.method == 'GET':
        return render_template('prediksi.html')

    if request.method == 'POST':
        # print(request.files)

        if 'file' not in request.files:

            # jika upload gagal
            return render_template('prediksi.html')

            # jika upload sukses
        file = request.files['file']

        if file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                file.filename
            )
            # save image
            file.save(image_location)

        labels, max_probability, classes0, classes1, classes2, prob0, prob1, prob2, total, image_location = get_cat_name(
            image_location=image_location)

    return render_template('prediksi.html', labels=labels, probs=max_probability,
                           classes0=classes0, classes1=classes1, classes2=classes2,
                           prob0=prob0, prob1=prob1, prob2=prob2, total=total, image_location=image_location)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="5000")
    # app.run(debug=True)
