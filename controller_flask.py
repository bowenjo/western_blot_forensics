import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask_app.model_flask import Figure, Match
from flask_app.compute_flask import DisplayFigure, AffineVisualizer 
import os
import params.config as config

app = Flask(__name__)

UPLOAD_DIR = 'uploads/'

app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.secret_key = 'MySecretKey'

if not os.path.isdir(UPLOAD_DIR):
    os.mkdir(UPLOAD_DIR)

# Allowed file types for file upload
ALLOWED_EXTENSIONS = set(['jpg','png', 'npy'])

def allowed_file(filename):
    """Does filename have the right extension?"""
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def edit_match_choices(form, choices):
    form["match"].group_id.choices = [(choice, choice) for i, choice in enumerate(choices)]

def save_file_to_server(form, key):
    """Save uploaded file on server if it exists and is valid"""
    if request.files:
        file = request.files[form[key].filename.name]
        if file and allowed_file(file.filename):
            # Make a valid version of filename for any file system
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return filename



@app.route('/wb_forensics_demo', methods = ["GET", "POST"])
def index():
    # intitialize the form objects
    form = {
            "figure": Figure(request.form),
            "match": Match(request.form)
            }

    if os.path.isfile("static/Figure.png"):
        result = {
              "figure": "static/Figure.png",
              "image_match": None,
              "feature_match": None
              }
        edit_match_choices(form, choices = np.load("static/Matches.npy").item()["image_pairs"][0])
    else:
        result = {
              "figure": None,
              "image_match": None,
              "feature_match": None
              }        



    filename = None  # default

    if request.method == 'POST' and request.form['btn'] == 'Display Figure':
        filename = save_file_to_server(form, "figure")
        config.r_T = form["figure"].thresh.data
        DF = DisplayFigure(filename, show_matches=form["figure"].show_match.data)
        # update match choices for figure
        np.save('static/Database.npy', DF.local_database)
        np.save('static/Matches.npy', DF.local_matches)
        np.save('static/figure.npy', DF.figure)
        edit_match_choices(form, choices = np.load("static/Matches.npy").item()["image_pairs"][0])
        result["figure"] = DF.plotfile


    elif request.method == 'POST' and request.form['btn'] == 'Check Match':
        Database = np.load("static/Database.npy").item()
        Matches = np.load("static/Matches.npy").item()
        figure = np.load("static/figure.npy")

        AV = AffineVisualizer(Database, Matches, AFFINE=form["match"].affine_match.data, CONTRAST=form["match"].contrast.data, HEATMAP=form["match"].heatmap.data)
        AV.compute_transform(image_pair=form["match"].group_id.data, figure=figure)
        AV.construct_aligned()

        result["image_match"] = AV.image1_file, AV.image2_file
        result["feature_match"] = AV.features
        
    else:
        result = result

    return render_template("wb_forensics_app.html", form=form, result=result)




if __name__ == '__main__':
    app.run(debug=True)
