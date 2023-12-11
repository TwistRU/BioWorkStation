from flask import render_template, Flask, redirect, url_for, send_file
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap5
from sqlalchemy import TIMESTAMP, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from werkzeug.utils import secure_filename
from wtforms.csrf.core import CSRFTokenField
from wtforms import BooleanField, FormField
from spectral import *
from math import floor
from xml.dom import minidom
from zipfile import ZipFile
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import spectral.io.envi as envi
import subprocess
import datetime



from config import GPT_PATH, _basedir

app = Flask(__name__)
app.config.from_pyfile('config.py')
bootstrap = Bootstrap5(app)
mpl.use('pdf')

class Base(DeclarativeBase):
    pass


db = SQLAlchemy(app, model_class=Base)

try:
    os.makedirs(app.instance_path)
except OSError:
    pass

try:
    os.makedirs(os.path.join(app.instance_path, 'raw_data'))
except OSError:
    pass


# Forms
class RawDataFileForm(FlaskForm):
    file = FileField(validators=[FileRequired()])

class BandParamForm(FlaskForm):
    profilePlot = BooleanField()
    colorMap = BooleanField()

def AnalysParamForm_from_builder(band_names=[]):
    class AnalysParamForm(FlaskForm):
        pass
    for name in band_names:
        setattr(AnalysParamForm, name, FormField(BandParamForm, label=name))

    return AnalysParamForm()

# Models
class RawData(db.Model):
    id: Mapped[int] = mapped_column(db.Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(db.String, unique=True, nullable=False)
    status: Mapped[str] = mapped_column(db.String, default="Not processed")
    created_at: Mapped[TIMESTAMP] = mapped_column(db.DateTime, default=func.now())
    dim_path: Mapped[str] = mapped_column(db.String,default="", nullable=True)

    def __repr__(self):
        return f"id:{self.id} filename:{self.filename}"
    
#Compute classes
class Band:
    def __init__(self, path):
        self.lib = envi.open(path)
        metadata = self.lib.metadata
        info = metadata['map info']
        self.path = path
        self.name = metadata['band names'][0]
        self.description = metadata['description']
        self.eps = float(info[5])
        self.centerX, self.centerY = map(floor, map(float, info[1:3]))
        self.centerLong, self.centerLati = map(float, info[3:5])
        self.startLong = self.centerLong - (self.centerX - 1) * self.eps
        self.startLat = self.centerLati + (self.centerY - 1) * self.eps
        
    def read_band_data(self):
        try:
            self.band
        except:
            self.band = self.lib.read_band(0)
        return self.band

    def get_pixel(self, coord):
        return (round((self.startLat - coord[0])/self.eps), round((coord[1] - self.startLong)/self.eps))
        
    def profile_plot(self, dir, pix_exp_data=[]):
        band = self.read_band_data()
        exp_data = [
            [43.1043166666667, 131.833416666667],
            [43.1071666666667, 131.814583333333],
            [43.1098666666667, 131.796050000000],
            [43.1121333333333, 131.777083333333],
            [43.1149666666667, 131.758516666667],
            [43.1174500000000, 131.740433333333],
            [43.1198833333333, 131.721733333333],
            [43.1223166666667, 131.703883333333],
            [43.0927500000000, 131.714833333333],
            [43.0779166666667, 131.730583333333]
            ]
        pix_exp_data = list(map(self.get_pixel, exp_data))
        start = pix_exp_data[0]
        end = pix_exp_data[1]
        points = get_path(start, end)
        scatter_points = [0, len(points) - 1]
        start = pix_exp_data[1]
        for i in range(2, len(pix_exp_data)):
            end = pix_exp_data[i]
            points = points[:-1]
            points.extend(get_path(start, end))
            scatter_points.append(len(points) - 1)
            start = pix_exp_data[i]

        profile = list(map(lambda p: band[p[0],p[1]], points))
        scatter = list(map(lambda p: band[p[0],p[1]], pix_exp_data))

        fig, ax = plt.subplots()
        ax.plot(range(len(profile)), profile)
        ax.scatter(scatter_points, scatter, color="red")
        for i in range(len(scatter)):
            ax.annotate(f"{i+1}", 
                        xy=(scatter_points[i], scatter[i]), xycoords='data', xytext=(0, 10), textcoords="offset points")
        file_format = 'pdf'
        path = os.path.join(dir, self.name)
        f = f"{path}_PP.{file_format}"
        fig.savefig(f, format=file_format)
        plt.close()
        return f

    def color_map(self, dir):
        band = self.read_band_data()
        fig, ax = plt.subplots()
        up, down = 600, 1500
        left, right = 1800, 2500
        step = 100
        width, height = band.shape
        des = pd.DataFrame(band[up:down, left:right].reshape(-1), dtype=np.float64).describe()
        ax.grid(True)
        # Y
        yticks = list(range(0, down - up + 1, step))
        ytickLabels = list(map(lambda x: self.startLat - (up + x) * self.eps, yticks))
        ax.set_yticks(yticks)
        ax.set_yticklabels(list(map(lambda x: f"{int(x)}°{round((x - np.floor(x)) * 60, 4)}′", ytickLabels)))
        # X
        xticks = list(range(0, right - left + 1, step))
        xtickLabels = list(map(lambda x: self.startLong + (left + x) * self.eps, xticks))
        ax.set_xticks(xticks)
        ax.set_xticklabels(list(map(lambda x: f"{int(x)}°{round((x - np.floor(x)) * 60, 4)}′", xtickLabels)), rotation=45, ha='right')
        # SHOW
        color_map = mpl.colormaps['jet']
        vmin = des[0]['mean'] - 3 * des[0]['std']
        vmax = des[0]['mean'] + 3 * des[0]['std']
        band_show = ax.imshow(band[up:down, left:right], cmap=color_map, vmin=vmin, vmax=vmax)
        fig.colorbar(band_show)
        file_format = 'pdf'
        path = os.path.join(dir, self.name)
        f = f"{path}_CM.{file_format}"
        fig.savefig(f, format=file_format)
        plt.close()
        return f
        
class Dim:
    def __init__(self, path):
        with open(path) as f:
            data = minidom.parse(f)
        bands_path_elems = data.getElementsByTagName('DATA_FILE_PATH')
        bands_paths = list(map(lambda x: x.attributes['href'].value, bands_path_elems))
        bands = []
        dirname = os.path.dirname(path)
        for bands_path in bands_paths:
            bands.append(Band(os.path.join(dirname,bands_path)))
        self.bands = {band.name:band for band in bands}
    
    def get_band_names(self):
        return list(self.bands.keys())[::-1]
    
    def get_band(self, name) -> Band:
        return self.bands.get(name, None)

# Routes
@app.route('/', methods=['GET', 'POST'])
def uploadRawFile():
    form = RawDataFileForm()
    if form.validate_on_submit():
        f = form.file.data
        filename = secure_filename(f.filename)
        session = db.session
        select = db.select(RawData).filter_by(filename=filename)
        rd: RawData = session.execute(select).scalar_one_or_none()
        if rd is None:
            rd = RawData(filename=filename)
            session.add(rd)
            session.commit()
            try:
                os.makedirs(os.path.join(app.instance_path, 'raw_data', str(rd.id)))
            except OSError:
                pass
            f.save(os.path.join(
                app.instance_path, 'raw_data', str(rd.id), filename
            ))
        if rd.status != "Ready":
            compute_gpt(db, rd)
        return redirect(url_for('analytics', rawdata_id=rd.id))
    return render_template('uploadData.html.jinja', form=form)


@app.route('/analytics/<int:rawdata_id>', methods=['POST', 'GET'])
def analytics(rawdata_id=None):
    if rawdata_id is None:
        return redirect(url_for('analytics_history'))
    rd = db.get_or_404(RawData, rawdata_id)
    band_names = []
    if rd.status == "Ready":
        dim = Dim(rd.dim_path)
        band_names = dim.get_band_names()
    form = AnalysParamForm_from_builder(band_names)
    if form.is_submitted():
        return process_request(form, dim)
    return render_template('analytic.html.jinja', rd=rd, form=form, band_names=band_names)

@app.route('/history')
def analytics_history():
    rawdatas = RawData.query.all()
    return render_template('analyticsHistory.html.jinja', rawdatas=rawdatas)

def process_request(form, dim: Dim):
    with tempfile.TemporaryDirectory() as tmpdirname:
        files = []
        for band_form in form:
            if isinstance(band_form, CSRFTokenField):
                continue
            name = band_form.name
            data = band_form.data
            if data['colorMap'] or data['profilePlot']:
                band = dim.get_band(name)
            if data['colorMap']:
                files.append(band.color_map(tmpdirname))
            if data['profilePlot']:
                files.append(band.profile_plot(tmpdirname))
        zip_path = f'{os.path.join(tmpdirname, str(datetime.datetime.now()))}.zip'
        with ZipFile(zip_path, 'w') as myzip:
            for f in files:
                myzip.write(f, os.path.basename(f))
        return send_file(zip_path, as_attachment=True)

def get_path(p1, p2):
    # https://prog-cpp.ru/brezenham/
    path = [p1]
    x1,y1,x2,y2 = *p1, *p2
    A = y2 - y1
    B = x1 - x2
    sign = 1 if abs(A) > abs(B) else -1
    signa = -1 if A < 0 else 1
    signb = -1 if B < 0 else 1
    f = 0
    x,y = x1,y1
    if sign == -1:
        f += A * signa
        if f > 0:
            f -= B * signb
            y += signa
        x -= signb
        path.append((x,y))
        while x != x2 or y != y2:
            f += A * signa
            if f > 0:
                f -= B * signb
                y += signa
            x -= signb
            path.append((x,y))
    else:
        f += B * signb
        if f > 0:
            f -= A * signa
            x -= signb
        y += signa
        path.append((x,y))
        while x != x2 or y != y2:
            f += B * signb
            if f > 0:
                f -= A * signa
                x -= signb
            y += signa
            path.append((x,y))
    return path

def compute_gpt(db, rd):
    rd.status = "Proccessing"
    db.session.commit()
    print("Start processing")
    ifile_path = os.path.join(
        _basedir, "instance/raw_data", str(rd.id), rd.filename
        )
    ofile_path = os.path.join(
        _basedir, "instance/raw_data", str(rd.id), rd.filename+".dim"
        )
    graph_path = os.path.join(_basedir, "graph.xml")
    process = subprocess.run(
        [GPT_PATH,
          f"{graph_path}",
            f"-Pifile={ifile_path}",
              f"-Pofile={ofile_path}",],
                capture_output=True)
    if "Error" in str(process.stderr):
        rd.status = "Error GPT"
        return
    if "done" in str(process.stdout):
        rd.status = "Ready"
    else:
        print(process.stdout, process.stderr, sep='\n')
        rd.status = "Error while processing"
    rd.dim_path = ofile_path
    db.session.commit()
    print("End processing")

with app.app_context():
    db.create_all()
