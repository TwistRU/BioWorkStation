import csv
from flask import render_template, Flask, redirect, url_for, send_file, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap5
from sqlalchemy import TIMESTAMP, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from werkzeug.utils import secure_filename
from wtforms.csrf.core import CSRFTokenField
from wtforms import BooleanField, FormField, FloatField
from spectral import *
from math import floor
from xml.dom import minidom
from zipfile import ZipFile
from matplotlib.ticker import PercentFormatter
from io import BytesIO
import os
import tempfile
import json
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
    colorMap = BooleanField(label='Map')
    minVal = FloatField()
    maxVal = FloatField()

class CoordsParamForm(FlaskForm):
    sLong   = FloatField(label='Долгота')
    sLat    = FloatField(label='Широта')
    eLong   = FloatField(label='Долгота')
    eLat    = FloatField(label='Широта')

def AnalysParamForm_from_builder(bands_info={}):
    coords_info = bands_info['coords']
    class BandsForm(FlaskForm):
        pass

    for name, info in bands_info['bands'].items():
        setattr(BandsForm, name,
            FormField(
                BandParamForm,
                default={'minVal':info['min'], 'maxVal':info['max']},
                label=name)
                )
        
    class AnalysParamForm(FlaskForm):
        file = FileField()
        bands = BandsForm()
        coords = FormField(CoordsParamForm, default={
            'sLong': coords_info['sLong'],
            'sLat':coords_info['sLat'],
            'eLong':coords_info['eLong'],
            'eLat':coords_info['eLat']
            })
        
    return AnalysParamForm()

# Models
class RawData(db.Model):
    id: Mapped[int] = mapped_column(db.Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(db.String, unique=True, nullable=False)
    status: Mapped[str] = mapped_column(db.String, default="Not processed")
    created_at: Mapped[TIMESTAMP] = mapped_column(db.DateTime, default=func.now())
    dim_path: Mapped[str] = mapped_column(db.String, default="", nullable=True)
    info:Mapped[str] = mapped_column(db.Text, default="{}", nullable=True)

    def __repr__(self):
        return f"id:{self.id} filename:{self.filename}"
    
#Compute classes
class Band:
    def __init__(self, path, dim):
        self.dim = dim
        self.lib = envi.open(path)
        metadata = self.lib.metadata
        info = metadata['map info']
        self.shape = (int(metadata['lines']), int(metadata['samples']))
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

    def get_coords(self):
        ans = dict()
        ans['sLong'] = self.startLong
        ans['sLat'] = self.startLat
        ans['eLong'] = self.centerLong + self.centerX * self.eps
        ans['eLat'] = self.centerLati - self.centerY * self.eps
        return ans

    def get_info(self):
        band = self.read_band_data()
        des = pd.DataFrame(band.reshape(-1), dtype=np.float64).describe()
        ans = des[0].to_dict()
        return ans

    def get_pixel(self, coord):
        return (
                min(self.shape[0], max(0, round((self.startLat - coord[0]) / self.eps))),
                min(self.shape[1], max(0, round((coord[1] - self.startLong) / self.eps)))
                )
    
    def get_coord(self, pixel):
        coords = self.get_coords()
        return (
                min(self.startLat, max(coords['eLat'], coords['sLat'] - pixel[0] * self.eps)),
                max(self.startLong, min(coords['eLong'], coords['sLong'] + pixel[1] * self.eps))
                )

    def _profile_plot_pdf(self, scatter_points, profile, dir):
        fig, ax = plt.subplots()
        ax.plot(range(len(profile)), profile)
        ax.scatter(sorted(list(scatter_points.keys())), [scatter_points[i][1] for i in sorted(list(scatter_points.keys()))], color="red")
        for i, v in enumerate(scatter_points.items()):
            ax.annotate(f"{i+1}", 
                        xy=(v[0],
                            v[1][1]),
                        xycoords='data',
                        xytext=(0, 10),
                        textcoords="offset points")
        ax.set_title(self.name+"\n"+self.dim.time)
        file_format = 'pdf'
        path = os.path.join(dir, self.name)
        f = f"{path}_PP.{file_format}"
        fig.savefig(f, format=file_format)
        plt.close()
        return f

    def _profile_plot_txt(self, scatter_points, coords_points, profile, dir):
        f = os.path.join(dir, f'{self.name}.csv')
        with open(f, 'w') as file:
            writer = csv.writer(file)
            j = 1
            for i in range(len(profile)):
                if scatter_points.get(i, None) is None:
                    row = [i, *coords_points[i], profile[i]]
                else:
                    row = [i, *scatter_points[i][0], scatter_points[i][1], j]
                    j += 1
                writer.writerow(row)
        return f
    
    def profile_plot(self, dir, exp_data=[]):
        band = self.read_band_data()
        pix_exp_data = list(map(self.get_pixel, exp_data))
        start = pix_exp_data[0]
        end = pix_exp_data[1]
        points = get_path(start, end)
        scatter_points = dict()
        scatter_points[0] = [exp_data[0], band[pix_exp_data[0][0], pix_exp_data[0][1]]]
        scatter_points[len(points) - 1] = [exp_data[1], band[pix_exp_data[1][0], pix_exp_data[1][1]]]
        
        start = pix_exp_data[1]
        for i in range(2, len(pix_exp_data)):
            end = pix_exp_data[i]
            points = points[:-1]
            points.extend(get_path(start, end))
            scatter_points[len(points) - 1] = [exp_data[i], band[pix_exp_data[i][0], pix_exp_data[i][1]]]
            start = pix_exp_data[i]
        coords_points = list(map(self.get_coord, points))
        
        profile = list(map(lambda p: band[p[0],p[1]], points))

        f = [self._profile_plot_txt(scatter_points, coords_points, profile, dir),
             self._profile_plot_pdf(scatter_points, profile, dir)]
        return f

    def color_map(self, bounds, coords, exp_data):
        band = self.read_band_data()
        fig, ax = plt.subplots()
        down, left = self.get_pixel((coords['eLat'], coords['sLong']))
        up, right = self.get_pixel((coords['sLat'], coords['eLong']))
        step_ver = (down - up) // 10
        step_hor = (right - left) // 10
        des = pd.DataFrame(band[up:down, left:right].reshape(-1), dtype=np.float64).describe()
        ax.grid(True)
        # Y
        yticks = list(range(0, down - up + 1, step_ver))
        ytickLabels = list(map(lambda x: self.startLat - (up + x) * self.eps, yticks))
        ax.set_yticks(yticks)
        ax.set_yticklabels(list(map(lambda x: f"{round(x, 2)}", ytickLabels)))
        # X
        xticks = list(range(0, right - left + 1, step_hor))
        xtickLabels = list(map(lambda x: self.startLong + (left + x) * self.eps, xticks))
        ax.set_xticks(xticks)
        ax.set_xticklabels(list(map(lambda x: f"{round(x, 2)}", xtickLabels)), rotation=45, ha='right')
        # SHOW
        color_map = mpl.colormaps['jet']
        if bounds is None:
            vmin = des[0]['mean'] - 3 * des[0]['std']
            vmax = des[0]['mean'] + 3 * des[0]['std']
        else:
            vmin = bounds[0]
            vmax = bounds[1]
        band_show = ax.imshow(band[up:down, left:right], cmap=color_map, vmin=vmin, vmax=vmax)
        text_size = 10
        if exp_data is not None:
            pix_exp_data = list(map(self.get_pixel, exp_data))
            for i, point in enumerate(pix_exp_data):
                ax.annotate(
                    f'{i + 1}',
                    xy=(point[1] - left, point[0] - up),
                    ha='center',
                    va='bottom',
                    size=text_size
                )
                ax.plot(point[1] - left, point[0] - up, 'o', markersize=3, color='black')
        fig.colorbar(band_show)
        ax.set_title(self.name+"\n"+self.dim.time)
        return fig
    
    def get_color_map(self, dir, bounds, coords, exp_data):
        fig = self.color_map(bounds, coords, exp_data)
        file_format = 'pdf'
        path = os.path.join(dir, self.name)
        f = f"{path}_CM.{file_format}"
        fig.savefig(f, format=file_format)
        plt.close(fig)
        return f
    
    def get_preview_color_map(self):
        bounds = [float(request.args.get('minVal')), float(request.args.get('maxVal'))]
        coords = {}
        for i in ['sLat', 'sLong', 'eLat', 'eLong']:
            coords[i] = float(request.args.get(i))
        fig = self.color_map(bounds, coords, None)
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf
    
    def get_preview_hist(self):
        ban = self.read_band_data().reshape(-1)
        h_max = float(request.args.get('maxVal'))
        h_min = float(request.args.get('minVal'))
        h_ban = ban[(~np.isnan(ban)) & (ban <= h_max) & (ban >= h_min)]
        fig, ax = plt.subplots()
        ax.hist(h_ban, bins=100)
        # plt.yscale('log')
        fig.gca().yaxis.set_major_formatter(PercentFormatter(h_ban.shape[0]))
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf
        
class Dim:
    def __init__(self, path):
        with open(path) as f:
            data = minidom.parse(f)
        bands_path_elems = data.getElementsByTagName('DATA_FILE_PATH')
        self.time = data.getElementsByTagName('PRODUCT_SCENE_RASTER_START_TIME')[0].firstChild.nodeValue
        bands_paths = list(map(lambda x: x.attributes['href'].value, bands_path_elems))
        bands = []
        dirname = os.path.dirname(path)
        for bands_path in bands_paths:
            bands.append(Band(os.path.join(dirname,bands_path), self))
        self.bands = {band.name:band for band in bands}
    
    def get_bands_info(self):
        ans = dict()
        ans['bands'] = {name:band.get_info() for name, band in self.bands.items()}
        ans['coords'] = list(self.bands.values())[0].get_coords()
        return ans

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
    return render_template('uploadData.html', form=form)

@app.route('/analytics/<int:rawdata_id>/<string:band_name>', methods=['GET'])
def get_preview_image(rawdata_id, band_name):
    rd = db.get_or_404(RawData, rawdata_id)
    dim = Dim(rd.dim_path)
    band = dim.get_band(band_name)
    if request.args.get('type') == 'hist':
        buf = band.get_preview_hist()
    elif request.args.get('type') == 'cm':
        buf = band.get_preview_color_map()
    return send_file(buf, mimetype='image/PNG')

@app.route('/analytics/<int:rawdata_id>', methods=['POST', 'GET'])
def analytics(rawdata_id=None):
    if rawdata_id is None:
        return redirect(url_for('analytics_history'))
    rd = db.get_or_404(RawData, rawdata_id)
    info = json.loads(rd.info)
    form = AnalysParamForm_from_builder(info)
    if form.is_submitted():
        return process_request(form, rd)
    return render_template('analytic.html', rd=rd, form=form, band_names=list(info.keys()))

@app.route('/history')
def analytics_history():
    rawdatas = RawData.query.all()
    return render_template('analyticsHistory.html', rawdatas=rawdatas)

@app.route('/instruction')
def instruction():
    return render_template('instruction.html')

def process_request(form, rd: RawData):
    dim = Dim(rd.dim_path)
    with tempfile.TemporaryDirectory() as tmpdirname:
        exp_data_file = form.file.data
        to_profile = None
        if exp_data_file is not None:
            sec_filename = secure_filename(exp_data_file.filename)
            exp_data_file.save(os.path.join(tmpdirname, sec_filename))
            exp_data = exp_data_file.getvalue().decode()
            exp = list(map(lambda x: x.split(), exp_data.split('\n')))
            exp = list(map(lambda x: list(map(float, x)), list(sorted(exp[:-1], key=lambda x: int(x[2])))))
            to_profile = list(map(lambda x: x[:2], exp))
        files = []
        for band_form in form.bands:
            if isinstance(band_form, CSRFTokenField) or isinstance(band_form, FileField):
                continue
            name = band_form.name
            data = band_form.data
            if data['colorMap'] or data['profilePlot']:
                band = dim.get_band(name)
            if data['colorMap']:
                files.append(band.get_color_map(tmpdirname, (data['minVal'], data['maxVal']), form.coords.data, to_profile))
            if data['profilePlot'] and to_profile is not None:
                files.extend(band.profile_plot(tmpdirname, to_profile))
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
    print('Start subprocess')
    process = subprocess.run(
        [GPT_PATH,
          f"{graph_path}",
            f"-Pifile={ifile_path}",
              f"-Pofile={ofile_path}",],
                capture_output=True)
    # print(process, process.stderr, process.stdout)
    if "Error" in str(process.stderr):
        rd.status = "Error GPT"
        return
    if "done" in str(process.stdout):
        rd.info = json.dumps(Dim(ofile_path).get_bands_info())
        rd.status = "Ready"
        print("Subprocess done successfully")
    else:
        print(process.stdout, process.stderr, sep='\n')
        rd.status = "Error while processing"
    rd.dim_path = ofile_path
    db.session.commit()
    print("End processing")

with app.app_context():
    db.create_all()
