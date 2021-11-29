from flask import Flask, request, render_template
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from os.path import join, dirname, realpath
import os
from MedNet import MedNet
 
model = torch.load('saved_model')
classliste=['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']


app = Flask(__name__) 

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route("/action_page", methods=['GET', 'POST'])
def upload():
    dirnames = join(dirname(realpath(__file__)), 'static\\upload')
    UPLOAD_FOLDER = dirnames
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':
            file1 = request.files['file1']
            path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
            file1.save(path)
            return   predict(path)

def scaleImage(y):          
    if(y.min() < y.max()):  
        y = (y - y.min())/(y.max() - y.min()) 
        z = y - y.mean()  
        return z      


def predict(path):
    img = Image.open(path)
    my_transforms = transforms.Compose([transforms.ToTensor(),transforms.Resize(64),transforms.Normalize([0.5 ],[0.5 ])])
    xtest= my_transforms(img).unsqueeze(0)
    xtest=scaleImage(xtest)
    yOut = model(xtest)
    indices = yOut.max(1)[1].tolist()[0]
    pred = classliste[indices]
    return render_template('index.html',file1=path ,prediction_text='Picture: {} '.format(pred))


if __name__ == '__main__':
    app.run(debug=True)