import os
import uuid
from flask import Flask, render_template, request, send_file,session
import urllib
from PIL import Image
from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.metrics import mean_absolute_error
from werkzeug.utils import secure_filename
import secrets
# from werkzeug.datastructures import  FileStorage

app = Flask(__name__)


base_dir = os.path.dirname(os.path.abspath(__file__))

def mae_in_months(x, y):
    return mean_absolute_error((41.18*x + 127.320), 
                               (41.18*y +127.320))

model = load_model(os.path.join(base_dir,'models/model.hdf5'))
# model = load_model(os.path.join(base_dir,'models/xception_weak_model.hdf5'),compile=False,custom_objects={"mae_in_months": mae_in_months}); 

UPLOAD_FOLDER = os.path.join('static', 'uploads')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


app.secret_key = secrets.token_hex(16)


def deleteFiles():
    app.logger.info('Starting')
    for file_name in os.listdir(UPLOAD_FOLDER):
        file = UPLOAD_FOLDER +'/' + file_name
        if os.path.isfile(file):
            app.logger.info('Deleting file:', file)
            os.remove(file)


# Adding allowed extensions of images only

allowed_extensions = ['png']

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1] in allowed_extensions


classes = ['male' ,'female']


def runModel(filename , model):
    img = load_img(filename , target_size = (256 , 256))
    img = img_to_array(img)
    img = img.reshape((256,256,3))

    img = img.astype('float32')
    img = img/255.0
    result = model.predict(img)

    dict_result = {}
    for i in range(10):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]
    
    prob_result = []
    class_result = []
    for i in range(3):
        prob_result.append((prob[i]*100).round(2))
        class_result.append(dict_result[prob[i]])

    return class_result , prob_result



@app.route("/",methods=['GET'])
@app.route("/index",methods=['GET'])
def hello_world():
    deleteFiles()
    return render_template('index.html')


@app.route("/predict",methods=['GET','POST'])

def predict():
    error = ''
    
    if request.method == 'POST':
        if (request.files):
            file = request.files['file']
            # file_name = secure_filename(file.filename)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
            # session['upload_image'] = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            # return render_template('image.html',image = os.path.join(app.config['UPLOAD_FOLDER'],file_name))
            
            if file and allowed_file(file.filename):
                file_name = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
                img = file.filename
                class_result , prob_result = runModel((os.path.join(app.config['UPLOAD_FOLDER'], img)) , model)
                results = {
                        "class1":class_result[0],
                            "class2":class_result[1],
                            "class3":class_result[2],
                            "prob1": prob_result[0],
                            "prob2": prob_result[1],
                            "prob3": prob_result[2],
                    }
                # return predictions
                return render_template('success.html',predictions=results,image= os.path.join(app.config['UPLOAD_FOLDER'],file_name))
            else:
                error = "Please upload images of jpg , jpeg and png extension only"
            
            
        else:
            return  {'error': 'Please add Files'}

    else:
        return {'error':'Requires post method'}
            



if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host="0.0.0.0", port=8080)

