import os
from flask import Flask, render_template, request
from keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
import secrets
from keras.metrics import mean_absolute_error

from keras_preprocessing.image import load_img,img_to_array
from keras.models import model_from_json

# from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import preprocess_input

# from werkzeug.datastructures import  FileStorage

app = Flask(__name__)


base_dir = os.path.dirname(os.path.abspath(__file__))

# Mean boneage
mean_bone_age = 127.3207517246848

# Standard deviation of boneage
std_bone_age = 41.182021399396326

# Image size
IMG_SIZE = 256


# Function to return mean absolute error in months
def mae_in_months(x, y):
    return mean_absolute_error((std_bone_age*x + mean_bone_age), 
                               (std_bone_age*y + mean_bone_age))

# load json and create model
json_file = open(os.path.join(base_dir,'models/xception_weak_model.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(os.path.join(base_dir,"models/xception_weak_model.h5"))
#print("Loaded model from disk")

#compile model
loaded_model.compile(loss ='mse', optimizer= 'adam', metrics = [mae_in_months] )


def mae_in_months(x, y):
    return mean_absolute_error((41.18*x + 127.320), 
                               (41.18*y +127.320))

# model = load_model(os.path.join(base_dir,'models/model.hdf5'))
# model = load_model(os.path.join(base_dir,'models/xception_weak_model.hdf5'),compile=False,custom_objects={"mae_in_months": mae_in_months}); 

# UPLOAD_FOLDER = os.path.join('static', 'uploads')
UPLOAD_FOLDER=os.path.join(base_dir,'static/uploads/')

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




def runModel(image):
    # Load the image
    image = load_img(image, target_size=(IMG_SIZE, IMG_SIZE))

    # Preprocess the image
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    # Make prediction using model
    prediction = loaded_model.predict(image)
   # Convert model prediction (which is normalized value) into bone age in months
    predicted_age = (prediction * std_bone_age) + mean_bone_age
    app.logger.info(predicted_age)
    return predicted_age



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
            
            if file and allowed_file(file.filename):
                file_name = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
                img = file.filename
                result = runModel((os.path.join(app.config['UPLOAD_FOLDER'], img)))
                # return {'result':float(result[0][0])}
                
                return render_template('success.html',predictions=float("{:.4f}".format(float(result[0][0]))),image= f'static/uploads/{file_name}')
            else:
                error = "Please upload images of jpg , jpeg and png extension only"
                return  {'error':error}
            
            
        else:
            return  {'error': 'Please add Files'}

    else:
        return {'error':'Requires post method'}
            

@app.route("/api",methods=['GET','POST'])

def api():
    error = ''
    
    if request.method == 'POST':
        if (request.files):
            file = request.files['file']
            
            if file and allowed_file(file.filename):
                file_name = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
                img = file.filename
                result = runModel((os.path.join(app.config['UPLOAD_FOLDER'], img)))
                return {'result':float(result[0][0])}
                # return render_template('success.html',predictions=float(result[0][0]),image= f'static/uploads/{file_name}')
            else:
                error = "Please upload images of jpg , jpeg and png extension only"
                return  {'error':error}
            
            
        else:
            return  {'error': 'Please add Files'}

    else:
        return {'error':'Requires post method'}


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(port=8080)
