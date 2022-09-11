import os
import uuid
from flask import Flask, render_template, request, send_file
import urllib
from PIL import Image
from keras.models import load_model
from keras.utils import load_img, img_to_array


app = Flask(__name__)


base_dir = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(base_dir,'models/model.hdf5'))


# Adding allowed extensions of images only

allowed_extensions = ['png','jpg','jpeg']

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1] in allowed_extensions


classes = ['airplane' ,'automobile', 'bird' , 'cat' , 'deer' ,'dog' ,'frog', 'horse' ,'ship' ,'truck']


def runModel(filename , model):
    img = load_img(filename , target_size = (32 , 32))
    img = img_to_array(img)
    img = img.reshape(1 , 32 ,32 ,3)

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
    return render_template('index.html')

@app.route("/predict/",methods=['GET','POST'])

def predict():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if (request.files):
            if file and allowed_file(file.filename):
                file = request.files['file']
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename
                class_result , prob_result = predict(img_path , model)
                predictions = {
                        "class1":class_result[0],
                            "class2":class_result[1],
                            "class3":class_result[2],
                            "prob1": prob_result[0],
                            "prob2": prob_result[1],
                            "prob3": prob_result[2],
                    }
            else:
                error = "Please upload images of jpg , jpeg and png extension only"
            
            if(len(error) == 0):
                return  render_template('predict.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)


            



if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host="0.0.0.0", port=8080)

