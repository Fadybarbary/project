






from flask import Flask, jsonify, request
import pickle
import numpy as np

# Load the trained model and TF-IDF vectorizer from disk
model = pickle.load(open('t.pkl', 'rb'))
vectorizer = pickle.load(open('tf.pkl', 'rb'))

# Initialize a Flask app
app = Flask(__name__)

# Define a route for the endpoint that takes in a POST request with a JSON payload containing the tweet text
@app.route('/predict', methods=['POST'])
def predict():
    # Get the tweet text from the JSON payload
    tweet_text = request.json['tweet_text']

    # Transform the tweet text into a TF-IDF feature vector
    tweet_tfidf = vectorizer.transform([tweet_text])

    # Predict the class label of the tweet using the trained LinearSVC model
    class_label = model.predict(tweet_tfidf)[0]

    # Get the distance of the tweet feature vector from the decision boundary of each class
    distances = model.decision_function(tweet_tfidf)[0]

    # Convert the distances into probability estimates using the softmax function
    class_probs = np.exp(distances) / np.sum(np.exp(distances))

    # Sort the class probabilities from high to low
    sorted_probs = sorted(zip(class_probs, ['Hate', 'Offensive', 'Neither']), reverse=True)

    # Return the predicted class label and its corresponding percentage probabilities as a JSON response
    return jsonify({
        'class_label': int(class_label),
        'class_probs': {
            sorted_probs[0][1]: '{:.2f}%'.format(sorted_probs[0][0] * 100),
            sorted_probs[1][1]: '{:.2f}%'.format(sorted_probs[1][0] * 100),
            sorted_probs[2][1]: '{:.2f}%'.format(sorted_probs[2][0] * 100)
        }
    })
    
@app.errorhandler(404)  
def invalid_route(e):
    return "invalid route"   

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)































































































# from flask import Flask, request, jsonify
# import json
# # from tensorflow.python.keras.models import load_model
# import numpy as np
# import tensorflow as tf 
# from keras.models import load_model
# import pickle
# model = pickle.load(open('my_model.h5','rb'))
# # load sentiment model
# #model = load_model("my_model.h5")

# #load json file

# with open("my_model.h5") as json_file:
# 	json_tokenizer = json.load(json_file)

# #

# def tokenize(comment):
#     new_commentlists = []
#     new_commentlist = []
#     for word in comment.split():
#         word = word.lower()
#         if(len(new_commentlist) < 100 and word in json_tokenizer):
# 	        new_commentlist.append(json_tokenizer[word])
#     if(len(new_commentlist) < 100):
#      zeros = list(np.zeros(100 - len(new_commentlist),dtype=int))
#      new_commentlist = zeros + new_commentlist
#     new_commentlists.append(new_commentlist)
#     return np.array(new_commentlist , dtype=np.dtype(np.int32))

# # create an instance of this class

# app = Flask(__name__)

# array = {'output':""}

# # Use the route() decorator to tell Flask what URL should trigger our 
# @app.route('/', methods=['POST', 'GET'])
# def index():
#     finalResult = ""
#     request_data = json. loads (request.data.decode('utf-8'))
#     text_from_app = request_data['text']
#     print(text_from_app)
#     token_text = tokenize(text_from_app)
#     sentiment_score = model.predict(token_text) # to 1 
#     print(sentiment_score[0][0])

#     array['score'] = str(sentiment_score[0][0])
#     return jsonify(array)
# if __name__ == '_main_':
#  app.run(host='0.0.0.0',debug = True)




# # # from flask_cors import CORS
# # # import pandas as pd
# # # import numpy as np
# # # import pickle
# # # from tensorflow.python.keras.models import  keras 
# # from tensorflow import keras 
# # import pickle
# # # import os
# # # import joblib
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.naive_bayes import GaussianNB
# # from sklearn.tree import DecisionTreeClassifier
# # # app = Flask(__name__)
# # # CORS(app)

# # # # Load the GrU  model
# # # # Load the model and tokenizer
# # # model = load_model('Next_word_predictionModel.h5')


# # # # Define the API endpoint
# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     text=request.form["text"]
# # #     print(text)
    
# # #     # Get the uploaded file from the request
# # #     # text=request.form["text"]
# # #     # print(text)
# # #     # Save the uploaded file to disk
# # #     #file_path = os.path.join(
# # #      #   app.config['UPLOAD_FOLDER'], uploaded_file.filename)
# # #     #uploaded_file.save(file_path)

# # #     predicted_text = Predict_Next_Words(
# # #         model=model,tokenizer=tokenizer,text=text )

# # #     response = {'predicted_text': predicted_text}
# # #     print(response)
    
# # #     return jsonify(response)
  


# # # # Define the main function
# # # if __name__ == '__main__':
# # #     app.config['UPLOAD_FOLDER'] = './uploads'
# # #     app.run(debug=True)



# # app = Flask(__name__)
# # # Load the pre-trained model
# # model = pickle.load(open('my_model.h5','rb'))

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     # Get the input data from the request
# #     data = request.get_json['text']

# #     # Preprocess the text data (e.g., tokenize, vectorize)
# #     # preprocessed_text = preprocess(data)

# #     # Make predictions using the loaded model
# #     predictions = model.predict(data['input_data'])

# #     # Convert predictions to numerical values (assuming regression task)
# #     # output_number = predictions[0][0]

# #     # Return the predicted number as a response
# #     return jsonify(predictions.tolist())   

# # # def preprocess(text):
# # #     # Tokenize and vectorize the input text (example only)
# # #     tokenizer = tf.keras.preprocessing.text.Tokenizer()
# # #     tokenizer.fit_on_texts([text])
# # #     sequences = tokenizer.texts_to_sequences([text])
# # #     preprocessed_text = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
# # #     return preprocessed_text

# # if __name__ == '_main_':
# #     app.run( debug=True)

# from flask import Flask,request,jsonify
# app= Flask(__name__)
# @app.route('/')
# def home ():
#     return "hello world"
# # @app.route('/predict',methods=['POST'])
# # def predict():
# #     processed_tweets = request.form.get('processed_tweets')
# #     result = {'processed_tweets':processed_tweets}
# #     return jsonify(result)
# if __name__ == 'main':
#     app.run(debug=True)

# from flask import Flask,request,jsonify
# app= Flask(__name__)
# @app.route('/')
# def home ():
#     return "hello world"
# @app.route('/predict',methods=['POST'])
# def predict():
#     processed_tweets = request.form.get('processed_tweets')
#     result = {'processed_tweets':processed_tweets}
#     return jsonify(result)
# if __name__ == 'main':
#     app.run(debug=True)