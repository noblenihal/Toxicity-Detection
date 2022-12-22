from flask import Flask, request
from flask_cors import CORS, cross_origin
import pandas as pd
import pickle
from ContractionPreprocessor import expand_contraction, rem_special_sym, remove_url
from ProfanityPreprocessor import PatternTokenizer
from SourceCodePreprocessor import IdentifierTokenizer
# from official.nlp import optimization
# import tensorflow as tf
# import tensorflow_hub as hub
# import tensorflow_text as text


app = Flask(__name__, template_folder='templates')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# def get_optimizer():
#     epochs = 20
#     steps_per_epoch = 19571
#     num_train_steps = steps_per_epoch * epochs
#     num_warmup_steps = int(0.1 * num_train_steps)
#     init_lr = 3e-5
#     optimizer = optimization.create_optimizer(init_lr=init_lr,
#                                                 num_train_steps=num_train_steps,
#                                                 num_warmup_steps=num_warmup_steps,
#                                                 )
#     return optimizer
# model = pickle.load(open('model-BERT-bert-profane-False-keyword-True-split-False.h5', 'rb'))
# epochs = 20
# optimizer = get_optimizer()
# options = tf.saved_model.LoadOptions(
#     allow_partial_checkpoint=False,
#     experimental_io_device="/job:localhost",
#     experimental_skip_checkpoint=False,
#     experimental_variable_policy=None
# )
# model = tf.keras.models.load_model('model-BERT-bert-profane-False-keyword-True-split-False.h5', custom_objects={'KerasLayer': hub.KerasLayer,'AdamWeightDecay': optimizer}, options=options)

model = pickle.load(open("model-RF-tfidf-profane-True-keyword-False-split-False.pickle", "rb"))


def process_text(text):

    profanity_checker=PatternTokenizer()
    source_code_checker = IdentifierTokenizer()
    
    # mandatory preprocessing
    processed_text = remove_url(text)
    processed_text = expand_contraction(processed_text)
    processed_text = profanity_checker.process_text(processed_text)
    processed_text = rem_special_sym(processed_text)
    # optional preprocessing

    processed_text = source_code_checker.remove_keywords(processed_text)
    return processed_text


def preprocess(dataframe):
    dataframe["message"] = dataframe.message.astype(str).apply(process_text)

    dataframe["profane_count"] = 0
    dataframe["anger_count"] = 0
    dataframe["emoticon_count"] = 0

    return dataframe

@cross_origin()
@app.route('/detect-toxicity', methods=['GET'])
def helloWorld():
    return "This is Toxicity Detection API. Make a POST request to this endpoint '/detect-toxicity'"


@cross_origin()
@app.route('/detect-toxicity', methods=['POST'])
def detectToxicity():
    try:
        features = []
        print(request.form)
        for val in request.form.values():
            features.append(val)

        df = pd.DataFrame(features, columns=['message'])
        df = preprocess(df)
        print(df)


        prediction = model.predict(df).tolist()

        result = {
            "result": prediction[0][0]
        }

        print(result)
        return result
    except:
        return {
            "error" :  "Something went wrong !"
        }

if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=5050)