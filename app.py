# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Model import CreditModel, SK_ID
import pandas as pd


# 2. Create app and model objects
app = FastAPI()
model = CreditModel()


@app.get('/')
def test():
    return {'message': 'Hello, stranger'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict_score')
def calc_score(id: SK_ID):
    data = id.dict()
    score = model.predict_score(data['id_number']
    )
    return {
        'score': score
    }

@app.post('/explain_score')
def calc_score(id: SK_ID):
    data = id.dict()
    sp_value, sp_base_value, sp_data, sp_feat_names = model.explanation(data['id_number'])
    return {
        'value': sp_value.tolist(),
        'base_value': sp_base_value.tolist(),
        'data': pd.Series(sp_data.reshape(-1)).fillna('missing_value').tolist(),
        'feat_names': sp_feat_names
    }


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
#if __name__ == '__main__':
 #   uvicorn.run(app, host='https://credit-score-oc.herokuapp.com/', port=8000)
    #uvicorn.run(app, host='127.0.0.1', port=8000)