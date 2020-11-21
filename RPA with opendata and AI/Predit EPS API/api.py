from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS, cross_origin
import datetime
import time
import hashlib
import numpy as np
import pandas as pd

import inference as main

app = Flask(__name__)
cors = CORS(app)

####### PUT YOUR INFORMATION HERE #######
TEAM_NAME = 'ACCA'                  #
SALT = 'rpa_with_opendata_and_ai_salt'                  #
#########################################

def generate_server_uuid(input_string):
    """ Create your own server_uuid
    @param input_string (str): information to be encoded as server_uuid
    @returns server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string+SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid

@app.route('/healthcheck', methods=['POST'])
def healthcheck():
    """ API for health check """
    data = request.get_json(force=True)
    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(TEAM_NAME+ts)
    server_timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'client_uuid': data['client_uuid'], 'server_uuid': server_uuid, 'server_timestamp': server_timestamp})


@app.route('/inference', methods=['POST'])
@cross_origin(origin='*')
def inference():
    """ API that return image base64 string when web calls this API """
    data = request.get_json(force=True) 

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(TEAM_NAME+ts)

    
    try:
        print('start transferring...')
        start = time.time()

        # 讀取json, 並轉成pandas dataframe
        # json_data = data["data"]
        json_data = data
        test = pd.DataFrame.from_dict(json_data)

        # 呼叫主程式進行預測
        result = main.to_predict(test)

        # 將結果轉成Json格式, 以便回傳
        answer = result.to_dict(orient="records")

        print('total run time: ' , time.time() - start)

    except:
        raise ValueError('Model error.')
    server_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return jsonify( answer)
    # return jsonify({'server_uuid': server_uuid, 'result': answer, 'server_timestamp': server_timestamp, 'client_uuid': data['client_uuid']})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
