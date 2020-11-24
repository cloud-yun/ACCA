from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS, cross_origin
from flask_apidoc import ApiDoc

import datetime
import time
import hashlib
import numpy as np
import pandas as pd

import inference as main

app = Flask(__name__)
cors = CORS(app)
doc = ApiDoc(app=app)

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
    """ 
    @api {post} /inference  inference
    @apiVersion 1.0.0
    @apiName inference
    @apiGroup Main

    @apiDescription 呼叫機器學習進行上市公司年度EPS的推測


    @apiParam {String} YearMonth 資料年月
    @apiParam {String} CompanyCode 公司代號
    @apiParam {String} CompanyName 公司名稱
    @apiParam {String} Category 產業別
    @apiParam {Number} ThisMonthRevenue 營業收入-當月營收
    @apiParam {Number} PreviousMonthRevenue 營業收入-上月營收
    @apiParam {Number} ThisMonthRevenueOfLastYear 營業收入-去年當月營收
    @apiParam {Number} RevenueGrowthRateFromLastMonth 營業收入-上月比較增減(%)
    @apiParam {Number} RevenueGrowthRateInTheSameMonthLastYear 營業收入-去年同月增減(%)
    @apiParam {Number} CumulativeRevenues 累計營業收入-當月累計營收
    @apiParam {Number} CumulativeRevenuesLastYear 累計營業收入-去年累計營收
    @apiParam {Number} GrowthRateOfCumulativeRevenues 累計營業收入-前期比較增減(%)
    @apiParam {Number} CapitalStock 實收資本額

    @apiParamExample {json} Request-Body(Example):
    [
        {
            "YearMonth": "109/10",
            "CompanyCode": "1101",
            "CompanyName": "台泥",
            "Category": "水泥工業",
            "ThisMonthRevenue": 10293900,
            "PreviousMonthRevenue": 9745147,
            "ThisMonthRevenueOfLastYear": 11211942,
            "RevenueGrowthRateFromLastMonth": 5.631038710857825,
            "RevenueGrowthRateInTheSameMonthLastYear": -8.188073038551217,
            "CumulativeRevenues": 92542392,
            "CumulativeRevenuesLastYear": 98773809,
            "GrowthRateOfCumulativeRevenues": -6.308774626682666,
            "CapitalStock": 59414007210
        },
        {
            "YearMonth": "109/10",
            "CompanyCode": "1102",
            "CompanyName": "亞泥",
            "Category": "水泥工業",
            "ThisMonthRevenue": 7433331,
            "PreviousMonthRevenue": 7256519,
            "ThisMonthRevenueOfLastYear": 7016219,
            "RevenueGrowthRateFromLastMonth": 2.436595287630336,
            "RevenueGrowthRateInTheSameMonthLastYear": 5.944968365440133,
            "CumulativeRevenues": 61997504,
            "CumulativeRevenuesLastYear": 73225486,
            "GrowthRateOfCumulativeRevenues": -15.333434591338868,
            "CapitalStock": 33614471980
        }
    ]

    @apiSuccess {String} YearMonth 資料年月
    @apiSuccess {String} CompanyCode 公司代號
    @apiSuccess {String} CompanyName 公司名稱
    @apiSuccess {String} Category 產業別
    @apiSuccess {Number} ThisMonthRevenue 營業收入-當月營收
    @apiSuccess {Number} PreviousMonthRevenue 營業收入-上月營收
    @apiSuccess {Number} ThisMonthRevenueOfLastYear 營業收入-去年當月營收
    @apiSuccess {Number} RevenueGrowthRateFromLastMonth 營業收入-上月比較增減(%)
    @apiSuccess {Number} RevenueGrowthRateInTheSameMonthLastYear 營業收入-去年同月增減(%)
    @apiSuccess {Number} CumulativeRevenues 累計營業收入-當月累計營收
    @apiSuccess {Number} CumulativeRevenuesLastYear 累計營業收入-去年累計營收
    @apiSuccess {Number} GrowthRateOfCumulativeRevenues 累計營業收入-前期比較增減(%)
    @apiSuccess {Number} CapitalStock 實收資本額
    @apiSuccess {Number} EPS 年度EPS

    @apiSuccessExample {json} Success-Response(Example):
    HTTP/1.1 200 OK
    [
        {
            "YearMonth": "109/10",
            "CompanyCode": "1101",
            "CompanyName": "台泥",
            "Category": "水泥工業",
            "ThisMonthRevenue": 10293900,
            "PreviousMonthRevenue": 9745147,
            "ThisMonthRevenueOfLastYear": 11211942,
            "RevenueGrowthRateFromLastMonth": 5.631038710857825,
            "RevenueGrowthRateInTheSameMonthLastYear": -8.188073038551217,
            "CumulativeRevenues": 92542392,
            "CumulativeRevenuesLastYear": 98773809,
            "GrowthRateOfCumulativeRevenues": -6.308774626682666,
            "CapitalStock": 59414007210,
            "EPS": 3.5303080081939697
        },
        {
            "YearMonth": "109/10",
            "CompanyCode": "1102",
            "CompanyName": "亞泥",
            "Category": "水泥工業",
            "ThisMonthRevenue": 7433331,
            "PreviousMonthRevenue": 7256519,
            "ThisMonthRevenueOfLastYear": 7016219,
            "RevenueGrowthRateFromLastMonth": 2.436595287630336,
            "RevenueGrowthRateInTheSameMonthLastYear": 5.944968365440133,
            "CumulativeRevenues": 61997504,
            "CumulativeRevenuesLastYear": 73225486,
            "GrowthRateOfCumulativeRevenues": -15.333434591338868,
            "CapitalStock": 33614471980,
            "EPS": 3.087939500808716
        }
    ]

    @apiErrorExample {json} Error-Response(Example):
    HTTP/1.1 400 Bad Request
    {
        "error": "error message."
    }   
    """

    
    try:
        data = request.get_json(force=True) 

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

        return jsonify( answer)

    except Exception as e:
        print('error:' , str(e))
        return jsonify({'error': e.args[0] }) , 400
        #raise ValueError('Model error.')
    #server_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    #return jsonify( answer)
    # return jsonify({'server_uuid': server_uuid, 'result': answer, 'server_timestamp': server_timestamp, 'client_uuid': data['client_uuid']})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
