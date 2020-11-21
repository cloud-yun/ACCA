import numpy as np
import pandas as pd
import tensorflow as tf
import time

# batch size
DNN_MODEL = "./models/dnn_model"
DATA_SCHEMA = "./data/training data/Data_Schema.csv"

def load_schema():
  train_features = pd.read_csv(DATA_SCHEMA)
  train_features.pop('EPS')
  return train_features

def load_model():
  return tf.keras.models.load_model(DNN_MODEL) 


def reshape_columns(data):
  # 已有公司代碼, 不需再把公司名稱加入訓練, 故刪除此欄位
  data = data.drop(['CompanyName'], axis = 1) 
  # 將年月分開
  data[['DataYear','DataMonth']] = data.YearMonth.str.split("/", expand=True)

  data['DataYear'] = data['DataYear'].astype(int)
  data['DataMonth'] = data['DataMonth'].astype(str)

  # 將民國年改成西元年
  data['DataYear'] = data['DataYear'] + 1911

  # YearMonth不在訓練集中, 故刪除此欄位
  data = data.drop(['YearMonth'], axis = 1) 

  return data   

def fix_datatype(data):

  data['CompanyCode'] = data['CompanyCode'].astype(str)

  data['CapitalStock'] = pd.to_numeric(data['CapitalStock'], errors='coerce')
  data['ThisMonthRevenue'] = pd.to_numeric(data['ThisMonthRevenue'], downcast='integer', errors='coerce')
  data['PreviousMonthRevenue'] = pd.to_numeric(data['PreviousMonthRevenue'], downcast='integer', errors='coerce')
  data['ThisMonthRevenueOfLastYear'] = pd.to_numeric(data['ThisMonthRevenueOfLastYear'], downcast='integer', errors='coerce')
  data['RevenueGrowthRateFromLastMonth'] = pd.to_numeric(data['RevenueGrowthRateFromLastMonth'], errors='coerce') 
  data['RevenueGrowthRateInTheSameMonthLastYear'] = pd.to_numeric(data['RevenueGrowthRateInTheSameMonthLastYear'], errors='coerce')
  data['CumulativeRevenues'] = pd.to_numeric(data['CumulativeRevenues'], downcast='integer', errors='coerce')
  data['CumulativeRevenuesLastYear'] = pd.to_numeric(data['CumulativeRevenuesLastYear'], downcast='integer', errors='coerce')
  data['GrowthRateOfCumulativeRevenues'] = pd.to_numeric(data['GrowthRateOfCumulativeRevenues'], errors='coerce')

  return data 

def fill_nan(data):
  data.dropna(inplace=True) 

  return data

def data_clean(df):
  data = df.copy()

  data = reshape_columns(data)
  data = fix_datatype(data)
  data = fill_nan(data)
  data["CapitalStock"] = (data["CapitalStock"] / 1000).astype(int)

  return data

def get_dummies(df):
  data = df.copy()

  train = load_schema()
  test = pd.get_dummies(data, prefix='', prefix_sep='')
  
  missing_cols = set( train.columns ) - set( test.columns )
  
  # Add a missing column in test set with default value equal to 0
  for column in missing_cols:
    test[column] = 0
  # Ensure the order of column in the test set is in the same order than in train set
  test = test[train.columns]

  return test

def get_result(origin_df, clean_df, predictions):

  clean_df['EPS'] = predictions
  clean_df['DataYear'] = clean_df['DataYear'].astype(int)  - 1911
  clean_df['YearMonth'] = clean_df['DataYear'].astype(str).str.cat(clean_df['DataMonth'].astype(str),sep="/")

  predict_df = clean_df[['YearMonth','CompanyCode','EPS']].copy()

  if (origin_df.CompanyCode.dtype == np.int64):
    predict_df['CompanyCode'] = pd.to_numeric(predict_df['CompanyCode'], downcast='integer', errors='coerce')

  result = pd.merge(origin_df, predict_df, on=['YearMonth', 'CompanyCode'], how='left')

  return result


def to_predict(test):
  # 呼叫函數清理資料
  clean_data = data_clean(test)

  # clean_data.to_csv('./results/clean_data.csv', encoding='utf-8-sig', index=False)

  # 進行OneHot encode
  test_features = get_dummies(clean_data)

  # 載入訓練模型
  dnn_model = load_model()

  # 進行預測
  test_predictions = dnn_model.predict(test_features)

  # 依原始資料組合成預測結果
  predict_result = get_result(test, clean_data, test_predictions)

  return predict_result


def build_testdata():
  train = pd.read_csv("./data/training data/TrainingData.csv")

  test = train.loc[train['CompanyCode'] == 2330].drop(['EPS'], axis = 1)
  test.to_csv('../data/test data/2330.csv', encoding='utf-8-sig', index=False)

  test['DataYear'] = test['DataYear'].astype(int)  - 1911

  test['YearMonth'] = test['DataYear'].astype(str).str.cat(test['DataMonth'].astype(str),sep="/")

  test = test.drop(['DataYear', 'DataMonth'], axis = 1)

  test.to_json('../data/test data/2330.json',orient="records")


def test():
  start_time = time.time()
  print("start inference")
  test = pd.read_csv("./data/test data/2330.csv")

  # 呼叫函數進行缺失值補值
  test_features = fill_nan(test)

  # 呼叫函數修正欄位格式
  test_features = fix_column_datatype(test_features)

  # 進行OneHot encode
  test_features = get_dummies(test_features)

  # 載入訓練模型
  dnn_model = load_model()

  # 進行預測
  test_predictions = dnn_model.predict(test_features)

  test['EPS'] = test_predictions

  test.to_csv('./results/result.csv', encoding='utf-8-sig', index=False)
  

  end_time = time.time()
  print('total time: ' , end_time - start_time)

# build_testdata()