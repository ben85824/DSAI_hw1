# DSAI_hw1 111
Dataset: 2016-2018 台灣電力公司_過去電力供需資訊
## 模型結構
採用seq2seq模型 用前14天資料預測後7天資料
LSTM16-LSTM32-LSTM16_state-LSTM16-Fc1
透過前三層LSTM 提取特徵
透過最後一層 LSTM 預測未來7天資訊
## 資料特徵
除了尖峰負載之外還加入 weekday資訊
預測時也會加入 weekday資訊
## 訓練:
batchsiz = 8
epochsize = 100
訓練時間約 25分鐘
