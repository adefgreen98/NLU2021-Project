mkdir data
cd data
Invoke-WebRequest https://github.com/howl-anderson/ATIS_dataset/blob/master/data/raw_data/ms-cntk-atis/atis.train.pkl?raw=true
Invoke-WebRequest https://github.com/howl-anderson/ATIS_dataset/blob/master/data/raw_data/ms-cntk-atis/atis.test.pkl?raw=true
cd ..