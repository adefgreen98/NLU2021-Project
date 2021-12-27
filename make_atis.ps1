mkdir ATIS
cd ATIS
Invoke-WebRequest https://raw.githubusercontent.com/howl-anderson/ATIS_dataset/master/data/standard_format/rasa/train.json -OutFile train.json
Invoke-WebRequest https://raw.githubusercontent.com/howl-anderson/ATIS_dataset/master/data/standard_format/rasa/test.json -OutFile test.json 
cd ..