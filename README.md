# NLU2021 - Project
This is the repository for the Natural Language Understanding project for year 2021. The main objective is to carry on an analysis of recurrent models applied to **concept tagging** (aka slot filling). 

## Usage
Install the requirements via `pip install requirements.txt`

To start a training session, use `python code/main.py`; please use the argument `--exp` to determine which experiment to run (between `Models_Bidirectionality`, `LSTM_architecture`, `Decoder_input`, `Attention`, `Beam_Search`). Other variables that can be fixed are:
* `--batch_size` (defaults to 64)
* `--iterations`: number of repetitions of each configuration for each experiment, for statistical measurement (default to 10)
