# NLU2021 - Project
This is the repository for the Natural Language Understanding project for year 2021. The main objective is to carry on an analysis of recurrent models applied to **concept tagging** (aka slot filling). 
It consists in reproducing recent results on neural concept tagging (e.g. seq2seq models, different attention mechanisms); furthermore, it contributes with an analysis of 
how 5 different input modalities affect performance. More information can be found in the [report](https://github.com/adefgreen98/NLU2021-Project/blob/main/report.pdf).

## Usage
Install the requirements via `pip install -r requirements.txt`.

To start a training session, use `python code/main.py`; please use the argument `--exp` to determine which experiment to run (between `Models_Bidirectionality`, `LSTM_architecture`, `Decoder_input`, `Attention`, `Beam_Search`). Other variables that can be fixed are:
* `--batch_size` (defaults to 64).
* `--iterations`: number of repetitions of each configuration for each experiment, for statistical measurement (default to 10).
* `--save_models`: save models in the 'models' folder (defaults to `False`); note that each new model will ovveride one of the same configuration, if present. 
* `--no_save_stats`: do not save stats for current exexcution

The `statistics.py` provides utilities to visualize graphics from each experiment and if executed from command line with already full experiment directories it produces the graphs present in the report.

