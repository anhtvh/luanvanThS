Install requirement env:
	pip install -r re.txt


Train:
	python train_{NN} -LM {LM} - batch {batch_size} -data {train_data_filename}


Example:
	with CNN and word2vec:
		python train_CNN.py -LM w2v -batch 2000 -data data_vphc_train.csv

	with CNN and fasttext :
		python train_CNN.py -LM fasttext -batch 2000 -data data_vphc_train.csv

	with LSTM and fasttext :
		python train_LSTM.py -LM glove -batch 300 -data data_vphc_train.csv
		