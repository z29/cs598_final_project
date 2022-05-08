* Original Paper: https://doi.org/10.48550/arXiv.1911.12216
* Original Code: https://github.com/Accountable-Machine-Intelligence/ConCare
* Data Download: https://www.dropbox.com/s/wnnrgvda5t659nv/data.rar?dl=0
* Pretrained Model: model/concare0


* Dependencies:

## Run the code:

* Navigate to the project folder
* Download and unizp the data folder from the link provided at the top of this readme into the project folder
* Run the notebook

### Baseline methods 

Open Anaconda navigator and create a new environment called Logistic
   
Open terminal in that environment and run below

	conda install tensorflow=2.1.0
	conda install keras=2.3.1
	conda install pandas=0.23.4
	conda install -c conda-forge tqdm
	conda install -c free scikit-learn=0.18.2

	python -um mimic3models.in_hospital_mortality.logistic.main --l2 --C 0.001 --output_dir mimic3models/in_hospital_mortality/logistic

Open Anaconda navigator and create a new environment called OtherBaseLine Models

To run lstm, Tlstm, grualpha and retain , please prepare anaconda navigator with below 

	conda install tensorflow=2.1.0
	conda install keras=2.3.1
	conda install pandas=0.23.4
	conda install -c conda-forge tqdm
	conda install -c scikit-learn

	python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/in_hospital_mortality

	python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/tlstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/in_hospital_mortality >> tlstm_output.txt

	python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/grualpha.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/in_hospital_mortality >>gruaplha_output.txt

	python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/retain.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/in_hospital_mortality >> retain_output.txt