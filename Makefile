# Makefile will include make test make clean make build make run 

# specify desired location for adpy python binary 
VENV:= /home/$(USER)/anaconda3/envs/saed
PYTHON:= ${VENV}/bin/python

# clean automatic generated files
clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -rf *.egg-info
	rm -rf ./logs/*
#	rm -f cufile.log
#	rm train.log

cleanvenv:
	rm -rf ${VENV}

sync:
	git pull
	git pull origin main

# check if the virtual environment is created or not, creating one
activate: 
	conda activate ${VENV}/

# test: activate
# 	$(PYTHON) -m unittest discover

run: activate
	$(PYTHON) src/saed/run.py

eval: activate
	$(PYTHON)  src/saed/eval.py
