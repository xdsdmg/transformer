run:
	python3 main.py

freeze:
	pip3 freeze > requirements.txt

.PHONY: run
