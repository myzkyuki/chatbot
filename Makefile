install: 
	pip install -Ur requirements.txt
update-requirements:
	pip freeze > requirements.txt
uninstall:
	pip uninstall -r requirements.txt
