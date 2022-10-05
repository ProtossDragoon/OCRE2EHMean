test :
	@python3 -m unittest discover -s . -p "test_*.py" -v

install :
	@python3 -m pip install -r requirements.txt

uninstall :
	@python3 -m pip uninstall -r requirements.txt