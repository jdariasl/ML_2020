lab = lab1
parte = parte1
lab_notebook = $(lab)_$(parte).ipynb
data = AirQuality.data

clean:
	rm -f *.ipynb
	rm -f *.py
	rm -f *.csv
	rm -f *.data
	rm -f *.mat
	rm -f *.txt

stage:
	cp ../$(lab)$//$(lab_notebook)$  $(lab_notebook)$ 
	cp ../commons/utils/general.py general.py
	cp ../commons/utils/imports.py imports.py
	cp ../commons/utils/$(lab).py $(lab).py
	cp ../commons/utils/data/$(data) $(data)

recreate:
	$(MAKE) clean
	cp ../$(lab)$//$(lab_notebook)$  $(lab_notebook)$ 
	cp ../commons/utils/general.py general.py
	cp ../commons/utils/imports.py imports.py
	cp ../commons/utils/$(lab).py $(lab).py
	cp ../commons/utils/data/$(data) $(data)

to_prod:
	\cp general.py ../commons/utils/general.py
	\cp imports.py ../commons/utils/imports.py
	\cp $(lab_notebook)$  ../$(lab)$//$(lab_notebook)$
	\cp $(lab).py ../commons/utils/$(lab).py
	-\cp *.csv ../commons/utils/data/
	-\cp *.data ../commons/utils/data/
	-\cp *.txt ../commons/utils/data/
	-\cp *.mat ../commons/utils/data/
	$(MAKE) clean