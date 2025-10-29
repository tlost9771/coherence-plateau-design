all: run

run:
	python make_figures.py --Gamma 0.6

clean:
	rm -f figures/*.pdf
