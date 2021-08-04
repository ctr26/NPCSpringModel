all: build.conda.env

build.conda.env:
	conda env create --file environment.yml --force