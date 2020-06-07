# How to start
Project use `python3.7` and `keras`.

## Requirements
Required LeagueofLegends.csv file in folder:

`src/lol/dataset/`

You can download one from: [here](https://www.kaggle.com/chuckephron/leagueoflegends)

## Create virtual environment
Linux: `python3.7 -m venv venv`

Windows: `py -m venv venv`
## Activate virtual environment
Linux: `source ./venv/bin/activate`

Windows: `.\venv\Scripts\activate`

## Install packages
`pip install --upgrade pip` to get latest version of pip

`pip install -r req.txt`

## Save packages
`pip freeze > req.txt`

## Run Jupyter 
`jupyter lab` or `jupyter notebook`

# Structure
`docs` - jupyter notebooks, documentations

`src` - source files

`req.txt` - requirements file.

`app.py` - main file - run this to start ap