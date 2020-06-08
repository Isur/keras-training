# How to start
Project use `python3.7.
`
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
