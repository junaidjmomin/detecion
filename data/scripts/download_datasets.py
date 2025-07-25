import os
import zipfile
import gdown

url = 'https://drive.google.com/uc?id=1cGjZySn0l6XYL9cvjXAEZ0Xx2kWy1N5F'
output = 'levir_cd.zip'

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('data/raw/')
