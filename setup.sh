echo "[`date`]": "START"
echo "[`date`]": "Creating Python 3.9 Virtual Environment" 
python3.9 -m venv ./venv
echo "[`date`]": "activate venv"
source ./venv/bin/activate
echo "[`date`]": "installing the requirements" 
pip install -r requirements.txt
echo "[`date`]": "END" 