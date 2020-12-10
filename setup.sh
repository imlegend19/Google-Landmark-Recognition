KEY=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 32)
echo "SECRET_KEY = \"$KEY\"" > local_settings.py

# For flask image storage
mkdir "uploads"

# Setting up venv
python -m venv venv
source venv/bin/activate

# Installing requirements
pip install -r requirements

echo "Downloading delg saved models..."
kaggle datasets download --force -d camaskew/delg-saved-models
7z x delg-saved-models.zip -odelg-saved-models/
rm delg-saved-models.zip
mv delg-saved-models glr/

echo "Downloading Paris Dataset..."
wget https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz
wget https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz
tar -xf paris_1.tgz
tar -xf paris_2.tgz
mv paris glr/
rm paris_1.tgz
rm paris_2.tgz

echo "Downloading saved global features for Paris10k..."
gdown --id 1--FW7LRxvl7_fK4lDcpgXyV27X9DX9FK  # test global features
gdown --id 1-4bSPGTJ0iW-E2JSGJpLIJJMIQ4kTrKS  # train global features
mkdir glr/data
mv test_gf.pkl glr/data/
mv train_gf.pkl glr/data

echo "Downloading saved local features for Paris10k..."
gdown --id 19UkF92pS0kYnTnuppb7GZHf1vADTS03r  # train local features
gdown --id 1-3pIT_u4MlHmU-hOp27oW0z9osBsBgpz  # test local features
mkdir glr/train_lf && tar -xf train_lf.tar.gz -C glr/train_lf
mkdir glr/test_lf && tar -xf test_lf.tar.gz -C glr/test_lf
rm train_lf.tar.gz
rm test_lf.tar.gz

echo "Setup Complete!"
