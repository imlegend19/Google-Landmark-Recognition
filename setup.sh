KEY=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 32)
echo "SECRET_KEY = \"$KEY\"" > local_settings.py
