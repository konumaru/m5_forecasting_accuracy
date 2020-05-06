sudo apt update

sudo apt install -y build-essential # c++コンパイラ
sudo apt install -y libffi-dev
sudo apt install -y zlib1g-dev
sudo apt install -y liblzma-dev
sudo apt install -y libbz2-dev libreadline-dev libsqlite3-dev # bz2, readline, sqlite3
sudo apt install -y python-pip
sudo apt install -y make
sudo apt install -y wget
sudo apt install -y curl
sudo apt install -y llvm
sudo apt install -y libncurses5-dev
sudo apt install -y libncursesw5-dev
sudo apt install -y xz-utils
sudo apt install -y tk-dev
sudo apt install -y libffi-dev
sudo apt install -y libssl-dev # openssl

# Install pyenv
git clone https://github.com/pyenv/pyenv.git ~/.pyenv

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
pyenv install 3.8.2
pyenv global 3.8.2

pip install -U pip
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
poetry completions bash > /etc/bash_completion.d/poetry.bash-completion
source $HOME/.poetry/env
echo 'source $HOME/.poetry/env' >> ~/.bashrc
source ~/.bashrc
poetry config virtualenvs.in-project true
poetry install

