function pyenv_install {
    sudo apt install libsqlite3-dev
    curl https://pyenv.run | bash
}

function pyenv_start {
    PYENV_ROOT="$HOME/.pyenv"
    PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
}

function setup_python {
    pyenv_install
    pyenv_start
    echo "N" | pyenv install 3.12
    conda deactivate
    pyenv local 3.12
}

function install_pipx {
    python -m pip install --user pipx
    USERPATH="$HOME/.local/bin"
    export PATH="$USERPATH:$PATH"
}

function setup_poetry {
    install_pipx
    python -m pipx install poetry==1.7.1
}


setup_python
setup_poetry
poetry install --no-root
poetry run python -m ipykernel install --user --name=salesreinforcervenv