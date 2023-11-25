function setup_python {
    pyenv_install
    pyenv_start
    echo "N" | pyenv install 3.10
    conda deactivate
    pyenv local 3.10
}

function pyenv_install {
    sudo apt install libsqlite3-dev
    curl https://pyenv.run | bash
}

function pyenv_start {
    PYENV_ROOT="$HOME/.pyenv"
    PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
}

function setup_poetry {
    install_pipx
    python -m pipx install poetry==1.7.1
}

function install_pipx {
    python -m pip install --user pipx
    USERPATH="$HOME/.local/bin"
    export PATH="$USERPATH:$PATH"
}

conda_environment_exists() {
  local environment="$1"

  conda env list | grep -q "^${environment}[[:space:]]"
  return $?
}

configure_jupyter_packages() {
    conda install -c anaconda ipykernel --yes
    python -m ipykernel install --user --name=$ENV_NAME
    jupyter_frontend_ipywidgets_version=$(conda activate azureml_py38 && pip show ipywidgets | grep -oP '(?<=Version: )\d+\.\d+\.\d+')
    poetry add --group dev ipywidgets==$jupyter_frontend_ipywidgets_version
}

create_conda_environment() {
    conda init bash
    conda deactivate
    conda env remove --name $ENV_NAME
    conda create -n $ENV_NAME python=3.10 --yes
    conda activate $ENV_NAME
    configure_jupyter_packages
    poetry install --no-root
}


ENV_NAME=salesreinforcercondavenv

setup_python
setup_poetry
if conda_environment_exists $ENV_NAME; then
    conda deactivate
    conda activate "$ENV_NAME"
else
    create_conda_environment
fi