# Installation

It is recommended to first create a virtual environment to manage dependencies (using conda, for example):

```bash
conda create -n GenomicTools python=3.13.3
conda activate GenomicTools
```

For more information on conda, visit [Conda's official website](https://conda.io).

- **Clone the repository:**

```bash
git clone https://git.yale.edu/av746/GenomicTools.git
cd GenomicTools
```

- **Run the installation script:**

```bash
make install
```

- **Reload your shell configuration:**

```bash
if [ -n "$ZSH_VERSION" ]; then
    source ~/.zshrc
else
    source ~/.bashrc
fi
```

The required Python packages are listed in the `requirements.txt` file. It is advised to install these packages manually for a smoother experience. Alternatively, you can install all required libraries using:

```bash
make install_libraries
```
