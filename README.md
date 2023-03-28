# Phenomena

The project involves using machine learning and natural language processing techniques to extract and mine disease-phenotype relationships from large datasets that may not be obvious or easily discoverable through traditional methods such as manual curation.

The expected timelines and deliverables are described in the Projects tab. More details of the project inside the [docs](docs) folder.

## Set up

Prerequirements:
1. Download Poetry following the instructions here: https://python-poetry.org/docs/#installation
2. Install the Python version of the project (3.10.8). To do so, you can use Pyenv as a Python version manager: `curl https://pyenv.run | bash`

Install project dependencies:
```bash
poetry env use 3.10.8
poetry install --sync
```

**Ready!** You can now send commands from the project env like `poetry run python src/main.py` or `poetry run pytest`. You can also use the `poetry shell` command to open a shell in the project env.
