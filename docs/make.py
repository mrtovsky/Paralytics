"""Python script for building the Paralytics documentation."""


import argparse
import codecs
import json
import pathlib
import shutil


DOCS_PATH = pathlib.Path(__file__).parent
REFERENCE_PATH = DOCS_PATH.joinpath('source', 'reference')
CONFIG_FILE = DOCS_PATH.joinpath('config.json')


class DocsBuilder(object):
    """

    Parameters
    ----------
    reference_path: str
        Path to the reference directory.
    config_file: str
        File with .json extension.
    """
    def __init__(self, reference_path, config_file):
        self.reference_path = reference_path
        with codecs.open(config_file) as file:
            self.config = json.load(file)

    def make_reference(self):
        pathlib.Path(str(REFERENCE_PATH.resolve())).mkdir(
            parents=True, exist_ok=True
        )
        pass

    @staticmethod
    def clean(paths):
        """Clean files from directories.

        Parameters
        ----------
        paths: str or list
        """
        try:
            shutil.rmtree(paths, ignore_errors=True)
        except TypeError:
            for path in paths:
                shutil.rmtree(path, ignore_errors=True)


def main():
    with codecs.open(str(CONFIG_FILE.resolve()), 'r') as file:
        config = json.load(file)


if __name__ == '__main__':
    main()
