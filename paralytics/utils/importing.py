"""Utilities for importing files."""


import csv

from collections import OrderedDict


__all__ = [
    'get_csv'
]


def get_csv(in_filename, cols_to_keep=None, dtype=float):
    """Get csv as a list of OrderedDicts."""
    with open(in_filename, encoding='UTF-8') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        if cols_to_keep is None:
            cols_to_keep = reader.fieldnames
        data = [
            OrderedDict((key, dtype(row[key])) for key in cols_to_keep)
            for row in reader
        ]
    return data
