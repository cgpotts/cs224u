"""Removes the answers from homework notebooks.
"""
import json
import re
import sys

__author__ = 'Chris Potts'


def main(src_filename):
    output_filename = src_filename.replace('_solved', '')
    doc = None
    with open(src_filename, 'rt') as f:
        doc = json.load(f)
        new_cells = []
        for i, cell in enumerate(doc['cells']):
            if cell['cell_type'] == 'code':
                cell_start = cell['source'][0]
                if not re.search(r"^#+\s*SOLUTION", cell_start):
                    removing = False
                    new_source = []
                    for line in cell['source']:
                        if "# <<<<<<<<<< TO BE COMPLETED" in line:
                            removing = True
                        if not removing:
                            new_source.append(line)
                        if "# >>>>>>>>>>" in line:
                            removing = False
                        cell['source'] = new_source
                    new_cells.append(cell)
            else:
                new_cells.append(cell)
        doc['cells'] = new_cells

    with open(output_filename, 'wt') as output:
        json.dump(doc, output)

if __name__ == '__main__':

    main(sys.argv[1])
