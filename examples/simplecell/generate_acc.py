#!/usr/bin/env python

'''Example for generating a mixed JSON/ACC Arbor cable cell description (with optional axon-replacement)

 $ python generate_acc.py --output-dir test_acc/ --replace-axon

 Will save 'simple_cell.json', 'simple_cell_label_dict.acc' and 'simple_cell_decor.acc'
 into the folder 'test_acc' that can be loaded in Arbor with:
     'cell_json, morpho, decor, labels = \
        ephys.create_acc.read_acc("test_acc/simple_cell_cell.json")'
 An Arbor cable cell is then created with
     'cell = arbor.cable_cell(morphology=morpho, decor=decor, labels=labels)'
 The resulting cable cell can be output to ACC for visual inspection 
 and e.g. validating/deriving custom Arbor locset/region/iexpr 
 expressions in the Arbor GUI (File > Cable cell > Load) using
     'arbor.write_component(cell, "simple_cell_cable_cell.acc")'
'''
import argparse

from bluepyopt import ephys

import simplecell_model
from generate_hoc import param_values


def main():
    '''main'''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('-o', '--output-dir', dest='output_dir',
                        help='Output directory for JSON/ACC files')
    parser.add_argument('-ra', '--replace-axon', action='store_true',
                        help='Replace axon with Neuron-dependent policy')
    args = parser.parse_args()

    cell = simplecell_model.create(do_replace_axon=args.replace_axon)
    if args.replace_axon:
        nrn_sim = ephys.simulators.NrnSimulator()
        cell.instantiate_morphology_3d(nrn_sim)

    # Add modcc-compiled external mechanisms catalogues here
    # ext_catalogues = {'cat-name': 'path/to/nmodl-dir', ...}

    if args.output_dir is not None:
        cell.write_acc(args.output_dir,
                       param_values,
                       # ext_catalogues=ext_catalogues,
                       create_mod_morph=True)
    else:
        output = cell.create_acc(
            param_values,
            template='acc/*_template.jinja2',
            # ext_catalogues=ext_catalogues,
            create_mod_morph=True)
        for el, val in output.items():
            print("%s:\n%s\n" % (el, val))


if __name__ == '__main__':
    main()
