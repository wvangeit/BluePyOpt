"""Run simple cell optimisation"""

"""
Copyright (c) 2016, EPFL/Blue Brain Project

 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.

 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

"""
This optimisation is based on L5PC optimisations developed by Etay Hay in the
context of the BlueBrain project
"""


# pylint: disable=R0914

import bluepyopt

# TODO store definition dicts in json
# TODO rename 'score' into 'objective'
# TODO add functionality to read settings of every object from config format

import l5pc_evaluator
evaluator = l5pc_evaluator.create()


def evaluate(parameter_array):
    """Global evaluate function"""

    return evaluator.evaluate(parameter_array)

opt = bluepyopt.Optimisation(
    evaluator=evaluator,
    eval_function=evaluate,
    offspring_size=2,
    use_scoop=True)


def main():
    """Main"""

    import argparse
    parser = argparse.ArgumentParser(description='L5PC example')
    parser.add_argument('--start', action="store_true")
    parser.add_argument('--continue_cp', action="store_true")
    parser.add_argument('--analyse', action="store_true")

    # TODO read checkpoint filename from arguments
    cp_filename = 'checkpoint.pkl'

    args = parser.parse_args()

    if args.start or args.continue_cp:
        opt.run(
            max_ngen=200,
            continue_cp=args.continue_cp,
            cp_filename=cp_filename)

    if args.analyse:
        import l5pc_analysis
        l5pc_analysis.analyse_cp(opt=opt, cp_filename=cp_filename)
        l5pc_analysis.analyse_releasecircuit_model(opt=opt)

        import matplotlib.pyplot as plt
        plt.show()

if __name__ == '__main__':
    main()
