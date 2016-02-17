"""Test l5pc example"""

import os
import sys

import nose.tools as nt

L5PC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '../../examples/l5pc'))

sys.path.insert(0, L5PC_PATH)

import bluepyopt
bluepyopt.neuron.h.nrn_load_dll(
    os.path.join(
        L5PC_PATH,
        'x86_64/.libs/libnrnmech.so'))


def load_from_json(filename):
    """Load structure from json"""

    import json

    with open(filename) as json_file:
        return json.load(json_file)


def dump_to_json(content, filename):
    """Dump structure to json"""

    import json

    with open(filename, 'w') as json_file:
        return json.dump(content, json_file, indent=4, separators=(',', ': '))


def test_import():
    """L5PC: test import"""

    import l5pc_template  # NOQA
    import l5pc_evaluator  # NOQA
    import opt_l5pc  # NOQA

    # Delete the optimisation inside the module
    del opt_l5pc.opt


class TestL5PCTemplate(object):

    """Test L5PC template"""

    def __init__(self):
        self.l5pc_cell = None

    def setup(self):
        """Set up class"""
        sys.path.insert(0, L5PC_PATH)

        import l5pc_template  # NOQA

        self.l5pc_cell = l5pc_template.create()
        nt.assert_is_instance(
            self.l5pc_cell,
            bluepyopt.electrical.celltemplate.CellTemplate)

    def teardown(self):
        """Teardown"""
        pass


class TestL5PCEvaluator(object):

    """Test L5PC evaluator"""

    def __init__(self):
        self.l5pc_evaluator = None

    def setup(self):
        """Set up class"""

        import l5pc_evaluator  # NOQA

        self.l5pc_evaluator = l5pc_evaluator.create()

        nt.assert_is_instance(
            self.l5pc_evaluator,
            bluepyopt.electrical.cellevaluator.CellEvaluator)

    def test_eval(self):
        """L5PC: test evaluation of l5pc evaluator"""

        # Parameters in release circuit model
        parameters = {
            'gIhbar_Ih.basal': 0.000080,
            'gNaTs2_tbar_NaTs2_t.apical': 0.026145,
            'gSKv3_1bar_SKv3_1.apical': 0.004226,
            'gIhbar_Ih.apical': 0.000080,
            'gImbar_Im.apical': 0.000143,
            'gNaTa_tbar_NaTa_t.axonal': 3.137968,
            'gK_Tstbar_K_Tst.axonal': 0.089259,
            'gamma_CaDynamics_E2.axonal': 0.002910,
            'gNap_Et2bar_Nap_Et2.axonal': 0.006827,
            'gSK_E2bar_SK_E2.axonal': 0.007104,
            'gCa_HVAbar_Ca_HVA.axonal': 0.000990,
            'gK_Pstbar_K_Pst.axonal': 0.973538,
            'gSKv3_1bar_SKv3_1.axonal': 1.021945,
            'decay_CaDynamics_E2.axonal': 287.198731,
            'gCa_LVAstbar_Ca_LVAst.axonal': 0.008752,
            'gamma_CaDynamics_E2.somatic': 0.000609,
            'gSKv3_1bar_SKv3_1.somatic': 0.303472,
            'gSK_E2bar_SK_E2.somatic': 0.008407,
            'gCa_HVAbar_Ca_HVA.somatic': 0.000994,
            'gNaTs2_tbar_NaTs2_t.somatic': 0.983955,
            'gIhbar_Ih.somatic': 0.000080,
            'decay_CaDynamics_E2.somatic': 210.485284,
            'gCa_LVAstbar_Ca_LVAst.somatic': 0.000333
        }

        result = self.l5pc_evaluator.evaluate_with_dicts(
            param_dict=parameters)

        expected_results = load_from_json('expected_results.json')

        # Use two lines below to update expected result
        # expected_results['TestL5PCEvaluator.test_eval'] = result
        # dump_to_json(expected_results, 'expected_results.json')

        nt.assert_items_equal(
            result,
            expected_results['TestL5PCEvaluator.test_eval'])

    def teardown(self):
        """Teardown"""
        pass
