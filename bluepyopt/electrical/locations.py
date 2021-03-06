"""Location classes"""

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


import bluepyopt as nrp


class Location(object):

    """EPhys feature"""

    def __init__(self, name):
        """Constructor"""

        self.name = name

# TODO make all these locations accept a cell name
# TODO instantiate should get the entire simulation environment
# TODO find better/more general name for this
# TODO specify in document abrevation comp=compartment, sec=section, ...


class NrnSeclistCompLocation(Location):

    """Compartment in a sectionlist"""

    def __init__(
            self,
            name,
            seclist_name=None,
            sec_index=None,
            comp_x=None):
        """Constructor"""

        Location.__init__(self, name)
        self.seclist_name = seclist_name
        self.sec_index = sec_index
        self.comp_x = comp_x

    def instantiate(self, cell):
        """Find the instantiate compartment"""

        import itertools

        isectionlist = getattr(cell.icell, self.seclist_name)

        # Get nth element of isectionlist
        # Sectionlists don't support direct indexing
        isection = next(
            itertools.islice(
                isectionlist,
                self.sec_index,
                self.sec_index + 1))
        icomp = isection(self.comp_x)

        return icomp

    def __str__(self):
        """String representation"""

        return '%s[%s](%s)' % (self.seclist_name, self.sec_index, self.comp_x)


class NrnSeclistLocation(Location):

    """Section in a sectionlist"""

    def __init__(
            self,
            name,
            seclist_name=None,
            sec_index=None):
        """Constructor"""

        Location.__init__(self, name)
        self.seclist_name = seclist_name

    def instantiate(self, cell):
        """Find the instantiate compartment"""

        isectionlist = getattr(cell.icell, self.seclist_name)

        return (isection for isection in isectionlist)

    def __str__(self):
        """String representation"""

        return '%s' % (self.seclist_name)


class NrnSeclistSecLocation(Location):

    """Section in a sectionlist"""

    def __init__(
            self,
            name,
            seclist_name=None,
            sec_index=None):
        """Constructor"""

        Location.__init__(self, name)
        self.seclist_name = seclist_name
        self.sec_index = sec_index

    def instantiate(self, cell):
        """Find the instantiate compartment"""

        import itertools

        isectionlist = getattr(cell.icell, self.secist_name)

        # Get nth element of isectionlist
        # Sectionlists don't support direct indexing
        isection = next(
            itertools.islice(
                isectionlist,
                self.sec_index,
                self.sec_index + 1))

        return isection

    def __str__(self):
        """String representation"""

        return '%s[%s]' % (self.seclist_name, self.sec_index)


class NrnSomaDistanceCompLocation(Location):

    """Compartment at distance from soma"""

    def __init__(self, name, soma_distance=None, seclist_name=None):
        """Constructor"""

        Location.__init__(self, name)
        self.soma_distance = soma_distance
        self.seclist_name = seclist_name

    # TODO this definitely has to be unit-tested
    # TODO add ability to specify origin
    # TODO rename 'seg' in 'compartment' everywhere
    def instantiate(self, cell):
        """Find the instantiate compartment"""

        soma = cell.icell.soma[0]

        nrp.neuron.h.distance(0, 0.5, sec=soma)

        iseclist = getattr(cell.icell, self.seclist_name)

        icomp = None
        max_diam = 0.0

        for isec in iseclist:
            start_distance = nrp.neuron.h.distance(1, 0.0, sec=isec)
            end_distance = nrp.neuron.h.distance(1, 1.0, sec=isec)

            min_distance = min(start_distance, end_distance)
            max_distance = max(start_distance, end_distance)

            if min_distance <= self.soma_distance <= end_distance:
                comp_x = float(self.soma_distance - min_distance) / \
                    (max_distance - min_distance)

                comp_diam = isec(comp_x).diam

                if comp_diam > max_diam:
                    icomp = isec(comp_x)

        if icomp is None:
            raise Exception(
                'No comp found at %s distance from soma' %
                self.soma_distance)

        return icomp

    def __str__(self):
        """String representation"""

        return '%f mum from soma in %s' % (
            self.soma_distance, self.seclist_name)
