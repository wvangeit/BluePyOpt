"""Responses classes"""

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


import pandas


class Response(object):

    """Response to stimulus"""

    def __init__(self, name):
        """Constructor"""

        self.response = None
        self.name = name


class TimeVoltageResponse(Response):

    """Response to stimulus"""

    def __init__(self, name, time=None, voltage=None):
        """Constructor"""

        Response.__init__(self, name)

        self.response = pandas.DataFrame()
        self.response['time'] = pandas.Series(time)
        self.response['voltage'] = pandas.Series(voltage)

    def read_csv(self, filename):
        """Load response from csv file"""

        self.response.read_csv(filename)

    def to_csv(self, filename):
        """Write response to csv file"""

        self.response.to_csv(filename)

    def __getitem__(self, index):
        """Return item at index"""

        return self.response.__getitem__(index)

    # This plot has to be generalised to several subplots
    def plot(self, axes):
        """Plot the response"""

        axes.plot(
            self.response['time'],
            self.response['voltage'],
            label='%s' %
            self.name)
