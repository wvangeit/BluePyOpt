import random
import functools
import deap.tools
import numpy
from deap import algorithms
from bluepyopt.deapext.optimisations import DifferentialEvolutionOptimisation

class DifferentialEvolutionOptimisation(Optimisation):
    """Class for Differential Evolution optimisation"""

    def run(self, max_ngen, offspring_size=None, continue_cp=False, cp_filename=None, cp_frequency=1):
        '''Run the Differential Evolution algorithm'''

        # Generate initial population
        population = self.toolbox.population(n=offspring_size)

        # Evaluate fitness of individuals
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Run the Differential Evolution algorithm
        for gen in range(1, max_ngen + 1):
            population = algorithms.eaMuCommaLambda(population, self.toolbox, mu=offspring_size, lambda_=offspring_size, cxpb=self.cxpb, mutpb=self.mutpb, ngen=1)

        return population

    def setup_deap(self):
        """Set up optimisation"""

        # Number of objectives
        OBJ_SIZE = len(self.evaluator.objectives)

        # Set random seed
        random.seed(self.seed)

        # Eta parameter of crossover / mutation parameters
        # Basically defines how much they 'spread' solution around
        # The lower this value, the more spread
        ETA = self.eta

        # Number of parameters
        IND_SIZE = len(self.evaluator.params)
        if IND_SIZE == 0:
            raise ValueError(
                "Length of evaluator.params is zero. At least one "
                "non-fix parameter is needed to run an optimization."
            )

        # Bounds for the parameters

        LOWER = []
        UPPER = []

        for parameter in self.evaluator.params:
            LOWER.append(parameter.lower_bound)
            UPPER.append(parameter.upper_bound)

        # Register the 'uniform' function
        self.toolbox.register("uniformparams", utils.uniform, LOWER, UPPER,
                              IND_SIZE)

        # Register the individual format
        # An indiviual is create by WSListIndividual and parameters
        # are initially
        # picked by 'uniform'
        self.toolbox.register(
            "Individual",
            deap.tools.initIterate,
            functools.partial(WSListIndividual, obj_size=OBJ_SIZE),
            self.toolbox.uniformparams)

        # Register the population format. It is a list of individuals
        self.toolbox.register(
            "population",
            deap.tools.initRepeat,
            list,
            self.toolbox.Individual)

        # Register the evaluation function for the individuals
        # import deap_efel_eval1
        self.toolbox.register(
            "evaluate",
            self.evaluator.init_simulator_and_evaluate_with_lists
        )

        # Register the mate operator
        self.toolbox.register(
            "mate",
            deap.tools.cxSimulatedBinaryBounded,
            eta=ETA,
            low=LOWER,
            up=UPPER)

        # Register the mutation operator
        self.toolbox.register(
            "mutate",
            deap.tools.mutPolynomialBounded,
            eta=ETA,
            low=LOWER,
            up=UPPER,
            indpb=0.5)

        # Register the variate operator
        self.toolbox.register("variate", deap.algorithms.varAnd)

        # Register the selector (picks parents from population)
        if self.selector_name == 'IBEA':
            self.toolbox.register("select", tools.selIBEA)
        elif self.selector_name == 'NSGA2':
            self.toolbox.register("select", deap.tools.emo.selNSGA2)
        elif self.selector_name == 'DifferentialEvolution':
            self.toolbox.register("select", DifferentialEvolutionOptimisation)
        else:
            raise ValueError('DEAPOptimisation: Constructor selector_name '
                             'argument only accepts "IBEA", "NSGA2" or "DifferentialEvolution"')

        import copyreg
        import types
        copyreg.pickle(types.MethodType, utils.reduce_method)

        if self.use_scoop:
            if self.map_function:
                raise Exception(
                    'Impossible to use scoop is providing self '
                    'defined map function: %s' %
                    self.map_function)

            from scoop import futures
            self.toolbox.register("map", futures.map)

        elif self.map_function:
            self.toolbox.register("map", self.map_function)

    def run(self,
            max_ngen=10,
            offspring_size=None,
            continue_cp=False,
            cp_filename=None,
            cp_frequency=1,
            cp_period=None,
            parent_population=None,
            terminator=None):
        """Run optimisation"""
        # Allow run function to override offspring_size
        # TODO probably in the future this should not be an object field
        # anymore
        # keeping for backward compatibility
        if offspring_size is None:
            offspring_size = self.offspring_size

        # Generate the population object
        if parent_population is not None:

            if len(parent_population) != offspring_size:
                offspring_size = len(parent_population)
                self.offspring_size = len(parent_population)
                logger.warning(
                    'The length of the provided population is different from '
                    'the offspring_size. The offspring_size will be '
                    'overwritten.'
                )

            OBJ_SIZE = len(self.evaluator.objectives)
            IND_SIZE = len(self.evaluator.params)

            pop = []
            for ind in parent_population:

                if len(ind) != IND_SIZE:
                    raise Exception(
                        'The length of the provided individual is not equal '
                        'to the number of parameter in the evaluator ')

                pop.append(WSListIndividual(ind, obj_size=OBJ_SIZE))

        else:
            pop = self.toolbox.population(n=offspring_size)

        stats = deap.tools.Statistics(key=lambda ind: ind.fitness.sum)
        import numpy
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        param_names = []
        if hasattr(self.evaluator, "param_names"):
            param_names = self.evaluator.param_names

        pop, hof, log, history = algorithms.eaAlphaMuPlusLambdaCheckpoint(
            pop,
            self.toolbox,
            offspring_size,
            self.cxpb,
            self.mutpb,
            max_ngen,
            stats=stats,
            halloffame=self.hof,
            cp_frequency=cp_frequency,
            cp_period=None,
            continue_cp=continue_cp,
            cp_filename=cp_filename,
            terminator=terminator,
            param_names=param_names)

        # Update hall of fame
        self.hof = hof

        return pop, self.hof, log, history

