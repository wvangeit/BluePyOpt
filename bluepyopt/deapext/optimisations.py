from deap import algorithms
from bluepyopt.deapext.optimisations import Optimisation

class DifferentialEvolutionOptimisation(Optimisation):
    """Class for Differential Evolution optimisation"""

    def run(self, max_ngen, offspring_size=None, continue_cp=False, cp_filename=None, cp_frequency=1):
        """Run the Differential Evolution algorithm"""

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

