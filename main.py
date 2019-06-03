import ai.algorithms as algorithms
import ai.environments as environments

import cmd
import readline
import sys

class AIShell(cmd.Cmd):
    intro = 'Type help or ? to list commands.\n'
    prompt = '(ai) '

    def do_run(self, arg):
        """Run an algorithm: RUN MonteCarloPrediction Blackjack

        If the algorithm and environment are not specified, you will be given some to choose from.
        """
        algorithm = None
        environment = None
        args = arg.split()
        if len(args) >= 1:
            if not hasattr(algorithms, args[0]):
                print('ERROR: The algorithm %s is not supported'.format(args[0]))
            else:
                algorithm = getattr(algorithms, args[0])
        if len(args) >= 2:
            if not hasattr(environments, args[1]):
                print('ERROR: The environment %s is not supported'.format(args[1]))
            else:
                environment = getattr(environments, args[1])

        self.run(algorithm, environment)

    def do_list_algorithms(self, arg):
        'List all available algorithms'
        for module in algorithms.__all__:
            print(module)

    def do_list_environments(self, arg):
        'List all available environments'
        for module in environments.__all__:
            print(module)

    def do_describe_algorithm(self, arg):
        """Print details for a given algorithm.

        Usage: describe_algorithm <algorithm name>
        """
        args = arg.split()
        if len(args) != 1:
            self.print_usage_error(self.do_describe_algorithm)
            return
        if not hasattr(algorithms, args[0]):
            print('ERROR: The algorithm %s is not supported'.format(args[0]))
            return
        algorithm = getattr(algorithms, args[0])
        print(algorithm.__doc__)

    def do_describe_environment(self, arg):
        """Print details for a given environment.

        Usage: describe_environment <environment name>
        """
        args = arg.split()
        if len(args) != 1:
            self.print_usage_error(self.do_describe_environment)
            return
        if not hasattr(environments, args[0]):
            print('ERROR: The environment %s is not supported'.format(args[0]))
            return
        environment = getattr(environments, args[0])
        print(environment.__doc__)

    def do_exit(self, arg):
        'Exit the shell'
        return True

# ------------------------------

    def print_usage_error(self, method):
        print('Error running {}'.format(method.__name__[3:]))
        usage = method.__doc__
        for line in method.__doc__.split('\n'):
            if 'Usage:' in line:
                usage = line.strip()
        print(usage)


    def run(self, alg_type, env_type):
        with env_type() as env:
            alg_instance = alg_type(env)
            alg_instance.run()

    def list_algorithms(self, filters):
        return []

    def list_environments(self, filters):
        return []

if __name__ == '__main__':
    AIShell().cmdloop()


