__author__ = 'johannes'


class Variable(object):
    """
    A variable in a constrained satisfaction problem.
    """
    def __init__(self, name, domain, value=None):
        """
        The constructor for a variable.
        :param domain: The domain of this variable. A list of possible values.
        :param value: The current value of this variable. Might be None if
                      it is not set yet.
        """
        self.name = name
        self.domain = domain
        self.value = value
        self.peers = []

    def set_value(self, value):
        """
        The setter for a variable, which checks whether the value is in the
        domain.
        :param value:
        """
        if value not in self.domain and value is not None:
            raise ValueError

        self.value = value

    def get_value(self):
        return self.value

    def __str__(self):
        """
        Returns a string representation of a variable
        :return: A str object
        """
        return "{} = {}".format(self.name, self.value)

    def __repr__(self):
        return str(self)


class UnequalConstraint(object):
    """
    A constraint in a constrained satisfaction problem.
    """
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    def consistent(self):
        """
        Test whether the values of the two variables are consistent with
        this constraint. I.e. they are either unequal or one of them is
        None.
        :return: A bool
        """
        if self.var1.get_value() is None or self.var2.get_value() is None:
            return True

        return self.var1.value != self.var2.value

    def satisfied(self):
        """
        Test whether the values of the two variables satisfy this
        constraint. I.e. they are unequal and none of them is None.
        :return: A bool
        """

        if self.var1.get_value() is None or self.var2.get_value() is None:
            return False

        return self.var1.get_value() != self.var2.get_value()

    def __str__(self):
        """
        Returns a string representation of a constraint
        :return: A str object
        """
        return "{} != {} ({})".format(self.var1.name,
                                      self.var2.name,
                                      self.satisfied())


class ConstrainedSatisfactionProblem(object):
    """
    The main CSP data structure. It contains all variables and all
    constraints.
    """
    def __init__(self, variables, constraints):
        """
        The constructor of a CSP. It automatically generates the peers
        member for each variable, i.e. the list of other variables that
        have a common constraint with a variable.
        :param variables:
        :param constraints:
        :return:
        """
        self.variables = variables
        self.constraints = constraints
        for c in constraints:
            c.var1.peers.append(c.var2)
            c.var2.peers.append(c.var1)

    def complete(self):
        """
        Test whether all constraints in this CSP are satisfied.
        :return: A bool
        """
        return all((constraint.satisfied() for constraint in self.constraints))

    def consistent(self):
        """
        Test whether all constraints in this CSP are satisfied.
        :return: A bool
        """
        return all((constraint.consistent() for constraint in self.constraints))

    def get_constraints_for_variable(self, var):
        """
        Returns a list of all constraints that concern a variable.
        :param var: A variable
        :return: A list of UnequalConstraints
        """
        return (constraint for constraint in self.constraints
                if var.name in [constraint.var1.name, constraint.var2.name])

    def __str__(self):
        """
        Returns a string representation of a CSP
        :return: A str object
        """
        _str = "Variables:\n"
        for variable in self.variables:
            _str += "    {}\n".format(str(variable))
        _str += "\nConstraints:\n"
        for constraint in self.constraints:
            _str += "    {}\n".format(str(constraint))
        return _str
