from abc import abstractmethod, ABC
import numpy as np
from ..aux_classes import Rule


class RuleCost(ABC):
    """
    Abstract base class representing the cost associated with a rule or a set of rules.

    This class is designed to be subclassed by specific cost calculation implementations.
    Each subclass should provide a specific cost calculation strategy by overriding the
    `__call__` method. The `__call__` method allows instances of the subclass to be used
    as if they were functions, directly invoking the cost calculation.
    """

    @abstractmethod
    def __call__(self):
        """
        Abstract method to be implemented by subclasses to calculate and return the cost.

        This method should take relevant parameters as input (e.g., the rule or set of
        rules being evaluated) and return a numerical value representing the cost. The
        exact implementation details and parameters are to be defined in the subclass.

        Returns
        -------
        cost : float
            The calculated cost of the rule or set of rules. This is a numerical value
            indicating the cost or penalty associated with the rule(s) based on the
            specific calculation strategy implemented in the subclass.
        """


class Length(RuleCost):
    """
    Calculate the cost of a rule based on its length.
    """

    def __call__(self, temp_rule: Rule, *args, **kwargs) -> int:
        """
        Calculate and return the cost of the rule.

        Parameters
        ----------
        temp_rule : Rule
            The rule to calculate the cost for.

        Returns
        -------
        int
            The cost of the rule, defined as its length.
        """
        cost = len(temp_rule)
        return cost


class Gini(RuleCost):
    """
    Calculate the Gini cost of a split.
    
    .. math::
       Gini = 1 - \\sum_{i=1}^{n}(p_i)^2,

    where
    
    - :math:`p_i` is the probability of an object being classified to a particular class.

    """

    def __call__(self, counts: np.ndarray, *args, **kwargs) -> float:
        """
        Calculate and return the Gini cost for a node.

        Parameters
        ----------
        counts : np.ndarray
            An array containing the counts of each class in a node.

        Returns
        -------
        float
            The Gini cost for the node.
        """
        probs = np.divide(counts, np.sum(counts))
        cost = 1 - np.sum(np.square(probs))
        return cost


class Mixed(RuleCost):
    """
    Calculate the mixed cost, combining class separation and data selection terms with
    a weighting parameter.
    
    The mixed cost for a rule is calculated as follows:

    .. math::
       Mixed = w \\times class\\_separation\\_term + (1 - w) \\times data\\_selection\\_term,

    where

    - :math:`w` is the weighting parameter to balance class separation and data selection terms,
    - :math:`class\\_separation\\_term` is :math:`1 - \\left(1 - \\left(\\frac{\\min(covers)}{number\\_of\\_classes}\\right)\\right)`,
    - :math:`data\\_selection\\_term` is :math:`1 - \\left(\\frac{number\\_of\\_classes}{total\\_samples}\\right)`.

    
    """

    def __init__(self, w: float = 0.7) -> None:
        """
        Initialize the mixed cost calculation with a weighting parameter.

        Parameters
        ----------
        w : float, optional, default=0.7
            Weighting parameter to balance class separation and data selection terms.
        """
        self.w = w

    def __call__(self, covers: np.ndarray, y: np.ndarray, *args, **kwargs) -> float:
        """
        Calculate and return the mixed cost for a rule.

        Parameters
        ----------
        covers : np.ndarray
            Array of cover sizes for the classes.

        y : np.ndarray
            The target array.

        Returns
        -------
        float
            The mixed cost for the rule.
        """
        class_separation_term = 1 - (1 - (np.min(covers) / covers.shape[0]))
        data_selection_term = 1 - (covers.shape[0] / y.shape[0])
        cost = self.w * class_separation_term + (1 - self.w) * data_selection_term
        return cost


class MixedSigmoid(RuleCost):
    """
    Calculate the mixed cost with a sigmoid adjustment based on weighting and alpha
    parameters.
    
    The sigmoid-adjusted mixed cost for a rule is calculated as follows:

    .. math::
       MixedSigmoid = \\frac{1}{1 + e^{-\\alpha (w \\times class\\_separation\\_term + (1 - w) \\times data\\_selection\\_term - 0.5)}},

    where

    - :math:`w` is the weighting parameter to balance class separation and data selection terms,
    - :math:`\\alpha` is the scaling parameter to adjust the steepness of the sigmoid function,
    - :math:`class\\_separation\\_term` is :math:`1 - \\left(1 - \\left(\\frac{\\min(covers)}{number\\_of\\_classes}\\right)\\right)`,
    - :math:`data\\_selection\\_term` is :math:`1 - \\left(\\frac{number\\_of\\_classes}{total\\_samples}\\right)`.    

    """

    def __init__(self, w: float = 0.7, alpha: float = 10) -> None:
        """
        Initialize the sigmoid-adjusted mixed cost calculation with weighting and
        alpha parameters.

        Parameters
        ----------
        w : float, optional, default=0.7
            Weighting parameter to balance class separation and data selection terms.

        alpha : float, optional, default=10
            Scaling parameter to adjust the steepness of the sigmoid function.
        """
        self.w = w
        self.alpha = alpha

    def __call__(self, covers: np.ndarray, y: np.ndarray, *args, **kwargs) -> float:
        """
        Calculate and return the sigmoid-adjusted mixed cost for a rule.

        Parameters
        ----------
        covers : np.ndarray
            Array of cover sizes for the classes.

        y : np.ndarray
            The target array.

        Returns
        -------
        float
            The sigmoid-adjusted mixed cost for the rule.
        """
        class_separation_term = 1 - (1 - (np.min(covers) / covers.shape[0]))
        data_selection_term = 1 - (covers.shape[0] / y.shape[0])
        cost = 1 / (
            1
            + np.exp(
                -self.alpha
                * (
                    self.w * class_separation_term
                    + (1 - self.w) * data_selection_term
                    - 0.5
                )
            )
        )
        return cost
