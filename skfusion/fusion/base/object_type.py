
__all__ = ['ObjectType']


class ObjectType(object):
    """Object type used for fusion.

    Attributes
    ----------
    name :
    rank :
    """
    def __init__(self, name, rank=5):
        self.name = name
        self.rank = rank

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return self.name == other

    def __ne__(self, other):
        return self.name != other

    def __str__(self):
        return self.name

    def set_rank(self, rank):
        """Set factorization rank.

        Parameters
        ----------
        rank :
        """
        self.rank = rank
