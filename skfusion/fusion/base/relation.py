
__all__ = ['Relation']


class Relation(object):
    """Relation data set used for fusion.

    Attributes
    ----------
    data :
    mask :
    row_names :
    col_names :
    row_type :
    col_type :
    """
    def __init__(self, data, row_type, col_type, mask=None,
                 row_names=None, col_names=None, **kwargs):
        self.__dict__.update(vars())
        self.__dict__.update(kwargs)
        del self.__dict__['kwargs']
        del self.__dict__['self']

    def __hash__(self):
        return hash(self.__str__())

    def __str__(self):
        return 'R(%s, %s)' % (str(self.row_type), str(self.col_type))

    def __repr__(self):
        return 'R(%s, %s)' % (str(self.row_type), str(self.col_type))
