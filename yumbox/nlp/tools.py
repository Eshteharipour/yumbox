from typing import Literal

import pandas as pd


class MapRed:
    def __init__(self):
        self.dictionary: dict[str, int] = {}
        self._df = None
        self.lowfreq = None
        self.stopwords = None

    def __call__(self, word):
        try:
            self.dictionary[word] = self.dictionary[word] + 1
        except KeyError:
            self.dictionary[word] = 1
        return self.dictionary[word]

    def __getitem__(self, key):
        return self.dictionary[key]

    def filter(self, count, op: Literal["gt", "lt", "eq", "gte", "lte"] = "eq"):
        D = self.dictionary
        if op == "gt":
            return {d: D[d] for d in D if D[d] > count}
        elif op == "lt":
            return {d: D[d] for d in D if D[d] < count}
        elif op == "eq":
            return {d: D[d] for d in D if D[d] == count}
        elif op == "gte":
            return {d: D[d] for d in D if D[d] >= count}
        elif op == "lte":
            return {d: D[d] for d in D if D[d] <= count}
        else:
            raise ValueError('Arg op must be one of: "gt", "lt", "eq", "gte", "lte"')

    def info(self):
        values = self.dictionary.values()
        imin = min(values)
        imax = max(values)
        count = len(values)
        print("min:", imin)
        print("max:", imax)
        print("count of values:", count)

    @property
    def df(self):
        if isinstance(self._df, type(None)):
            d = {"word": self.dictionary.keys(), "freq": self.dictionary.values()}
            df = pd.DataFrame(d)
            self._df = df
            return self._df
        else:
            return self._df

    def set_lowfreq(self, threshold):
        # list: 8.59it/s
        # set: 853901.44it/s
        self.lowfreq = set(self.filter(threshold, "lte"))

    def sort(self, order: Literal["asc", "desc"] = "asc"):
        if order == "asc":
            r = False
        elif order == "desc":
            r = True
        else:
            raise ValueError('Arg order must be one of "asc" or "desc"')
        self.dictionary = {
            k: v
            for k, v in sorted(
                self.dictionary.items(), key=lambda items: items[1], reverse=r
            )
        }


class defaultname:
    """Altertnate name to default name data structure.

    This resolves default name to itself and alternate names of a name
    to their default name. Kind of like the contecpt of defaultdict.
    Default name could be a "parent" name for example.

    # version 1 was default name to alt names (dict[str, set])
    # version 2 is alt name to default name (dict[str, str])
    # version 2 is current version, which is faster

    Example:
    x1, x2, y, y1 are all alternate names of x
    >>> resolver = defaultname()
    >>> resolver.add("x", "x1")
    'x'
    >>> resolver.add("x", "x2")
    'x'
    >>> resolver.add("y", "x1")
    'x'
    >>> resolver.add("y1", "x1")
    'x'
    >>> resolver._dict == {'x1': 'x', 'x2': 'x', 'y': 'x', 'y1': 'x'}
    True

    If we change order of names being added,
    x1, y1, x, x2 are alternate names of y
    >>> resolver = defaultname()
    >>> resolver.add("y", "x1")
    'y'
    >>> resolver.add("y1", "x1")
    'y'
    >>> resolver.add("x", "x1")
    'y'
    >>> resolver.add("x", "x2")
    'y'
    >>> resolver._dict == {'x1': 'y', 'y1': 'y', 'x': 'y', 'x2': 'y'}
    True

    Unite different defaults
    Such as default (x1) already has a parent (x)
    >>> resolver = defaultname()
    >>> resolver.add("x", "x1")
    'x'
    >>> resolver.add("x1", "x2")
    'x'
    >>> resolver.add("y", "x2")
    'x'
    >>> resolver._dict == {'x1': 'x', 'x2': 'x', 'y': 'x'}
    True

    If two alternate names are passed
    >>> resolver = defaultname()
    >>> resolver.add("x", "x1")
    'x'
    >>> resolver.add("x", "x2")
    'x'
    >>> resolver.add("x1", "x2")
    'x'
    >>> resolver._dict == {'x1': 'x', 'x2': 'x'}
    True

    """

    def __init__(self):
        self._dict: dict[str, str] = {}
        # self._backref_is_valid = True
        self._backref_dict: dict[str, list] = {}

    def add(self, new_default: str, new_alt: str):
        """Adds pair to dict returning the new default or default if found.

        1) If new_child and new_default both already have different defaults,
        unite their sets on new_default's default and delete new_child's default from dict.
        2) If new_default's default is found, add new_child to it's set.
        3) If new_child's default is found, add new_default to it's set.
        4) If no default is found, creates new default from new_default
        and add new_child to it's set.
        """

        if not new_default:
            raise ValueError("new_default is empty!")
        if not isinstance(new_default, str):
            raise ValueError("new_default is not str!")
        if not new_alt:
            raise ValueError("new_alt is empty!")
        if not isinstance(new_alt, str):
            raise ValueError("new_alt is not str!")

        if new_default == new_alt:
            return self.__getitem__(new_default)

        new_default_found = False
        new_alt_found = False

        default = self._search(new_default)
        if default:
            new_default_found = True
            new_default = default

        default = self._search(new_alt)
        if default:
            new_alt_found = True
            new_alt = default

        if new_default == new_alt:
            return new_default

        # self._backref_is_valid = False

        # unite two sets
        if new_default_found and new_alt_found:  # expensive
            # alts = self._backref_search(new_alt)
            # faster method
            alts = {
                my_alt
                for my_alt, my_default in self._dict.items()
                if my_default == new_alt
            }

            # for a in alts:
            #     self._dict[a] = new_default
            # faster method
            self._dict.update({a: new_default for a in alts})

            self._dict[new_alt] = new_default
            return new_default
        elif new_alt_found:
            self._dict[new_default] = new_alt
            return new_alt
        # elif new_default_found:
        #     self._dict[new_alt] = new_default
        #     return new_default
        else:
            self._dict[new_alt] = new_default
            return new_default

    def _build_backref(self):
        """Build backreferences dict which is default to alternate names."""

        self._backref_dict: dict[str, list[str]] = {}
        for alt, default in self._dict.items():
            try:
                self._backref_dict[default].append(alt)
            except KeyError:
                self._backref_dict[default] = [alt]

        # self._backref_is_valid = True

    def _backref_search(self, index: str):
        """Find backreferences which is default to alternate names."""

        # if not self._backref_is_valid:
        self._build_backref()

        return self._backref_dict[index]

    def _search(self, index: str):
        """Searches dict for key, returns default name if found."""

        try:
            return self._dict[index]
        except KeyError:
            return None

    def __getitem__(self, index: str):
        """Search for key, returns default if found, otherwise returns self."""

        if not index:
            raise ValueError("Index is empty!")
        if not isinstance(index, str):
            raise ValueError("Index is not str!")

        default = self._search(index)
        if default:
            return default
        else:
            return index

    def __call__(self, index):
        """Get Item wrapper but allows falsy value and returns it."""

        if not index:
            return index
        else:
            return self.__getitem__(index)

    def get_dataset(self):
        """Gets default name to alternate names list dataset.
        Uses _backref_dict and convert dict[str, set] to dict[str, list]."""

        # if not self._backref_is_valid:
        self._build_backref()

        return self._backref_dict
