from collections import defaultdict
from typing import Iterable, Literal

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

    # Performance improvement 1:
    # version 1 was default name to alt names (dict[str, set])
    # version 2 is alt name to default name (dict[str, str])
    # version 2 is current version, which is faster

    # Performance improvement 2:
    Instead of building the backref dict each time we encounter a case where we
    find default names for both args in function add(new_default and new_alt),
    we can continue to have it updated on all operations.

    # Performance improvement 2:
    This is faster:
    ```
    for a in alts:
        self._dict[a] = new_default
    ```
    Than:
    `self._dict.update({a: new_default for a in alts})`



    Example:
    a, b, y, c are all alternate names of x
    >>> resolver = defaultname()
    >>> resolver.add("x", "a")
    'x'
    >>> resolver.add("x", "b")
    'x'
    >>> resolver.add("y", "a")
    'x'
    >>> resolver.add("c", "a")
    'x'
    >>> resolver._dict
    {'a': 'x', 'b': 'x', 'y': 'x', 'c': 'x'}
    >>> resolver._backref_dict == {'x': {'a', 'b', 'y', 'c'}}
    True
    >>> resolver._build_backref() == resolver._backref_dict
    True


    If we change order of names being added,
    a, c, x, b are alternate names of y
    >>> resolver = defaultname()
    >>> resolver.add("y", "a")
    'y'
    >>> resolver.add("c", "a")
    'y'
    >>> resolver.add("x", "a")
    'y'
    >>> resolver.add("x", "b")
    'y'
    >>> resolver._dict
    {'a': 'y', 'c': 'y', 'x': 'y', 'b': 'y'}
    >>> resolver._backref_dict == {'y': {'a', 'c', 'x', 'b'}}
    True
    >>> resolver._build_backref() == resolver._backref_dict
    True

    Unite different defaults
    Such as default (a) already has a parent (x)
    >>> resolver = defaultname()
    >>> resolver.add("x", "a")
    'x'
    >>> resolver.add("a", "b")
    'x'
    >>> resolver.add("y", "b")
    'x'
    >>> resolver._dict
    {'a': 'x', 'b': 'x', 'y': 'x'}
    >>> resolver._backref_dict == {'x': {'a', 'b', 'y'}}
    True
    >>> resolver._build_backref() == resolver._backref_dict
    True

    If two alternate names are passed
    >>> resolver = defaultname()
    >>> resolver.add("x", "a")
    'x'
    >>> resolver.add("x", "b")
    'x'
    >>> resolver.add("a", "b")
    'x'
    >>> resolver._dict
    {'a': 'x', 'b': 'x'}
    >>> resolver._backref_dict == {'x': {'a', 'b'}}
    True
    >>> resolver._build_backref() == resolver._backref_dict
    True

    # This test reflects the issue in large dataset example
    # Old code was not passing the large dataset test and then this test
    >>> resolver = defaultname()
    >>> resolver.add("x", "a")
    'x'
    >>> resolver.add("y", "b")
    'y'
    >>> resolver.add("x", "y")
    'x'
    >>> resolver._dict
    {'a': 'x', 'b': 'x', 'y': 'x'}
    >>> resolver._backref_dict == {'x': {'a', 'b', 'y'}}
    True
    >>> resolver._build_backref() == resolver._backref_dict
    True

    >>> resolver = defaultname()
    >>> resolver.add("x", "a")
    'x'
    >>> resolver.add("y", "b")
    'y'
    >>> resolver.add("a", "b")
    'x'
    >>> resolver._dict
    {'a': 'x', 'b': 'x', 'y': 'x'}
    >>> resolver._backref_dict == {'x': {'a', 'b', 'y'}}
    True
    >>> resolver._build_backref() == resolver._backref_dict
    True

    >>> import random
    >>> from tqdm import tqdm
    >>> random.seed(362)
    >>> resolver = defaultname()
    >>> def generate_random_string():
    ...     characters = "abcdef"
    ...     return "".join(random.choices(characters, k=6))
    >>> for i in tqdm(range(0, 1000000)):
    ...     a = generate_random_string()
    ...     b = generate_random_string()
    ...     _ = resolver.add(a, b)
    >>> resolver._build_backref().keys()
    dict_keys(['bbcaea'])

    >>> resolver._backref_dict.keys()
    dict_keys(['bbcaea'])

    >>> set(resolver._dict.values())
    {'bbcaea'}

    >>> set(resolver._build_backref().keys()) == set(resolver._dict.values())
    True

    >>> sorted(resolver._backref_dict.keys()) == sorted(resolver._build_backref().keys())
    True

    >>> sorted(resolver._backref_dict.values()) == sorted(resolver._build_backref().values())
    True

    >>> resolver._build_backref() == resolver._backref_dict
    True

    """

    def __init__(self):
        # main dict which is alternate name (key) to default name (value).
        self._dict: dict[str, str] = {}
        # back-references which is default to alternate names.
        self._backref_dict: defaultdict[str, set] = defaultdict(set)

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

        default = self.search(new_default)
        if default:
            new_default_found = True
            new_default = default

        default = self.search(new_alt)
        if default:
            new_alt_found = True
            new_alt = default

        if new_default == new_alt:
            return new_default

        # Unite two sets
        if new_default_found and new_alt_found:
            # alts = {
            #     my_alt
            #     for my_alt, my_default in self._dict.items()
            #     if my_default == new_alt
            # }
            # Performance improvement: instead of searching or new_alt in _dict's values we use:
            alts = self._backref_dict.pop(new_alt)

            self._backref_dict[new_default].update(alts)

            for a in alts:
                self._dict[a] = new_default

            self._dict[new_alt] = new_default
            self._backref_dict[new_default].add(new_alt)
            return new_default
        elif new_alt_found:
            self._dict[new_default] = new_alt
            self._backref_dict[new_alt].add(new_default)
            return new_alt
        else:
            self._dict[new_alt] = new_default
            self._backref_dict[new_default].add(new_alt)
            return new_default

    def _build_backref(self):
        """Build backreferences dict which is default to alternate names.
        This is only to be used in tests."""

        _backref_dict: dict[str, set] = defaultdict(set)
        for alt, default in self._dict.items():
            _backref_dict[default].add(alt)

        return _backref_dict

    def _backref_search(self, index: str):
        """Find backreferences which is default to alternate names.
        Not used. Can be used to check if name is a default or not."""

        return self._backref_dict[index]

    def search(self, index: str):
        """Searches dict for key, returns default name if found."""

        # Major bug fix: we don't set default -> default in _dict so we need to search in:
        if index in self._backref_dict:
            return index
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

        default = self.search(index)
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
        """Gets default name to alternate names list dataset."""

        list_backref_dict: dict[str, list] = {}
        for alt, default in self._dict.items():
            try:
                list_backref_dict[default].append(alt)
            except KeyError:
                list_backref_dict[default] = [alt]

        return list_backref_dict

    def get_cluster(self, index: str):
        dn = self.__getitem__(index)
        # Create new set to not corrupt _backref_dict dataset
        cluster = set(self._backref_dict[dn])
        cluster.add(dn)
        return cluster


def join_defaultname(a: defaultname, b: defaultname):
    pass


def batch_defaultname(defaults: Iterable, alts: Iterable, num_workers: int):
    pass


def replace_fromstart(haystack: str, needle: str, replace=""):
    """Replace needle with replace if haystack starts with needle.

    Faster than Regex.

    Args:
        haystack (str): string
        needle (str): search
        replace (str): replacement (optional, strips needle if not provided)

    Returns:
        str: replaced string or itself if haystack does not start with needle.
    """

    if haystack.startswith(needle):
        return replace + haystack[len(needle) :]
    return haystack
