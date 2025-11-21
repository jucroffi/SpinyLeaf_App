# This file is part of ladybug.py.

#     ladybug.py is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     ladybug.py is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with ladybug.py.  If not, see <http://www.gnu.org/licenses/>.


from csv import DictReader, DictWriter
import itertools
from inspect import getargspec
from graph_utils import evaluation_order


def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


class Table(object):
    """A model for handling a CSV file"""
    columns = None
    instances = {}

    @classmethod
    def getinstance(cls):
        if cls not in cls.instances:
            cls.instances[cls] = cls()
        return cls.instances[cls]

    def __init__(self):
        super(Table, self).__init__()
        self.__result_class = None
        self.initalize_fields()
        if self.columns is None:
            self.columns = list(sorted(
                (name for name, _ in self.fields),
                key=lambda f: self.get_field(f)[1].field_order_id
            ))
        self.static_columns = list(sorted(
            (name for name, _ in self.__static_fields),
            key=lambda f: self.get_field(f)[1].field_order_id
        ))

    @property
    def fields(self):
        return (
            self.get_field(name)
            for name in dir(self)
            if isinstance(getattr(self, name), BaseField)
        )

    @property
    def __static_fields(self):
        return (
            self.get_field(name)
            for name in dir(self)
            if isinstance(getattr(self, name), StaticField)
        )

    @property
    def dynamic_fields(self):
        return (
            (name, self.get_field(name)[1],)
            for name in dir(self)
            if isinstance(getattr(self, name), DynamicField)
        )

    @classmethod
    def get_field(self, name):
        return (name, getattr(self, name))

    def get_field_of_column(self, name):
        for name, field in self.fields:
            if field.column == name:
                return field
        else:
            return None

    def initalize_fields(self):
        for name, member in self.fields:
            if member.column is None:
                member.column = name

        graph = dict()
        graph.update({name: [] for name, _ in self.__static_fields})
        graph.update({
            name: list(field.depends)
            for name, field in self.dynamic_fields
        })
        self.order_of_evaluation = list(evaluation_order(graph.keys(), graph))

    @classmethod
    def manager(cls):
        return Manager(cls)

    @classmethod
    def open(cls, csvfile, **kwargs):
        fieldnames = cls().columns
        manager = Manager(cls)
        with open(csvfile) as csvfile:
            manager.read_data(DictReader(csvfile, fieldnames, **kwargs))
        return manager

    @classmethod
    def create(cls):
        return Manager(cls)

    @property
    def result_class(self):
        if self.__result_class:
            return self.__result_class
        other_class = self.__class__

        class result_class(dict):

            def __init__(self, row):
                super(result_class, self).__init__()
                other = other_class.getinstance()
                self.row = row
                for name in other.static_columns:
                    value = row[name] if name in row else ""
                    self.update({name: other.get_field(name)[1](value)})

            def copy(self):
                return dict(self.iteritems())

        result_class.__name__ = self.__class__.__name__ + "Object"
        self.__result_class = result_class
        return result_class


class BaseField(object):
    """Base class for fields"""
    _counter = itertools.count()

    def __init__(self):
        super(BaseField, self).__init__()
        self.field_order_id = self._counter.next()


def Field(**kwargs):
    if "format" in kwargs:
        return StaticField(**kwargs)
    elif "function" in kwargs:
        return DynamicField(**kwargs)
    elif not kwargs:
        return StaticField()
    else:
        raise ValueError("A format or a function is required")


def field(*args, **kwargs):
    if len(args) == 1 and not kwargs:
        return Field(function=args[0], depends=getargspec(args[0]).args)
    elif kwargs.keys() == ["column"] and not args:
        column = kwargs["column"]

        def decorator(func):
            args = getargspec(func).args
            return Field(function=func, depends=args, column=column)
        return decorator
    else:
        raise ValueError()


class StaticField(BaseField):
    """A field that's mapped to a column of a CSV file"""
    def __init__(self, format=str, column=None, default=None, empty_value=lambda s: s == ""):
        super(StaticField, self).__init__()
        self.format = format
        self.column = column
        self.default = default
        self.empty_value = empty_value

    def __call__(self, value):
        if self.empty_value(value):
            return self.default
        else:
            return self.format(value)


class DynamicField(BaseField):
    """A field with a value calculated on the fly"""
    def __init__(self, function, column=None, depends=None):
        super(DynamicField, self).__init__()
        self.function = function
        self.column = column
        self.depends = depends if depends is not None else list()

    def __call__(self, **kwargs):
        return self.function(**kwargs)


class Manager(object):
    """A manager for an opened CSV file based on a Table"""
    def __init__(self, model, data=None, include=None):
        super(Manager, self).__init__()
        self.model = model.getinstance()
        self._data = data
        if include is not None:
            self._include = include
        else:
            self._include = range(len(data)) if data else list()

    def read_data(self, reader):
        self._data = list()
        self._include = list()
        index = 0
        for row in reader:
            self._data.append(self.model.result_class(row))
            self._include.append(index)
            index += 1

    def append_rows(self, source, **kwargs):
        """Append rows from a data source"""
        index = len(self._include)
        if not self._data:
            self._data = list()
        if not self._include:
            self._include = list()
        for source_row in source:
            row = dict()
            for field, source_field in kwargs.iteritems():
                row[field] = source_row[source_field]
            self._data.append(self.model.result_class(row))
            self._include.append(index)
            index += 1

    def export_rows(self, **kwargs):
        for row in self.rows:
            target_row = dict()
            for target_field, field in kwargs.iteritems():
                target_row[target_field] = row[field]
            yield target_row

    @property
    def rows(self):
        for i in self._include:
            yield self.eval_row(i)

    def eval_row(self, i):
        row = dict(self._data[i].iteritems())
        left = (name for name in self.model.order_of_evaluation if name not in row.keys())
        for name in left:
            field = self.model.get_field(name)[1]
            kwargs = {
                name: value for name, value in row.iteritems()
                if name in field.depends
            }
            value = field(**kwargs)
            row.update({name: value})
        return row

    @property
    def copy(self):
        new_data = list(row.copy() for row in self.rows)
        new_include = range(len(new_data))
        return Manager(
            model=self.model.__class__, data=new_data, include=new_include)

    def __iter__(self):
        return self.rows

    def save(self, csvfile, **kwargs):
        fieldnames = self.model.columns
        with open(csvfile, "w") as csvfile:
            writer = DictWriter(csvfile, fieldnames,  **kwargs)
            for row in self.rows:
                writer.writerow({
                    column: row[column] for column in fieldnames
                })

    def column(self, column):
        return [x[column] for x in self.rows]

    def update(self, **kwargs):
        for index in self._include:
            for name in kwargs:
                self._data[index][name] = kwargs[name]

    def insert(self, **kwargs):
        if len(self._data) == len(self._include):
            self._data.append(self.model.result_class(kwargs))
            self._include.append(len(self._include))
        else:
            raise ValueError("Cannot insert into filtered results")

    def __getitem__(self, key):
        return self.column(key)

    def filter(self, **kwargs):
        new_include = list()
        for index in self._include:
            if all(
                self.eval_row(index)[name] == value
                for name, value in kwargs.iteritems()
            ):
                new_include.append(index)
        return Manager(
            self.model.__class__, data=self._data, include=new_include)

    def order_by(self, *args):
        def cmp(a, b):
            a, b = self._data[a], self._data[b]
            for arg in args:
                if a[arg] > b[arg]:
                    return 1
                elif a[arg] < b[arg]:
                    return -1
                else:
                    continue
            return 0

        new_include = list(sorted(
            self._include, key=cmp_to_key(cmp)
        ),)

        return Manager(
            self.model.__class__, data=self._data, include=new_include)

    @property
    def reverse(self):
        new_include = list(self._include)
        new_include.reverse()

        return Manager(
            self.model.__class__, data=self._data, include=new_include)

    def group_by(self, column, function=list, key=None):
        if key:
            def func(rows):
                return function(row[key] for row in rows)
        else:
            func = function
        unique_values = set(row[column] for row in self)
        return {
            row: func(r for r in self if r[column] == row)
            for row in unique_values
        }
