import importlib
from enum import Enum
import os
from typing import Optional
from ..modul import Modul


class ResourceType(Enum):
    CLASSIFICATION = 0
    METABENCHMARK = 1
    DATA = 2


class Resource:
    paths: dict[ResourceType, str] = {
        ResourceType.CLASSIFICATION: r'fopimt/resource/classification',
        ResourceType.DATA: r'fopimt/resource/data',
        ResourceType.METABENCHMARK: r'fopimt/resource/metabenchmark'
    }

    @staticmethod
    def get_resources(resource_type: ResourceType) -> dict:
        out = {
            'short_names': [],
            'long_names': [],
            'descriptions': []
        }
        path = Resource.paths[resource_type]
        files = Resource._get_files(path)
        for file in files:
            resource = Resource._load_resource(path, file)
            desc = Resource._get_description(resource)
            for f in desc['functions']:
                out['short_names'].append(desc['short_name'] + '.' + f)
                out['long_names'].append(desc['long_name'] + " " + f)
                out['descriptions'].append(desc['description'])

        return out

    @staticmethod
    def get_resource_function(short_name: str, type: ResourceType):
        path = Resource.paths[type]

        # short_name must be parsed
        # TODO not a safe way to handle it, exampe: resource.gnbg.sample  (0: package, 1: class, 2: function)
        resource_name = short_name.split('.')[1]
        function_name = short_name.split('.')[2]

        file = Resource._get_files(path, resource_name)[0]
        resource = Resource._load_resource(path, file)
        return getattr(resource, function_name)


    @staticmethod
    def _get_files(base: str, name: Optional[str] = None) -> list[str]:
        files = [os.path.splitext(f)[0] for f in os.listdir(base) if
                 os.path.isfile(os.path.join(base, f)) and not f.startswith('_')]
        if name is not None:
            files = [f for f in files if name in f]
        return files



    @staticmethod
    def _load_resource(_dir: str, file: str):
        _package = _dir.replace('/', '.')
        module = importlib.import_module(f'.{file}', _package)
        names = dir(module)
        names.remove(Modul.__name__)
        for n in names:
            modul_class = getattr(module, n)
            if hasattr(modul_class, 'get_short_name'):
                return modul_class


    @staticmethod
    def _get_description(resource):
        functions = [f for f in dir(resource) if not f.startswith(
                        ('_', 'get_short_name', 'get_long_name', 'get_description', 'get_tags', 'get_parameters')
                    )]
        return {
            'short_name': getattr(resource, 'get_short_name')(),
            'long_name': getattr(resource, 'get_long_name')(),
            'description': getattr(resource, 'get_description')(),
            'functions': functions
        }
