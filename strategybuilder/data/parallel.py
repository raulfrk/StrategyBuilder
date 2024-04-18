import copy
import multiprocessing

import dill
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import _BaseComposition

from strategybuilder.data.fetcher import BarDataFetch, NewsDataFetch


class Union(BaseEstimator, TransformerMixin):
    def __init__(self, transformers):

        self.transformers = transformers

    def fit(self, X, y=None):
        for transformer in self.transformers:
            transformer.fit(X, y)
        return self

    def transform(self, X):
        l = []
        transformers = self.transformers
        if len(self.transformers[0]) == 2:
            transformers = [transformer for _, transformer in self.transformers]
        for transformer in transformers:
            l.append(transformer.transform(X))
        return l


def runner(op):
    transformer, X = op
    transformer = dill.loads(transformer)
    return transformer.transform(X)


class ParallelUnion(TransformerMixin, _BaseComposition):

    def __init__(self, transformers, max_workers=5):
        self.transformers = transformers
        self.max_workers = max_workers

    def fit(self, X, y=None):
        for transformer in self.transformers:
            transformer.fit(X, y)
        return self

    def transform(self, X):
        transformers = self.transformers
        if len(self.transformers[0]) == 2:
            transformers = [transformer for _, transformer in self.transformers]
        with multiprocessing.Pool(processes=self.max_workers) as pool:
            result = pool.map(runner, [(dill.dumps(transformer), X) for transformer in transformers])
        return tuple(result)


class MultiParamOverwritePipeline(ParallelUnion):
    def __init__(self, transformer: Pipeline, lkwargs: list[dict] | list[dict[str, dict]], single_layer_overwrite=True, max_workers=5):
        if single_layer_overwrite:
            transformers = self._handle_single_layer_overwrite(transformer, lkwargs)
        else:
            transformers = self._handle_multi_layer_overwrite(transformer, lkwargs)

        super().__init__(transformers, max_workers)

    def _handle_single_layer_overwrite(self, transformer, lkwargs):
        transformers = [copy.deepcopy(transformer) for _ in range(len(lkwargs))]
        for i, transformer in enumerate(transformers):
            layer = transformer[0] if "_layer" not in lkwargs[i].keys() else transformer[lkwargs[i]["_layer"]]

            for k, v in lkwargs[i].items():
                if k == "_layer":
                    continue
                if not hasattr(layer, k):
                    raise AttributeError(f"{layer} has no attribute {k}")
                setattr(layer, k, v)
        return transformers

    def _handle_multi_layer_overwrite(self, transformer, lkwargs):
        transformers = [copy.deepcopy(transformer) for _ in range(len(lkwargs))]
        for i, transformer in enumerate(transformers):
            for layer_name, overwrite in lkwargs[i].items():
                layer = transformer[layer_name]
                for attribute, ow_value in overwrite.items():
                    if not hasattr(layer, attribute):
                        raise AttributeError(f"{transformer} - {layer} has no attribute {attribute}")
                    setattr(layer, attribute, ow_value)
        return transformers
