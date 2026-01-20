import sklearn.utils.validation
import treeple.ensemble
from types import SimpleNamespace


def apply_treeple_patch():
    """
    Update to make it run with new versions of scikit-learn
    """

    # --- FIX 1: The Weight Checker (Standard & Intel) ---
    original_func = sklearn.utils.validation._check_sample_weight
    def patched_check_sample_weight(sample_weight, X, dtype=None, **kwargs):
        return original_func(sample_weight, X, dtype=dtype, **kwargs)
    
    sklearn.utils.validation._check_sample_weight = patched_check_sample_weight

    # --- FIX 2: Treeple internal tree classes ---
    try:
        import treeple._lib.sklearn.tree._classes as treeple_tree_classes
        treeple_tree_classes._check_sample_weight = patched_check_sample_weight
    except (ImportError, AttributeError):
        pass

    # --- FIX 3: Tag Fallback with Attribute Access (SimpleNamespace) ---
    target_class = treeple.ensemble.ObliqueRandomForestRegressor
    
    def patched_sklearn_tags(self):
        # We use SimpleNamespace so that sklearn can use dot notation (tags.requires_fit)
        # instead of failing on a dictionary.
        return SimpleNamespace(
            estimator_type="regressor",
            requires_fit=True,        # This is the attribute that was missing!
            multioutput=True,
            multilabel=False,
            poor_score=False,
            no_validation=False,
            requires_y=True,
            regressor_tags=SimpleNamespace(multi_label=False) # Nested tags
        )

    target_class.__sklearn_tags__ = patched_sklearn_tags
    
    print("Treeple-Sklearn compatibility patches active.")
    