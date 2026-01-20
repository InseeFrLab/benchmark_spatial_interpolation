import types
import sys

def apply_treeple_patch():
    """
    Total override patch. This replaces the problematic method inside 
    the treeple library's forest class to bypass the immutable tags error.
    """
    import sklearn.utils.validation
    from types import SimpleNamespace

    # 1. Define a completely flexible tags object
    def universal_tags_override(self):
        return SimpleNamespace(
            estimator_type="regressor",
            requires_y=True,
            requires_fit=True,
            target_tags=SimpleNamespace(single_output=True, multi_output=False),
            regressor_tags=SimpleNamespace(multi_label=False), # treeple can write to this now
            transformer_tags=SimpleNamespace(preserves_dtype=["float64"]),
            non_deterministic=False,
            no_validation=False,
            poor_score=False,
            multilabel=False
        )

    # 2. Force the override on the treeple classes
    # We target the specific file shown in your traceback: treeple/_lib/sklearn/ensemble/_forest.py
    try:
        import treeple._lib.sklearn.ensemble._forest as treeple_forest
        # Override the method on the base class used by ObliqueRandomForest
        treeple_forest.ForestRegressor.__sklearn_tags__ = universal_tags_override
        print("Successfully patched treeple.ForestRegressor class.")
    except Exception as e:
        print(f"Note: Could not patch treeple ForestRegressor directly: {e}")

    try:
        import treeple.ensemble
        treeple.ensemble.ObliqueRandomForestRegressor.__sklearn_tags__ = universal_tags_override
        print("Successfully patched ObliqueRandomForestRegressor class.")
    except Exception as e:
        print(f"Note: Could not patch ObliqueRandomForestRegressor directly: {e}")

    # 3. Handle the Weight Checker (Standard for 1.6+)
    original_weight_func = sklearn.utils.validation._check_sample_weight
    def patched_check_sample_weight(sample_weight, X, dtype=None, **kwargs):
        return original_weight_func(sample_weight, X, dtype=dtype, **kwargs)
    sklearn.utils.validation._check_sample_weight = patched_check_sample_weight

    print("Treeple-Sklearn compatibility: Total Override Patch Active.")