import math
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


def rotate_point(x, y, angle, center=None):
    """
    Rotate a 2D point counterclockwise by a given angle (in degrees) around a given center.

    Parameters:
    x (float or np.array): x-coordinate(s) to rotate.
    y (float or np.array): y-coordinate(s) to rotate.
    angle (float): Angle in degrees by which to rotate the point.
    center (tuple, optional): Center of rotation as (cx, cy). Defaults to the origin (0, 0).

    Returns:
    tuple: Rotated x and y coordinates.
    """
    cx, cy = center if center else (0, 0)
    dx = x - cx
    dy = y - cy
    ang_rad = math.radians(angle)

    xx = cx + dx * math.cos(ang_rad) - dy * math.sin(ang_rad)
    yy = cy + dx * math.sin(ang_rad) + dy * math.cos(ang_rad)

    return xx, yy


# A custom transformer to rotate geographical coordinates an arbitrary number of times
class AddCoordinatesRotation(BaseEstimator, TransformerMixin):
    """
    A custom transformer to rotate geographical coordinates an arbitrary number of times.

    Parameters:
    coordinates_names (tuple): Tuple of column names representing the x and y coordinates.
    number_axis (int): The number of rotations to apply to the coordinates.
    """
    def __init__(self, coordinates_names: tuple = None, number_axis: int = None):
        self.coordinates_names = coordinates_names
        self.number_axis = number_axis
        self.is_fitted = False

    def set_params(self, coordinates_names: tuple = None, number_axis: int = None):
        """
        Set parameters for the transformer.

        Parameters:
        coordinates_names (tuple): Tuple of column names representing the x and y coordinates.
        number_axis (int): The number of rotations to apply to the coordinates.

        Returns:
        self
        """
        self.coordinates_names = coordinates_names
        self.rotated_coordinates_names = []
        self.number_axis = number_axis
        return self

    def fit(self, X: pl.DataFrame, y=None):
        """
        Fit the transformer by checking for valid coordinate names and calculating the mean center.

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        self
        """
        assert isinstance(X, pl.DataFrame), "X must be a Polars DataFrame"
        coordinates_names = self.coordinates_names

        # Raise an error if the coordinates are not correct
        if coordinates_names is None:
            raise ValueError("Argument coordinates_names is missing.")
        if len(coordinates_names) != 2:
            raise ValueError("There must be exactly two coordinates.")
        if coordinates_names[0] not in X.columns:
            raise ValueError(f"Coordinate {coordinates_names[0]} is not in the data.")
        if coordinates_names[1] not in X.columns:
            raise ValueError(f"Coordinate {coordinates_names[1]} is not in the data.")

        # Raise an error if the number of axis is missing
        if self.number_axis is None:
            raise ValueError("Argument number_axis is missing.")

        x_coord, y_coord = self.coordinates_names
        # Compute the mean coordinates of the data
        self.center = (X[x_coord].mean(), X[y_coord].mean())

        self.is_fitted = True
        return self

    def transform(self, X: pl.DataFrame, y=None):
        """
        Rotate the coordinates and return the modified data.

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in transformation.

        Returns:
        pl.DataFrame: Transformed data with additional rotated coordinates.
        """
        assert isinstance(X, pl.DataFrame), "X must be a Polars DataFrame"
        x_coord, y_coord = self.coordinates_names
        rotated_coordinates_names = [x_coord, y_coord]

        for i in range(1, self.number_axis):
            # Compute coordinates after rotation
            x_temp, y_temp = rotate_point(
                x=X[x_coord],
                y=X[y_coord],
                angle=360 * (i / self.number_axis),
                center=self.center
            )

            # Add the rotated coordinates to the data
            X = X.with_columns(
                [
                    x_temp.alias(f"{x_coord}_rotated{i}"),
                    y_temp.alias(f"{y_coord}_rotated{i}")
                ]
            )
            rotated_coordinates_names = rotated_coordinates_names + [
                f"{x_coord}_rotated{i}", f"{y_coord}_rotated{i}"
            ]

        self.rotated_coordinates_names = rotated_coordinates_names
        self.names_features_output = X.columns
        return X

    def fit_transform(self, X, y=None):
        """
        Fit and transform the data in one step.

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        pl.DataFrame: Transformed data.
        """
        self.fit(X, y)
        return self.transform(X, y)

    def get_feature_names_out(self):
        """
        Get the names of the transformed features.

        Returns:
        list: Names of the transformed features.
        """
        return self.names_features_output


# A custom transformer to convert a polars DataFrame into Pandas
class ConvertToPandas(BaseEstimator, TransformerMixin):
    """
    Convert a Polars DataFrame to Pandas while:
    - converting string columns to categorical in Polars
    - storing category -> integer mappings at fit time
    - reapplying the same encoding at transform time
    """

    def __init__(self):
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: pl.DataFrame, y=None):
        """
        Does nothing.

        Parameters:
        X (pl.DataFrame): Input data.
        y (optional): Target values, not used in fitting.

        Returns:
        self
        """
        if not isinstance(X, pl.DataFrame):
            raise TypeError("Input must be a Polars DataFrame")

        self.feature_names = X.columns
        self.is_fitted = True
        return self

    def transform(self, X: pl.DataFrame, y=None):
        """
        Apply stored categorical encodings and convert to Pandas.
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer has not been fitted")

        if not isinstance(X, pl.DataFrame):
            raise TypeError("Input must be a Polars DataFrame")

        df = X.to_pandas()
        return df

    def fit_transform(self, X: pl.DataFrame, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def get_feature_names_out(self):
        """

        Returns:
        list: Names of the features.
        """
        return self.feature_names
