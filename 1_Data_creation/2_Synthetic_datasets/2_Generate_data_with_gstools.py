# %%
# Example 1: Creating a dataframe from a regular grid
import numpy as np
import polars as pl
import gstools as gs
import matplotlib.pyplot as plt

# Create regular grid coordinates
x = np.arange(0, 100, 1)
y = np.arange(0, 100, 1)

# Define covariance model
model = gs.Matern(
    latlon=True,
    temporal=False,
    var=1,
    len_scale=[1000, 100],
    geo_scale=gs.KM_SCALE,
)

# Initialize spatial random field generator
srf = gs.SRF(model, seed=20170519)

# Generate random field on a structured grid (mesh)
field = srf.structured([x, y])
srf.plot()
# %%
# Create DataFrame with coordinates and values
xx, yy = np.meshgrid(x, y, indexing='ij')  # coordinate meshgrid to match the field array shape
df = pl.DataFrame({
    'x': xx.flatten(),
    'y': yy.flatten(),
    'value': field.flatten()
})

# %%
# Plot the data
plt.figure(figsize=(8, 6))
sc = plt.scatter(df['y'], df['x'], c=df['value'], cmap='viridis', marker='o')
plt.colorbar(sc, label='Value')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Random field values on grid')
plt.show()


# %%
# Example 2: Creating a dataframe from a unstructured grid, where points are drawn from a uniform distribution

# Create regular grid coordinates
seed = gs.random.MasterRNG(19970221)
rng = np.random.RandomState(seed())
x = rng.uniform(0, 100, size=10000)
y = rng.uniform(0, 100, size=10000)

# Define covariance model
model = gs.Matern(
    latlon=True,
    temporal=False,
    var=1,
    len_scale=[1000, 100],
    geo_scale=gs.KM_SCALE,
)

# Initialize spatial random field generator
srf = gs.SRF(model, seed=20170519)

# Generate random field on an unstructured grid (mesh)
field = srf((x, y))
srf.plot()
# %%
# Create DataFrame with coordinates and values
df = pl.DataFrame({
    'x': x,
    'y': y,
    'value': field.flatten()
})
# %%
# Plot the data
plt.figure(figsize=(8, 6))
sc = plt.scatter(df['y'], df['x'], c=df['value'], cmap='viridis', marker='o')
plt.colorbar(sc, label='Value')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Random field values on grid')
plt.show()

