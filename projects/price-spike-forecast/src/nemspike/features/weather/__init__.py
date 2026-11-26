"""Pluggable weather feature builders.

To add a new family of weather features (for example wind-direction regimes, wind flow or
topography-based features):

1. Create a module in this package with a subclass of :class:`WeatherFeatureBuilder`.
2. Decorate it with ``@register`` and give it a unique ``name``.
3. Import the module below so it registers, and list its name under
   ``features.weather_builders`` in the config.

Builders receive weather that has already been tagged with ``available_at`` and must only join it
to origins with :func:`asof_join`, which guarantees no value is used before it was available.
"""

from nemspike.features.weather import basic as _basic  # noqa: F401  (registers "basic")
from nemspike.features.weather.base import (
    REGISTRY,
    WeatherFeatureBuilder,
    asof_join,
    build_weather_features,
    register,
    with_availability,
)

__all__ = [
    "REGISTRY",
    "WeatherFeatureBuilder",
    "asof_join",
    "build_weather_features",
    "register",
    "with_availability",
]
