"""Leakage-safe feature construction.

Every feature row is indexed by a forecast origin ``T`` (the end of the latest completed 5-minute
interval, UTC). A feature may only use information available at ``T``:

* prices and demand for intervals ending at or before ``T``;
* weather whose ``valid_time + availability_lag`` is at or before ``T``;
* deterministic facts about the target time (calendar, public holidays, sun position).
"""
