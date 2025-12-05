"""Pytest configuration and fixtures."""

import matplotlib
# Use non-interactive backend for CI
matplotlib.use("Agg")
