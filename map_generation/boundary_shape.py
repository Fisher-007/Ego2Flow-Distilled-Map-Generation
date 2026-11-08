"""Spline road shapes with fixed source endpoints and tangents.

Interior residuals define the curve; boundary constraints are enforced by
the geometric representation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline


@dataclass
class BoundaryShape:
    endpoints: np.ndarray
    tangents: np.ndarray
    knots: np.ndarray
    interior_residual: np.ndarray
    endpoint_tangent_scales: np.ndarray | None = None

    def __post_init__(self):
        self.endpoints = np.asarray(self.endpoints, dtype=float)
        self.tangents = np.asarray(self.tangents, dtype=float)
        self.knots = np.asarray(self.knots, dtype=float)
        self.interior_residual = np.asarray(self.interior_residual, dtype=float).reshape(-1,self.endpoints.shape[1])
        self.tangents = self.tangents / np.linalg.norm(self.tangents, axis=1, keepdims=True)
        self.endpoint_tangent_scales = np.asarray([1.,1.] if self.endpoint_tangent_scales is None else self.endpoint_tangent_scales)
        length = np.linalg.norm(self.endpoints[1]-self.endpoints[0])
        self.base = CubicHermiteSpline([0., 1.], self.endpoints, length*self.tangents*self.endpoint_tangent_scales[:,None])
        zero = np.zeros_like(self.endpoints[0])
        values = np.vstack([zero, self.interior_residual, zero])
        self.residual = CubicSpline(self.knots, values, bc_type=((1, zero), (1, zero)))

    def __call__(self, parameter, derivative=0):
        return self.base(parameter, nu=derivative)+self.residual(parameter, nu=derivative)

    def as_dict(self):
        return {'endpoints_xy_m': self.endpoints.tolist(), 'tangents_xy_unit': self.tangents.tolist(),
                'parameter_knots': self.knots.tolist(), 'interior_residual_xy_m': self.interior_residual.tolist(),
                'endpoint_tangent_scales': self.endpoint_tangent_scales.tolist(),
                'representation': 'Cubic Hermite boundary curve + clamped cubic residual; residual value and derivative zero at both ends.',
                'boundary_enforcement_source': 'Deterministic representation, not generated or learned.',
                'scope': 'XY shape only; source topology, permissions, width and elevation remain external constraints.'}

    def polynomial_segments(self):
        """Global XY coefficients, ascending powers of each segment's [0,1]."""
        combined = CubicHermiteSpline(self.knots, self(self.knots), self(self.knots, 1))
        return [combined.c[:, i][::-1]*duration**np.arange(4)[:, None]
                for i, duration in enumerate(np.diff(self.knots))]

    @classmethod
    def from_dict(cls, data):
        return cls(data['endpoints_xy_m'], data['tangents_xy_unit'], data['parameter_knots'],
                   data['interior_residual_xy_m'],data.get('endpoint_tangent_scales'))

    @classmethod
    def from_polyline(cls, points, endpoints, tangents):
        points = np.asarray(points, dtype=float)
        # Source polylines can repeat a junction vertex; it has no arc length.
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]]
        stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        knots = stations/stations[-1]
        shape = cls(endpoints, tangents, knots, np.zeros_like(points[1:-1]))
        return cls(endpoints, tangents, knots, points[1:-1]-shape.base(knots[1:-1]))
