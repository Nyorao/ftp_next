"""Label-free sagittal layout from verified LPS image geometry.

This utility is for a new input manifest. It does not silently change legacy
PNG orientation, the ongoing frozen experiments, or their reconstruction rule.
"""
from itertools import permutations

import numpy as np


def transformed_geometry(size_xyz, spacing_xyz, origin_xyz, direction, transpose_zyx, flip_axes_zyx=()):
    """Geometry for array.transpose(transpose_zyx), then np.flip(flip_axes_zyx).

    Preserves every voxel's physical coordinate; no interpolation or masks.
    """
    if sorted(transpose_zyx) != [0, 1, 2] or len(set(flip_axes_zyx)) != len(flip_axes_zyx) or any(x not in (0,1,2) for x in flip_axes_zyx):
        raise ValueError('Invalid 3D permutation/flip')
    size = np.asarray(size_xyz, dtype=np.int64)
    spacing = np.asarray(spacing_xyz, dtype=float)
    origin = np.asarray(origin_xyz, dtype=float)
    matrix = np.asarray(direction, dtype=float).reshape(3,3)
    if size.shape != (3,) or spacing.shape != (3,) or origin.shape != (3,):
        raise ValueError('Expected three-dimensional geometry')
    if (size <= 0).any() or (spacing <= 0).any() or not all(np.isfinite(x).all() for x in (spacing, origin, matrix)):
        raise ValueError('Invalid size or physical geometry')
    order = [2-axis for axis in reversed(transpose_zyx)]
    size, spacing, matrix = size[order], spacing[order], matrix[:, order]
    origin = origin.copy()
    for axis_zyx in flip_axes_zyx:
        axis = 2-axis_zyx
        origin += matrix[:,axis]*spacing[axis]*(size[axis]-1)
        matrix[:,axis] *= -1
    return {'size_xyz':size.tolist(), 'spacing_xyz_mm':spacing.tolist(),
            'origin_xyz':origin.tolist(), 'direction':matrix.reshape(-1).tolist()}


def sagittal_plan(geometry, min_axis_cosine=0.8):
    """Plan canonical (leftward slices, posterior columns, caudal rows).

    Output is (slice, AP, SI), directly matching per-slice model tensor (AP,SI).
    Increasing final-axis index moves caudally. Direction is interpreted in the
    LPS physical frame used by SimpleITK, and must come from a verified source.
    Dominant-axis permutation preserves native obliquity rather than resampling.
    Strong obliquity/invalid axes are rejected, never guessed from array sizes.
    """
    if not 0 < min_axis_cosine <= 1:
        raise ValueError('min_axis_cosine must be in (0,1]')
    size = np.asarray(geometry['size_xyz'])
    spacing = np.asarray(geometry['spacing_xyz_mm'], dtype=float)
    matrix = np.asarray(geometry['direction'], dtype=float).reshape(3,3)
    if size.shape != (3,) or (size <= 0).any() or spacing.shape != (3,) or (spacing <= 0).any():
        raise ValueError('Invalid volume geometry')
    if not np.isfinite(matrix).all() or not np.isfinite(spacing).all() or not np.allclose(matrix.T@matrix, np.eye(3), atol=1e-4):
        raise ValueError('Direction must be finite and orthonormal')
    axes = max(permutations(range(3)), key=lambda p: sum(abs(matrix[world,p[world]]) for world in range(3)))
    cosines = [float(matrix[world,axes[world]]) for world in range(3)]
    if min(map(abs,cosines)) < min_axis_cosine:
        raise ValueError('Oblique axes need explicit resampling; dominant-axis layout is insufficient')
    # Physical LPS: +X left, +Y posterior, +Z superior. Final axis is caudal.
    flips = [cosines[0] < 0, cosines[1] < 0, cosines[2] > 0]
    return {'input_shape_zyx':size[::-1].tolist(), 'transpose_zyx_to_slice_ap_si':[2-axis for axis in axes],
            'flip_canonical_axes':[i for i,flip in enumerate(flips) if flip],
            'canonical_shape':[int(size[axis]) for axis in axes],
            'canonical_spacing_mm':[float(spacing[axis]) for axis in axes],
            'axis_direction_cosines':cosines, 'physical_frame':'LPS',
            'final_axis_increases':'caudal', 'uses_labels':False, 'resampled':False}


def canonicalize(array_zyx, plan):
    if tuple(array_zyx.shape) != tuple(plan['input_shape_zyx']):
        raise ValueError('Array shape does not match verified geometry')
    value = array_zyx.transpose(plan['transpose_zyx_to_slice_ap_si'])
    if plan['flip_canonical_axes']:
        value = np.flip(value, tuple(plan['flip_canonical_axes']))
    return value.copy()


def restore_native(array_slice_ap_si, plan):
    if tuple(array_slice_ap_si.shape) != tuple(plan['canonical_shape']):
        raise ValueError('Prediction shape does not match canonical geometry')
    value = array_slice_ap_si
    if plan['flip_canonical_axes']:
        value = np.flip(value, tuple(plan['flip_canonical_axes']))
    return value.transpose(np.argsort(plan['transpose_zyx_to_slice_ap_si'])).copy()
