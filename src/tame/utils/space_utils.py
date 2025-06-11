from typing import List

from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Box
import numpy as np


def extend_dict_box_space(original_space: GymDict, add_size: int) -> GymDict:
    """Extends all Box spaces in a Dict space by a given size.

    This function creates a new Dict space where each Box subspace is extended along its first dimension
    by the specified size, while preserving the original low/high bounds and dtype.

    Args:
        original_space (GymDict): The original Dict space containing Box spaces to extend
        add_size (int): The number of additional dimensions to add to each Box space

    Returns:
        GymDict: A new Dict space with extended Box spaces

    Raises:
        TypeError: If the spaces in original_space are not Box spaces

    Example:
        >>> original = Dict({'obs': Box(low=0, high=1, shape=(3,))})
        >>> extended = extend_dict_box_space(original, 2)
        >>> extended.spaces['obs'].shape
        (5,)
    """
    return GymDict(
        {
            key: Box(
                low=space.low[0],  # type: ignore
                high=space.high[0],  # type: ignore
                shape=[space.shape[0] + add_size],  # type: ignore
                dtype=space.dtype,  # type: ignore
            )
            for key, space in original_space.spaces.items()
        }
    )


def extend_box_space(original_space: Box, add_size: int) -> GymDict:
    """Extends a Box space by adding dimensions to it.

    This function creates a new Box space with additional dimensions while maintaining
    the same low/high bounds and dtype from the original space.

    Args:
        original_space (Box): The original Box space to extend
        add_size (int): Number of additional dimensions to add to the space

    Returns:
        Box: A new Box space with extended dimensions, using the same bounds and dtype
            as the original space

    Example:
        >>> original = Box(low=0, high=1, shape=(3,))
        >>> extended = extend_box_space(original, 2)
        >>> extended.shape
        (5,)
    """
    return Box(
        low=original_space.low[0],  # type: ignore
        high=original_space.high[0],  # type: ignore
        shape=[original_space.shape[0] + add_size],  # type: ignore
        dtype=original_space.dtype,  # type: ignore
    )


def merge_spaces(spaces: List[Box]) -> Box:
    """
    Merge multiple gym Box spaces into a single Box space.

    This function takes multiple Box spaces from the gym library and combines them into
    a single Box space by concatenating their dimensions. The resulting space will have
    a shape equal to the sum of all individual space dimensions.

    Args:
        spaces (List[Box]): A list of gym.spaces.Box spaces to be merged.

    Returns:
        gym.spaces.Box: A merged Box space with shape equal to the sum of all input
            space dimensions, unbounded (-inf to inf), and same dtype as input spaces.

    Raises:
        ValueError: If any of the input spaces is not an instance of gym.spaces.Box.

    Example:
        >>> space1 = Box(low=-1, high=1, shape=(2,))
        >>> space2 = Box(low=0, high=2, shape=(3,))
        >>> merged = merge_spaces([space1, space2])
        >>> print(merged.shape)
        (5,)
    """
    if not all(isinstance(space, Box) for space in spaces):
        raise ValueError("All spaces must be gym.spaces.Box spaces")

    # Calculate total size by summing up all dimensions
    total_size = sum(np.prod(space.shape) for space in spaces)

    # Create merged space
    merged_space = Box(
        low=-np.inf,
        high=np.inf,
        shape=(int(total_size),),
        dtype=spaces[0].dtype,  # type: ignore
    )  # type: ignore

    return merged_space
