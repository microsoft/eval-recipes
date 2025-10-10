# Copyright (c) Microsoft. All rights reserved.

from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def parse_filter(filter_str: str) -> tuple[str, list[str]]:
    """
    Parse a filter string like 'field=value1,value2' into field path and values.

    Args:
        filter_str: Filter string in format 'field=value' or 'field=value1,value2'

    Returns:
        Tuple of (field_path, list_of_values)

    Examples:
        >>> parse_filter('name=claude_code')
        ('name', ['claude_code'])
        >>> parse_filter('task_info.difficulty=easy,medium')
        ('task_info.difficulty', ['easy', 'medium'])
    """
    if "=" not in filter_str:
        raise ValueError(f"Invalid filter format: '{filter_str}'. Expected 'field=value' or 'field=value1,value2'")
    field_path, values_str = filter_str.split("=", 1)
    values = [v.strip() for v in values_str.split(",")]
    return field_path.strip(), values


def get_nested_field(obj: BaseModel, field_path: str) -> Any:
    """
    Get a nested field value from a Pydantic model using dot notation.

    Args:
        obj: Pydantic model instance
        field_path: Field path using dot notation (e.g., 'name' or 'task_info.difficulty')

    Returns:
        The value of the field

    Raises:
        AttributeError: If the field path is invalid

    Examples:
        >>> get_nested_field(task_config, 'name')
        'blog_writing'
        >>> get_nested_field(task_config, 'task_info.difficulty')
        'medium'
    """
    parts = field_path.split(".")
    value: Any = obj

    for part in parts:
        if hasattr(value, part):
            value = getattr(value, part)
        else:
            raise AttributeError(f"Field '{part}' not found in path '{field_path}'")

    return value


def apply_filters(items: list[T], filters: list[str]) -> list[T]:
    """
    Apply multiple filters to a list of Pydantic models.

    Multiple filters use AND logic (all must match).
    Comma-separated values within a filter use OR logic (any can match).

    Args:
        items: List of Pydantic model instances
        filters: List of filter strings like ['name=value', 'field=value1,value2']

    Returns:
        Filtered list where each item matches ALL filters

    Examples:
        >>> # Single filter
        >>> apply_filters(tasks, ['difficulty=medium'])
        >>> # Multiple values (OR logic)
        >>> apply_filters(tasks, ['name=task1,task2'])
        >>> # Multiple filters (AND logic)
        >>> apply_filters(tasks, ['difficulty=medium', 'task_info.non_deterministic_evals=true'])
    """
    if not filters:
        return items

    filtered_items = items.copy()
    for filter_str in filters:
        field_path, allowed_values = parse_filter(filter_str)
        # Filter items - keep only those where the field value matches any allowed value
        filtered_items = [item for item in filtered_items if _matches_filter(item, field_path, allowed_values)]

    return filtered_items


def _matches_filter(item: BaseModel, field_path: str, allowed_values: list[str]) -> bool:
    """
    Check if an item matches a filter.

    Args:
        item: Pydantic model instance
        field_path: Field path to check
        allowed_values: List of allowed values (OR logic)

    Returns:
        True if the item's field value matches any of the allowed values
    """
    try:
        field_value = get_nested_field(item, field_path)

        # Convert field value to string for comparison
        field_value_str = str(field_value).lower()

        # Check if field value matches any of the allowed values
        for allowed_value in allowed_values:
            allowed_value_lower = allowed_value.lower()

            # Handle boolean values
            if isinstance(field_value, bool):
                if allowed_value_lower in ("true", "1", "yes") and field_value:
                    return True
                if allowed_value_lower in ("false", "0", "no") and not field_value:
                    return True
            # String comparison
            elif field_value_str == allowed_value_lower:
                return True

        return False

    except AttributeError:
        # Field doesn't exist on this item, so it doesn't match
        return False
