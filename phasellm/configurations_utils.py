import re

from warnings import warn


def coerce_azure_base_url(url: str) -> str:
    """
    This function coerces the base URL to the proper format for the Azure OpenAI API. This is used for backwards
    compatibility of base_url and api_base arguments.
    Args:
        url: The url to coerce.

    Returns:
        The coerced URL.

    """
    pass
