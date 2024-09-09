import math


def eiler_to_decart(l: float, Az: float, An: float) -> list[float]:
    """
    Translate coordinates from Eiler connatation to Decart
    
    Args:
        l (float): length
        Az (float): azimuth
        An (float): elevation

    Returns:
        x (float)
        y (float)
        z (float)
    """
    x = l * math.cos(An) * math.cos(Az)
    y = l * math.cos(An) * math.sin(Az)
    z = l * math.sin(An)
    return x, y, z


def decart_to_eiler(x: float, y: float, z: float) -> list[float]:
    """
    Translate coordinates from Decart connatation to Eiler

    Args:
        x (float)
        y (float)
        z (float)

    Returns:
        l (float): length
        Az (float): azimuth
        An (float): elevation
    """
    l = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    Az = math.atan(y / x)
    An = math.pi / 2 - math.acos(z / math.sqrt(x ** 2 + y ** 2 + z ** 2))
    if x < 0:
        Az -= math.pi
    return l, Az, An
