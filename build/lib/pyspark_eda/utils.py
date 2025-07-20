def round_off(num, dec=2):
    factor = 10 ** dec
    return int(num * factor) / factor

