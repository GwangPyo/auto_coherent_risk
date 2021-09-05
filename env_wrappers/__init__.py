from env_wrappers.wrapper import LunarLanderWrapper, BipedalWalkerWrapper, BipedalWalkerHardcoreWrapper


def wrapped_lunar_lander():
    return LunarLanderWrapper()


def wrapped_bipedal_walker():
    return BipedalWalkerWrapper()


def wrapped_bipdeal_walker_hardcore():
    return BipedalWalkerHardcoreWrapper()
