"""Settings for neighborhood subset analysis."""

ANALYSIS_INSTANCE_NAMES = ["D", "G", "F", "K"]
ANALYSIS_SEEDS = [11, 22, 33]
ANALYSIS_TIME_LIMIT_SECONDS = 60.0

OPERATOR_SETS = {
    "relocate_only": {
        "relocate_best": 1.0,
    },
    "remove_only": {
        "remove_reinsert_two": 1.0,
    },
    "eject_only": {
        "ejection_chain_light": 1.0,
    },
    "swap_only": {
        "swap_restricted": 1.0,
    },
    "relocate_remove": {
        "relocate_best": 1.0,
        "remove_reinsert_two": 1.0,
    },
    "relocate_eject": {
        "relocate_best": 1.0,
        "ejection_chain_light": 1.0,
    },
    "relocate_swap": {
        "relocate_best": 1.0,
        "swap_restricted": 1.0,
    },
    "remove_eject": {
        "remove_reinsert_two": 1.0,
        "ejection_chain_light": 1.0,
    },
    "remove_swap": {
        "remove_reinsert_two": 1.0,
        "swap_restricted": 1.0,
    },
    "eject_swap": {
        "ejection_chain_light": 1.0,
        "swap_restricted": 1.0,
    },
    "relocate_remove_eject": {
        "relocate_best": 1.0,
        "remove_reinsert_two": 1.0,
        "ejection_chain_light": 1.0,
    },
    "relocate_remove_swap": {
        "relocate_best": 1.0,
        "remove_reinsert_two": 1.0,
        "swap_restricted": 1.0,
    },
    "relocate_eject_swap": {
        "relocate_best": 1.0,
        "ejection_chain_light": 1.0,
        "swap_restricted": 1.0,
    },
    "remove_eject_swap": {
        "remove_reinsert_two": 1.0,
        "ejection_chain_light": 1.0,
        "swap_restricted": 1.0,
    },
    "all": {
        "relocate_best": 1.0,
        "remove_reinsert_two": 1.0,
        "ejection_chain_light": 1.0,
        "swap_restricted": 1.0,
    },
}