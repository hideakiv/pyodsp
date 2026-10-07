import json
import os
import warnings

# Default values
BM_ABS_TOLERANCE = 1e-6
BM_REL_TOLERANCE = 1e-6
BM_TIME_LIMIT = 3600
BM_SLACK_TOLERANCE = 1e-9
BM_MAX_CUT_AGE = 10
BM_CUT_SIM_TOLERANCE = 1e-12
BM_PURGE_FREQ = 1
BM_DUMMY_BOUND = 1e9
PBM_ML = 0.1
PBM_MR = 0.5
PBM_U_MIN = 1e-10
PBM_E_S = 1e-6
BM_LAMBDA_BOUND = 1e6
DEC_CUT_ABS_TOL = 1e-9
# SDDP stopping (see Lattice._termination and Lattice._bound_stalled).
# gap: stop when the simulated cost's upper confidence limit is within this
#   fraction of the bound.
# stable: stop when re-simulating the same paths changes their mean cost by
#   less than this fraction, in either direction.
# stall: stop when the bound has moved by less than SDDP_STALL_TOLERANCE
#   (relative) over the last SDDP_STALL_ITERATIONS iterations; 0 turns it off.
SDDP_GAP_TOLERANCE = 1e-2
SDDP_STABLE_TOLERANCE = 1e-3
SDDP_STALL_ITERATIONS = 20
SDDP_STALL_TOLERANCE = 1e-4
SDDP_SEED = 42

# Keys that used to mean something else. A file still carrying them gets a
# warning rather than silently running under the new rules.
_RENAMED = {
    "SDDP_REL_TOLERANCE": "SDDP_GAP_TOLERANCE (now one-sided: upper confidence "
    "limit minus bound, relative to the bound)",
    "SDDP_IMPROVE_TOLERANCE": "SDDP_STABLE_TOLERANCE (now relative to the "
    "mean cost, and two-sided)",
}


# Function to load parameters from a JSON file
def load_params_from_file(file_path):
    global \
        BM_ABS_TOLERANCE, \
        BM_REL_TOLERANCE, \
        BM_TIME_LIMIT, \
        BM_SLACK_TOLERANCE, \
        BM_MAX_CUT_AGE, \
        BM_CUT_SIM_TOLERANCE, \
        BM_PURGE_FREQ, \
        BM_DUMMY_BOUND, \
        PBM_ML, \
        PBM_MR, \
        PBM_U_MIN, \
        PBM_E_S, \
        BM_LAMBDA_BOUND, \
        DEC_CUT_ABS_TOL, \
        SDDP_GAP_TOLERANCE, \
        SDDP_STABLE_TOLERANCE, \
        SDDP_STALL_ITERATIONS, \
        SDDP_STALL_TOLERANCE, \
        SDDP_SEED
    try:
        with open(file_path, "r") as f:
            params = json.load(f)
            BM_ABS_TOLERANCE = params.get("BM_ABS_TOLERANCE", BM_ABS_TOLERANCE)
            BM_REL_TOLERANCE = params.get("BM_REL_TOLERANCE", BM_REL_TOLERANCE)
            BM_TIME_LIMIT = params.get("BM_TIME_LIMIT", BM_TIME_LIMIT)
            BM_SLACK_TOLERANCE = params.get("BM_SLACK_TOLERANCE", BM_SLACK_TOLERANCE)
            BM_MAX_CUT_AGE = params.get("BM_MAX_CUT_AGE", BM_MAX_CUT_AGE)
            BM_CUT_SIM_TOLERANCE = params.get(
                "BM_CUT_SIM_TOLERANCE", BM_CUT_SIM_TOLERANCE
            )
            BM_PURGE_FREQ = params.get("BM_PURGE_FREQ", BM_PURGE_FREQ)
            BM_DUMMY_BOUND = params.get("BM_DUMMY_BOUND", BM_DUMMY_BOUND)
            PBM_ML = params.get("PBM_ML", PBM_ML)
            PBM_MR = params.get("PBM_MR", PBM_MR)
            PBM_U_MIN = params.get("PBM_U_MIN", PBM_U_MIN)
            PBM_E_S = params.get("PBM_E_S", PBM_E_S)
            BM_LAMBDA_BOUND = params.get("BM_LAMBDA_BOUND", BM_LAMBDA_BOUND)
            DEC_CUT_ABS_TOL = params.get("DEC_CUT_ABS_TOL", DEC_CUT_ABS_TOL)
            SDDP_GAP_TOLERANCE = params.get("SDDP_GAP_TOLERANCE", SDDP_GAP_TOLERANCE)
            SDDP_STABLE_TOLERANCE = params.get(
                "SDDP_STABLE_TOLERANCE", SDDP_STABLE_TOLERANCE
            )
            SDDP_STALL_ITERATIONS = params.get(
                "SDDP_STALL_ITERATIONS", SDDP_STALL_ITERATIONS
            )
            SDDP_STALL_TOLERANCE = params.get(
                "SDDP_STALL_TOLERANCE", SDDP_STALL_TOLERANCE
            )
            for old, new in _RENAMED.items():
                if old in params:
                    warnings.warn(
                        f"{file_path}: {old} is no longer read; see {new}.",
                        stacklevel=2,
                    )
            SDDP_SEED = params.get("SDDP_SEED", SDDP_SEED)
    except FileNotFoundError:
        print(f"Parameter file {file_path} not found. Using default values.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}. Using default values.")


# Load parameters from the specified JSON file if provided
param_file_path = os.getenv("PYODSP_PARAM_PATH")
if param_file_path:
    load_params_from_file(param_file_path)
