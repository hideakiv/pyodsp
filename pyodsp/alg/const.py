import json
import os

# Default values
BM_ABS_TOLERANCE = 1e-6
BM_REL_TOLERANCE = 1e-6
BM_TIME_LIMIT = 3600
BM_SLACK_TOLERANCE = 1e-9
BM_MAX_CUT_AGE = 10
BM_CUT_SIM_TOLERANCE = 1e-12
BM_PURGE_FREQ = 1
DEC_CUT_ABS_TOL = 1e-9

# Function to load parameters from a JSON file
def load_params_from_file(file_path):
    global BM_ABS_TOLERANCE, BM_REL_TOLERANCE, BM_TIME_LIMIT, BM_SLACK_TOLERANCE, BM_MAX_CUT_AGE, BM_CUT_SIM_TOLERANCE, BM_PURGE_FREQ, DEC_CUT_ABS_TOL
    try:
        with open(file_path, 'r') as f:
            params = json.load(f)
            BM_ABS_TOLERANCE = params.get('BM_ABS_TOLERANCE', BM_ABS_TOLERANCE)
            BM_REL_TOLERANCE = params.get('BM_REL_TOLERANCE', BM_REL_TOLERANCE)
            BM_TIME_LIMIT = params.get('BM_TIME_LIMIT', BM_TIME_LIMIT)
            BM_SLACK_TOLERANCE = params.get('BM_SLACK_TOLERANCE', BM_SLACK_TOLERANCE)
            BM_MAX_CUT_AGE = params.get('BM_MAX_CUT_AGE', BM_MAX_CUT_AGE)
            BM_CUT_SIM_TOLERANCE = params.get('BM_CUT_SIM_TOLERANCE', BM_CUT_SIM_TOLERANCE)
            BM_PURGE_FREQ = params.get('BM_PURGE_FREQ', BM_PURGE_FREQ)
            DEC_CUT_ABS_TOL = params.get('DEC_CUT_ABS_TOL', DEC_CUT_ABS_TOL)
    except FileNotFoundError:
        print(f"Parameter file {file_path} not found. Using default values.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}. Using default values.")

# Load parameters from the specified JSON file if provided
param_file_path = os.getenv('PYODSP_PARAM_PATH')
if param_file_path:
    load_params_from_file(param_file_path)