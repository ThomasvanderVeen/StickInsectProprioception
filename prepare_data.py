from general import *

'''
This file extracts pitch, gait and joint angles data from the sim_data folder. They are all trimmed to an equal total time (nostep: 5 sec, step: 20 sec) and interpolated
'''

def pickle_open(filename):
    """
    Load data from a pickle file.

    Parameters:
    - filename: The path to the pickle file to be opened.

    Returns:
    - data: The data loaded from the pickle file.
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    return data

def interpolate(old_array, t_total, n_steps, boolean=False):
    """
    Interpolate an array over a specified number of steps and apply optional smoothing or thresholding.

    Parameters:
    - old_array: The input array to be interpolated.
    - t_total: The total duration corresponding to the length of the original array.
    - n_steps: The number of steps for the interpolated array.
    - boolean: If False (default), apply Gaussian smoothing to the interpolated array.
               If True, threshold the interpolated values to binary (0 or 1) based on a threshold of 0.5.

    Returns:
    - new_array: The interpolated array, either 1D or 2D depending on the input.
    """
    if old_array.ndim == 1:
        old_array = old_array.reshape((old_array.size, 1))

    new_array = np.zeros((n_steps, old_array.shape[1]))

    x_old = np.linspace(0, t_total, num=old_array.shape[0])
    x_new = np.linspace(0, t_total, num=n_steps)

    for i in range(old_array.shape[1]):
        new_array[:, i] = np.interp(x_new, x_old, old_array[:, i])
        if not boolean:
            new_array[:, i] = gaussian_filter1d(new_array[:, i], sigma=20)
        if boolean:
            new_array[:, i][new_array[:, i] > 0.5] = 1
            new_array[:, i][new_array[:, i] <= 0.5] = 0

    if new_array.shape[1] == 1:
        return np.ndarray.flatten(new_array)
    else:
        return new_array


# Constants for camera timing and frame calculations
dt_camera = 1 / 200
n_frames = int(constants['T_TOTAL'] / dt_camera)
n_frames_step = int(constants['T_TOTAL_STEP'] / dt_camera)

# Load simulation data
data_nostep = pickle_open('sim_data/simulation_data_nostep')
data_step = pickle_open('sim_data/simulation_data_step')

# Initialize arrays to store joint angles, gait data, and pitch data
joint_angles_nostep = np.zeros((constants['N_STEPS'], 18, 78))
joint_angles_step = np.zeros((constants['N_STEPS_STEP'], 18, 21))
gait_nostep = np.zeros((constants['N_STEPS'], 6, 78))
pitch_step = np.zeros((constants['N_STEPS_STEP'], 21))

# Initialize lists to store animal data
animals_nostep = []
animals_step = []

# Process the 'no step' simulation data
for i in range(78):
    # Extract and transpose joint angles and gait data
    joint_angles = np.array(data_nostep[f'simulation_{i}'][0]).T
    gait = np.array(data_nostep[f'simulation_{i}'][1]).T

    # Trim data to match the number of frames
    joint_angles = joint_angles[:n_frames]
    gait = gait[:n_frames]

    # Interpolate and store the data
    joint_angles_nostep[:, :, i] = interpolate(joint_angles, constants['T_TOTAL'], constants['N_STEPS'])
    gait_nostep[:, :, i] = interpolate(gait, constants['T_TOTAL'], constants['N_STEPS'], boolean=True)

    # Store animal data
    animals_nostep.append(data_nostep[f'simulation_{i}'][3])

# Process the 'step' simulation data
for i in range(21):
    # Extract and transpose joint angles and pitch data
    joint_angles = np.array(data_step[f'simulation_{i}'][0]).T
    pitch = np.array(data_step[f'simulation_{i}'][2])[1]

    # Trim data to match the number of frames
    joint_angles = joint_angles[:n_frames_step]
    pitch = pitch[:n_frames_step]

    # Interpolate and store the data
    joint_angles_step[:, :, i] = interpolate(joint_angles, constants['T_TOTAL_STEP'], constants['N_STEPS_STEP'])
    pitch_step[:, i] = interpolate(pitch, constants['T_TOTAL_STEP'], constants['N_STEPS_STEP'])

    # Store animal data
    animals_step.append(data_step[f'simulation_{i}'][3])

# Ensure the output folder exists
create_folder_if_not_exists('temp_data')

# Save the processed data to files
np.save('temp_data/joint_angles', joint_angles_nostep)
np.save('temp_data/gait', gait_nostep)
np.save('temp_data/joint_angles_step', joint_angles_step)
np.save('temp_data/pitch_step', pitch_step)
np.save('temp_data/animals_step', animals_step)
np.save('temp_data/animals_nostep', animals_nostep)
