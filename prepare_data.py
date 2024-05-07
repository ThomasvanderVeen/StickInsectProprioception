from general import *

def pickle_open(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    return data

def interpolate(old_array, t_total, n_steps, boolean=False):
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
            new_array[:, i][new_array[:, i] <= 0.50] = 0

    if new_array.shape[1] == 1:
        return np.ndarray.flatten(new_array)
    else:
        return new_array

dt_camera = 1/200
n_frames = int(constants['T_TOTAL'] / dt_camera)

data_nostep = pickle_open('sim_data/simulation_data_nostep')
data_step = pickle_open('sim_data/simulation_data_step')

joint_angles_nostep = np.zeros((constants['N_STEPS'], 18, 78))
joint_angles_step = np.zeros((constants['N_STEPS'], 18, 11))

gait_nostep = np.zeros((constants['N_STEPS'], 6, 78))

pitch_step = np.zeros((constants['N_STEPS'], 11))

for i in range(78):
    joint_angles = np.array(data_nostep[f'simulation_{i}'][0]).T
    gait = np.array(data_nostep[f'simulation_{i}'][1]).T

    joint_angles = joint_angles[:n_frames]
    gait = gait[:n_frames]

    joint_angles_nostep[:, :, i] = interpolate(joint_angles, constants['T_TOTAL'], constants['N_STEPS'])
    gait_nostep[:, :, i] = interpolate(gait, constants['T_TOTAL'], constants['N_STEPS'], True)


for i in range(11):
    joint_angles = np.array(data_step[f'simulation_{i}'][0]).T
    pitch = np.array(data_step[f'simulation_{i}'][2])[1]

    joint_angles = joint_angles[:n_frames]
    pich = pitch[:n_frames]

    joint_angles_step[:, :, i] = interpolate(joint_angles, constants['T_TOTAL'], constants['N_STEPS'])
    pitch_step[:, i] = interpolate(pitch, constants['T_TOTAL'], constants['N_STEPS'])

create_folder_if_not_exists('temp_data')

with open('temp_data/joint_angles_nostep', 'wb') as file:
    np.save(file, joint_angles_nostep)

with open('temp_data/gait_nostep', 'wb') as file:
    np.save(file, gait_nostep)

with open('temp_data/joint_angles_step', 'wb') as file:
    np.save(file, joint_angles_step)

with open('temp_data/pitch_step', 'wb') as file:
    np.save(file, pitch_step)