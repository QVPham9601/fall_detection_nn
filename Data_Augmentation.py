import random
from cmath import pi


#rotation of pi rad wrt a given axis
def rotate(motion, axis = 'x'):
    
    if axis != 'x' and axis != 'y' and axis != 'z':
        axis = 'x'

    rotated = motion.copy()

    rotated["acc_" + axis] *= -1
    rotated["gyro_" + axis] *= -1

    rotated["acc_x"] *= -1
    rotated["acc_y"] *= -1
    rotated["acc_z"] *= -1
    rotated["gyro_x"] *= -1
    rotated["gyro_y"] *= -1
    rotated["gyro_z"] *= -1

    if axis == 'x':
        rotated["pitch"] = (rotated['pitch'] + pi) % (2*pi)
    elif axis == 'y':
        rotated["roll"] = (rotated['roll'] + pi) % (2*pi)
    elif axis == 'z':
        rotated["azimuth"] = (rotated['azimuth'] + pi) % (2*pi)

    return rotated

#adds random noise to the values
def random_noise(motion, noise_percentage = 15):
    noisy = motion.copy()

    noisy["acc_x"] *= float(random.randint(100 - noise_percentage, 100 + noise_percentage))/100
    noisy["acc_y"] *= float(random.randint(100 - noise_percentage, 100 + noise_percentage))/100
    noisy["acc_z"] *= float(random.randint(100 - noise_percentage, 100 + noise_percentage))/100
    noisy["gyro_x"] *= float(random.randint(100 - noise_percentage, 100 + noise_percentage))/100
    noisy["gyro_y"] *= float(random.randint(100 - noise_percentage, 100 + noise_percentage))/100
    noisy["gyro_z"] *= float(random.randint(100 - noise_percentage, 100 + noise_percentage))/100
    noisy["azimuth"] *= float(random.randint(100 - noise_percentage, 100 + noise_percentage))/100
    noisy["pitch"] *= float(random.randint(100 - noise_percentage, 100 + noise_percentage))/100
    noisy["roll"] *= float(random.randint(100 - noise_percentage, 100 + noise_percentage))/100

    return noisy

#translates the graph left or right with a random step (35 is roughly 10%)
def random_translation(motion_list, min_translation = 100, max_translation = 150):
    #random step creation
    rand_step = random.randint(min_translation, max_translation)
    direction = 1 if random.randint(-1, 1) > 0 else -1
    rand_step = direction * rand_step
    
    #the offset is to compensate for the gravity acceleration
    offset = {}
    if direction == 1:
        offset["acc_x"] = motion_list[-1]["acc_x"]
        offset["acc_y"] = motion_list[-1]["acc_y"]
        offset["acc_z"] = motion_list[-1]["acc_z"]
        offset["azimuth"] = motion_list[-1]["azimuth"]
        offset["pitch"] = motion_list[-1]["pitch"]
        offset["roll"] = motion_list[-1]["roll"]
    else:
        offset["acc_x"] = motion_list[0]["acc_x"]
        offset["acc_y"] = motion_list[0]["acc_y"]
        offset["acc_z"] = motion_list[0]["acc_z"]
        offset["azimuth"] = motion_list[0]["azimuth"]
        offset["pitch"] = motion_list[0]["pitch"]
        offset["roll"] = motion_list[0]["roll"]



    translated_copy = []

    for i in range(len(motion_list)):
        copy = {}
        copy["label"] = motion_list[i]["label"]
        if (i + rand_step) >= 0 and (i + rand_step) < len(motion_list):
            copy["acc_x"] = motion_list[i + rand_step]["acc_x"]
            copy["acc_y"] = motion_list[i + rand_step]["acc_y"]
            copy["acc_z"] = motion_list[i + rand_step]["acc_z"]
            copy["gyro_x"] = motion_list[i + rand_step]["gyro_x"]
            copy["gyro_y"] = motion_list[i + rand_step]["gyro_y"]
            copy["gyro_z"] = motion_list[i + rand_step]["gyro_z"]
            copy["azimuth"] = motion_list[i + rand_step]["azimuth"]
            copy["pitch"] = motion_list[i + rand_step]["pitch"]
            copy["roll"] = motion_list[i + rand_step]["roll"]
        else:
            copy["acc_x"] = offset["acc_x"] + (1 if random.randint(-1, 1) > 0 else -1) * float(random.randint(1, 100))/10e5
            copy["acc_y"] = offset["acc_y"] + (1 if random.randint(-1, 1) > 0 else -1) * float(random.randint(1, 100))/10e5
            copy["acc_z"] = offset["acc_z"] + (1 if random.randint(-1, 1) > 0 else -1) * float(random.randint(1, 100))/10e5
            copy["gyro_x"] =(1 if random.randint(-1, 1) > 0 else -1) * float(random.randint(1, 100))/10e5
            copy["gyro_y"] =(1 if random.randint(-1, 1) > 0 else -1) * float(random.randint(1, 100))/10e5
            copy["gyro_z"] =(1 if random.randint(-1, 1) > 0 else -1) * float(random.randint(1, 100))/10e5
            copy["azimuth"] = offset["azimuth"] + (1 if random.randint(-1, 1) > 0 else -1) * float(random.randint(1, 100))/10e5
            copy["pitch"] = offset["pitch"] + (1 if random.randint(-1, 1) > 0 else -1) * float(random.randint(1, 100))/10e5
            copy["roll"] = offset["roll"] + (1 if random.randint(-1, 1) > 0 else -1) * float(random.randint(1, 100))/10e5
            
        translated_copy.append(copy)
    return translated_copy

#data_list is a stack of 6 tensors of length 1000
def get_rotated_data(axis, data_list):
    translated = random_translation(data_list)
    rotated = [rotate(elem, axis) for elem in translated]

    noisy_AccX =[random_noise(elem)["acc_x"] for elem in rotated]
    noisy_AccY =[random_noise(elem)["acc_y"] for elem in rotated]
    noisy_AccZ =[random_noise(elem)["acc_z"] for elem in rotated]
    noisy_RotX =[random_noise(elem)["gyro_x"] for elem in rotated]
    noisy_RotY =[random_noise(elem)["gyro_y"] for elem in rotated]
    noisy_RotZ =[random_noise(elem)["gyro_z"] for elem in rotated]
    noisy_pitch =[random_noise(elem)["pitch"] for elem in rotated]
    noisy_roll =[random_noise(elem)["roll"] for elem in rotated]
    noisy_azimuth =[random_noise(elem)["azimuth"] for elem in rotated]


    rotated_data = [noisy_AccX, noisy_AccY, noisy_AccZ, noisy_RotX, noisy_RotY, noisy_RotZ, noisy_pitch, noisy_roll, noisy_azimuth]

    return rotated_data