from cmath import pi
import csv
import pathlib
import Data_Augmentation

labels = {"STD" : 0,
          "WAL" : 1,
          "JOG" : 2,
          "JUM" : 3,
          "STU" : 4,
          "STN" : 5,
          "SCH" : 6,
          "SIT" : 7,
          "CHU" : 8,
          "CSI" : 9,
          "CSO" : 10,
          "FOL" : 11,
          "FKL" : 12,
          "BSC" : 13,
          "SDL" : 14
          #"PFF" : 15
          }

def get_datalist():
    data_list = [[]]

    for path in pathlib.Path("sampled_data").iterdir():
        if path.is_file():
            current_file = open(path, "r")
            csv_reader = csv.DictReader(current_file)

            for row in csv_reader:
                if row["acc_x"] == "":
                    data_list.append([])
                    continue

                new_data = {
                            "acc_x" : float(row["acc_x"]),
                            "acc_y" : float(row["acc_y"]),
                            "acc_z" : float(row["acc_z"]),
                            "gyro_x": float(row["gyro_x"]),
                            "gyro_y": float(row["gyro_y"]),
                            "gyro_z": float(row["gyro_z"]),
                            #"azimuth":float(row["azimuth"])*pi/180,
                            #"pitch" : float(row["pitch"])*pi/180,
                            #"roll" :  float(row["roll"])*pi/180, 
                            "label" : row["label"]
                        }
                if(len(data_list[-1])>0):
                    new_data["acc_x_change"] = new_data["acc_x"]-data_list[-1][-1]["acc_x"]
                    new_data["acc_y_change"] = new_data["acc_y"]-data_list[-1][-1]["acc_y"]
                    new_data["acc_z_change"] = new_data["acc_z"]-data_list[-1][-1]["acc_z"]
                    new_data["gyro_x_change"] = new_data["gyro_x"]-data_list[-1][-1]["gyro_x"]
                    new_data["gyro_y_change"] = new_data["gyro_y"]-data_list[-1][-1]["gyro_y"]
                    new_data["gyro_z_change"] = new_data["gyro_z"]-data_list[-1][-1]["gyro_z"]
                else:
                    new_data["acc_x_change"] = 0
                    new_data["acc_y_change"] = 0
                    new_data["acc_z_change"] = 0
                    new_data["gyro_x_change"] = 0
                    new_data["gyro_y_change"] = 0
                    new_data["gyro_z_change"] = 0

                data_list[-1].append(new_data)

    while True:
        try:
            data_list.remove([])
        except:
            break

    return data_list

def get_dataset(augment = False, reduced_frequency = False, target_frequency_ratio = 1):

    data_list = get_datalist()
    total_dataset = []

    for i in range(len(data_list)):
        sample = data_list[i]

        Acc_x = [elem["acc_x"] for elem in sample]
        Acc_y = [elem["acc_y"] for elem in sample]
        Acc_z = [elem["acc_z"] for elem in sample]
        Gyro_x = [elem["gyro_x"] for elem in sample]
        Gyro_y = [elem["gyro_y"] for elem in sample]
        Gyro_z = [elem["gyro_z"] for elem in sample]
        Acc_x_change = [elem["acc_x_change"] for elem in sample]
        Acc_y_change = [elem["acc_y_change"] for elem in sample]
        Acc_z_change = [elem["acc_z_change"] for elem in sample]
        Gyro_x_change = [elem["gyro_x_change"] for elem in sample]
        Gyro_y_change = [elem["gyro_y_change"] for elem in sample]
        Gyro_z_change = [elem["gyro_z_change"] for elem in sample]
        #Azimuth= [elem["azimuth"] for elem in sample]
        #Pitch =  [elem["pitch"] for elem in sample]
        #Roll =   [elem["roll"] for elem in sample]
        
        #Try to add the proximity sensor
        # https://itnext.io/android-proximity-sensor-as-clear-as-possible-593774d90dd2

        original_data = [Acc_x, Acc_y, Acc_z, Gyro_x, Gyro_y, Gyro_z, 
                         Acc_x_change,Acc_y_change,Acc_z_change,Gyro_x_change,
                         Gyro_y_change,Gyro_z_change]#, Pitch, Roll, Azimuth]
        label = sample[0]["label"]

        total_dataset.append((original_data, label))

        #Data augmentation
        x_rotated = []
        y_rotated = []
        z_rotated = []
        '''
        if(augment):
            x_rotated = Data_Augmentation.get_rotated_data('x', sample)
            y_rotated = Data_Augmentation.get_rotated_data('y', sample)
            z_rotated = Data_Augmentation.get_rotated_data('z', sample)
            total_dataset.append((x_rotated, label))
            total_dataset.append((y_rotated, label))
            total_dataset.append((z_rotated, label))
        '''
    for key, value in labels.items():
        print(key + " " + str(len([elem for elem in total_dataset if elem[1]== key])))
        
    if(reduced_frequency):
        assert(target_frequency_ratio >= 1)
        reduced_frequency_dataset = []
        for i in range(len(total_dataset)):
            if(total_dataset[i][1] != "LAY" and total_dataset[i][1] != "PFF"):
                reduced_frequency_dataset.append(([], total_dataset[i][1]))
                for j in range(len(total_dataset[i][0])):
                    entry = total_dataset[i][0][j][0:-1:target_frequency_ratio]
                    reduced_frequency_dataset[i][0].append(entry)
            else:
                reduced_frequency_dataset.append(([], total_dataset[i][1]))
                for j in range(len(total_dataset[i][0])):
                    entry = total_dataset[i][0][j]
                    reduced_frequency_dataset[i][0].append(entry)

        return reduced_frequency_dataset

    

    return total_dataset
            