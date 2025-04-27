import os
import numpy as np
from utils.audio_parser import AudioParser
from utils.files_loop import LoopWaveFile


def save2npz(file_list, npz_path):
    graph_data = []
    for file in file_list:
        audio_parser = AudioParser(file)
        x = audio_parser.magnitude_matrix.T
        adj = audio_parser.coo_adjacency
        y = audio_parser.label
        node_y = audio_parser.node_label
        print(audio_parser.rate)

        graph_data.append({
            "x": x,
            "adj": adj,
            "y": y,
            "node_y": node_y
        })
    # np.savez(npz_path, **{f"graph{i}": data for i, data in enumerate(graph_data)})

# 将ENFI数据集所有类型的设备声音读取保存为npz文件，区分正常和异常
def save_all_machine(root='/workspace/ENFI-AudioDatasets/Sensor'):
    # 读取所有的wav文件
    loop_wav = LoopWaveFile(root)
    wav_files = loop_wav.wav_list
    # 区分设备，总共五类设备
    machine_li = ["IncludeDraftFan", "UnloaderValve"]
    machine_list = ["IncludeDraftFanN", "UnloaderValveN", "IncludeDraftFanA", "UnloaderValveA"]
    machine_dict = {item: [] for item in machine_list}
    for file in wav_files:
        for i in range(len(machine_li)):
            if machine_li[i] in file:
                if 'normal' in file or 'Normal' in file:
                    machine_dict[machine_list[i]].append(file)
                else:
                    machine_dict[machine_list[i+2]].append(file)
    # 保存处理后的ENFI图数据集
    npz_folder = '/workspace/First/ENFI-GraphDatasets/Sensor'
    if not os.path.exists(npz_folder):
        os.makedirs(npz_folder)
    for key, value in machine_dict.items():
        npz_path = npz_folder + '/' + key +'.npz'
        save2npz(file_list=value, npz_path=npz_path)

# 将phone采样的设备声音信号保存为npz文件
def save_all_machine_v2(root='/workspace/ENFI-AudioDatasets/Phone'):
    # 读取所有的wav文件
    loop_wav = LoopWaveFile(root)
    wav_files = loop_wav.wav_list
    # 区分设备，总共五类设备
    machine_li = ["IncludeDraftFan", "UnloaderValve", "HydraulicStationOilTank", "PrimaryAirFan", "SecondaryAirFan"]
    machine_list = ["IncludeDraftFanN", "UnloaderValveN", "HydraulicStationOilTankN", "PrimaryAirFanN", "SecondaryAirFanN",
                    "IncludeDraftFanA", "UnloaderValveA", "HydraulicStationOilTankA", "PrimaryAirFanA", "SecondaryAirFanA"]
    machine_dict = {item: [] for item in machine_list}
    for file in wav_files:
        for i in range(len(machine_li)):
            if machine_li[i] in file:
                if 'normal' in file or 'Normal' in file:
                    machine_dict[machine_list[i]].append(file)
                else:
                    machine_dict[machine_list[i + 5]].append(file)
    # 保存处理后的ENFI图数据集
    npz_folder = '/workspace/First/ENFI-GraphDatasets/Phone'
    if not os.path.exists(npz_folder):
        os.makedirs(npz_folder)
    for key, value in machine_dict.items():
        npz_path = npz_folder + '/' + key + '.npz'
        save2npz(file_list=value, npz_path=npz_path)

# 将DCASE 2024数据集所有类型的设备声音读取保存为npz文件，区分正常和异常
def save_all_machine_dcase(root='/workspace/DCASE-Challenge/DevelopmentDataset'):
    # 读取所有的wav文件
    loop_wav = LoopWaveFile(root)
    wave_files = loop_wav.wav_list
    # 区分设备，总共七类设备
    machine_li = ['dev_bearing', 'dev_fan', 'dev_gearbox', 'dev_slider', 'dev_ToyCar', 'dev_ToyTrain', 'dev_valve']
    machine_list = ['dev_bearingN', 'dev_fanN', 'dev_gearboxN', 'dev_sliderN', 'dev_ToyCarN', 'dev_ToyTrainN', 'dev_valveN',
                    'dev_bearingA', 'dev_fanA', 'dev_gearboxA', 'dev_sliderA', 'dev_ToyCarA', 'dev_ToyTrainA', 'dev_valveA']
    machine_dict = {item: [] for item in machine_list}
    for file in wave_files:
        for i in range(len(machine_li)):
            if machine_li[i] in file:
                if 'normal' in file or 'Normal' in file:
                    machine_dict[machine_list[i]].append(file)
                else:
                    machine_dict[machine_list[i+7]].append(file)
    # 保存处理后的DCASE 2024数据集
    npz_folder = '/workspace/First/DCASE-GraphDatasets'
    if not os.path.exists(npz_folder):
        os.makedirs(npz_folder)
    for key, value in machine_dict.items():
        npz_path = npz_folder + '/' + key + '.npz'
        save2npz(file_list=value, npz_path=npz_path)

# 严格按照DCASE 2024挑战赛的标准划分数据集，即实现First-Shot任务（无监督），DevelopmentDataset
def save_all_machine_dcase_v1(root='/workspace/DCASE-Challenge/DevelopmentDataset'):
    # 读取所有的wav文件
    loop_wav = LoopWaveFile(root)
    wave_files = loop_wav.wav_list
    # 区分设备，不区分正常异常
    machine_li = ['dev_bearing', 'dev_fan', 'dev_gearbox', 'dev_slider', 'dev_ToyCar', 'dev_ToyTrain', 'dev_valve']
    data_type = ['train', 'test']
    machine_dict = {machine + '_' + typ: [] for machine in machine_li for typ in data_type}
    machine_dict_keys = list(machine_dict.keys())
    print(machine_dict_keys)
    for file in wave_files:
        for i in range(len(machine_li)):
            if machine_li[i] in file:
                if data_type[0] in file:
                    machine_dict[machine_dict_keys[i*2]].append(file)
                else:
                    machine_dict[machine_dict_keys[i*2+1]].append(file)
    # 保存处理后的DCASE 2024数据集
    npz_folder = '/workspace/DCASE-Challenge-2024/DevelopmentDataset'
    if not os.path.exists(npz_folder):
        os.makedirs(npz_folder)
    for key, value in machine_dict.items():
        npz_path = npz_folder + '/' + key + '.npz'
        save2npz(file_list=value, npz_path=npz_path)


if __name__ == '__main__':
    # save_all_machine()
    save_all_machine_v2()
    # save_all_machine_dcase()
    # save_all_machine_dcase_v1()
    # pass