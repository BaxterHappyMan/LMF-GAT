import numpy as np
import soundfile as sf
from scipy.signal.windows import hann
from scipy.fft import fft
import scipy.sparse as sp


class AudioParser:
    def __init__(self, audio_file_path, win_len=2048, hop_len=512, n_fft=2048):
        self.audio_file_path = audio_file_path
        self.win_len = win_len
        self.hop_len = hop_len
        self.n_fft = n_fft
        self.stft_matrix = None
        self.magnitude_matrix = None
        self.phase_matrix = None
        self.feature = None
        self.adjacency = None
        self.coo_adjacency = None
        self.node_label = None
        self.label = None
        self.mfcc_matrix = None
        self._initialize()

    def _initialize(self):
        self.raw_data, self.rate = self.audio_reader()
        self.calc_stft()
        self.calc_adjacency()

    def audio_reader(self):
        data, rate = sf.read(self.audio_file_path)
        if data.shape[-1] == 2:
            data = (data[:, 0] + data[:, 1])/2
        return data, rate

    def calc_stft(self):
        window = hann(self.win_len)
        n_per_seg = len(window)
        if self.win_len != self.n_fft:
            print('Please check the window length and fft length')
        n_seg = (len(self.raw_data) - n_per_seg) // self.hop_len + 1
        # 初始化STFT矩阵
        self.stft_matrix = np.zeros((self.n_fft//2+1, n_seg), dtype=complex)
        # 计算每一段的傅里叶变换
        for i in range(n_seg):
            start = i*self.hop_len
            end = start + n_per_seg
            segment = self.raw_data[start:end] * window
            self.stft_matrix[:, i] = fft(segment, self.n_fft)[:self.n_fft // 2 + 1]
        # 分离幅度谱和相位
        self.magnitude_matrix = np.nan_to_num(np.abs(self.stft_matrix)) # 消除NaN值
        self.phase_matrix = np.angle(self.stft_matrix)
        # 拼接幅度和相位特征
        log_magnitude = 20 * np.log10(self.magnitude_matrix + 1e-6)
        min_magval = np.min(log_magnitude)
        max_magval = np.max(log_magnitude)
        normalized_log_magnitude = (log_magnitude - min_magval) / (max_magval - min_magval)
        normalized_phase_periodic = self.phase_matrix / np.pi
        if self.feature is None:
            self.feature = np.concatenate((normalized_log_magnitude, normalized_phase_periodic), axis=0)
            # 消除NaN值
            self.feature = np.nan_to_num(self.feature)
        if 'normal' in self.audio_file_path or 'Normal' in self.audio_file_path:
            if self.label is None:
                self.label = 0
            if self.node_label is None:
                self.node_label = np.zeros(n_seg, dtype=int)
        else:
            if self.label is None:
                self.label = 1
            if self.node_label is None:
                self.node_label = np.ones(n_seg, dtype=int)


    def calc_adjacency(self):
        # 使用原始特征计算边
        if self.feature is None:
            self.calc_stft()
        # 计算特征向量点积
        dot_product = np.dot(self.feature.T, self.feature)
        # 计算每个节点特征向量的模长
        norm = np.linalg.norm(self.feature.T, axis=1).reshape(-1, 1)
        # 计算余弦相似性矩阵
        epsilon = 1e-8
        cosine_sim = dot_product / (norm * norm.T + epsilon)
        # 七五分位作为阈值设置边
        percent_75 = np.percentile(cosine_sim, 75)
        if self.adjacency is None:
            self.adjacency = (cosine_sim > percent_75).astype(np.float32)
        # 转换为COO格式稀疏矩阵
        if self.coo_adjacency is None:
            coo_matrix = sp.coo_matrix(self.adjacency)
            row_indices = coo_matrix.row
            col_indices = coo_matrix.col
            coo_matrix = np.vstack([row_indices, col_indices])
            self.coo_adjacency = coo_matrix
