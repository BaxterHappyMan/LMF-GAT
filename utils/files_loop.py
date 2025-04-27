import os


class LoopWaveFile:
    """
    Args:
        root: the root of pcm or wave file
    """
    def __init__(self, root: str):
        self.root = root
        self.pcm_list = None
        self.wav_list = None
        self._initialize()

    def _initialize(self):
        f_type, file_list = self.loop()
        if f_type == 'pcm':
            self.pcm_list = file_list
        else:
            self.wav_list = file_list

    def loop(self):
        f_type = ''
        all_files = []
        if self.root is not None:
            for roots, dirs, files in os.walk(self.root):
                for file in files:
                    _, extension = os.path.splitext(file)
                    if extension == '.wav':
                        f_type = 'wav'
                        file_path = os.path.join(roots, file)
                        all_files.append(file_path)
                    if extension == '.pcm':
                        f_type = 'pcm'
                        file_path = os.path.join(roots, file)
                        all_files.append(file_path)
        return f_type, all_files
