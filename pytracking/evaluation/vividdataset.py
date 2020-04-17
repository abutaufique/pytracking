import numpy as np
import os

from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList

def VIVIDDataset():
    return VIVIDDatasetClass().get_sequence_list()


class VIVIDDatasetClass(BaseDataset):
    """ VIVID dataset.
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vivid_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/frame{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
            sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        #anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        #try:
        #    ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        #except:
        #    ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)
        anno = sequence_info['anno_path']
        ground_truth_rect = np.array(anno.split(',')).reshape(1,-1).astype(np.float32)

        return Sequence(sequence_info['name'], frames, ground_truth_rect[init_omit:,:])

    def __len__(self):
        return len(self.sequence_info_list)


    def _get_sequence_info_list(self):

        sequence_info_list = [
            {"name": "egtest01", "path": "egtest01", "startFrame": 0, "endFrame":1820 , "nz": 5,
             "ext": "jpg", "anno_path": "119,15,21,25"},
            {"name": "egtest02", "path": "egtest02", "startFrame": 0, "endFrame": 1300, "nz": 5,
             "ext": "jpg", "anno_path": "437,369,46,37"},
            {"name": "egtest03", "path": "egtest03", "startFrame": 0, "endFrame":2570 , "nz": 5,
             "ext": "jpg", "anno_path": "439,274,49,33"},
            {"name": "egtest04", "path": "egtest04", "startFrame": 0, "endFrame":1832 , "nz": 5,
             "ext": "jpg", "anno_path": "321,171,16,11"},
            {"name": "egtest05", "path": "egtest05", "startFrame": 0, "endFrame":1763 , "nz": 5,
             "ext": "jpg", "anno_path": "21,322,87,43"},
            {"name": "pktest01", "path": "pktest01", "startFrame": 0, "endFrame":1459 , "nz": 5,
             "ext": "jpg", "anno_path": "114,123,30,14"},
            {"name": "pktest02", "path": "pktest02", "startFrame": 0, "endFrame":1594 , "nz": 5,
             "ext": "jpg", "anno_path": "249,164,32,20"},
            {"name": "pktest03", "path": "pktest03", "startFrame": 0, "endFrame":2010 , "nz": 5,
             "ext": "jpg", "anno_path": "142,117,18,10"}
        ]

        return sequence_info_list
