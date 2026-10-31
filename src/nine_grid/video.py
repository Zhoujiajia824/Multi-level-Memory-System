"""highway-env 原生物理子步录像；不使用 BEV 代替环境画面。"""
from __future__ import annotations
import numpy as np
from .paths import filesystem_path


class NativeVideoRecorder:
    def __init__(self,adapter,path,fps):
        self.adapter=adapter
        self.path=filesystem_path(path)
        self.frames_per_sec=float(fps)
        self.frames=0
        self.writer=None
        self.size=None
        self.closed=False

    def _capture_frame(self):
        """由原生 RecordVideo 接口调用，也接收每步最后一帧。"""
        if self.closed:raise RuntimeError('录像已关闭，不能继续写帧')
        rgb=self.adapter.render()
        if rgb is None:raise RuntimeError('原生渲染未返回录像画面')
        rgb=np.asarray(rgb)
        if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:
            raise ValueError('原生录像帧必须为 RGB uint8')
        import cv2
        size=(rgb.shape[1],rgb.shape[0])
        if self.writer is None:
            if any(length%2 for length in size):raise ValueError('MP4画面尺寸必须为偶数')
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self.writer=cv2.VideoWriter(str(self.path),cv2.VideoWriter_fourcc(*'mp4v'),self.frames_per_sec,size)
            if not self.writer.isOpened():raise RuntimeError('MP4编码器初始化失败')
            self.size=size
        if size!=self.size:raise ValueError('同一轮录像不能改变画面尺寸')
        self.writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        self.frames+=1

    def close(self):
        if self.writer is not None:self.writer.release()
        self.closed=True

    def metadata(self):
        return {'path':str(self.path),'frames':self.frames,'fps':self.frames_per_sec,
                'size':self.size,'simulated_duration_s':max(0,self.frames-1)/self.frames_per_sec,
                'encoded_duration_s':self.frames/self.frames_per_sec,
                'source':'highway_env_native_physics_substeps','plan_overlay':False}
