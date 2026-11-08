"""highway-env 原生物理子步录像；不使用 BEV 代替环境画面。"""
from __future__ import annotations
import numpy as np
import pygame
from highway_env.envs.common.graphics import EnvViewer, ObservationGraphics
from highway_env.road.graphics import RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics
from .paths import filesystem_path


class PlannedTrajectoryViewer(EnvViewer):
    """沿用1.12.1原生绘制顺序，在真实交通对象下方叠加最近规划。"""
    PLAN_COLOR = (0, 230, 255)

    def _draw_plan(self):
        points = getattr(self.env, '_planned_trajectory', None)
        if points is None:
            return False
        # 每次使用原生相机的当前原点、缩放；世界Y与native屏幕Y相反。
        pixels = [self.sim_surface.pos2pix(float(x), float(-y)) for x, y in points]
        pygame.draw.lines(self.sim_surface, (15, 45, 55), False, pixels, 6)
        pygame.draw.lines(self.sim_surface, self.PLAN_COLOR, False, pixels, 3)
        pygame.draw.circle(self.sim_surface, self.PLAN_COLOR, pixels[-1], 5, 2)
        return True

    def _draw_plan_label(self):
        if not hasattr(self, '_plan_label'):
            font_path = pygame.font.match_font(['microsoftyahei', 'simhei', 'notosanscjksc', 'wenquanyizenhei'])
            font = pygame.font.Font(font_path, 18)
            # 无中文系统字体时保留明确的计划标识，不显示缺字方框。
            self._plan_label = font.render('规划轨迹' if font_path else 'MPC PLAN', True, self.PLAN_COLOR)
        label = self._plan_label
        pygame.draw.rect(self.sim_surface, (15, 25, 35), (8, 8, label.get_width()+12, label.get_height()+8))
        self.sim_surface.blit(label, (14, 12))

    def display(self):
        """只插入计划图层；窗口、交通、观测及录像仍使用原生缓冲。"""
        if not self.enabled:
            return
        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)
        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(self.vehicle_trajectory, self.sim_surface, offscreen=self.offscreen)
        RoadGraphics.display_road_objects(self.env.road, self.sim_surface, offscreen=self.offscreen)
        plan_drawn = self._draw_plan()
        if EnvViewer.agent_display:
            EnvViewer.agent_display(self.agent_surface, self.sim_surface)
            if not self.offscreen:
                offset = ((0, self.config['screen_height']) if self.config['screen_width'] > self.config['screen_height']
                          else (self.config['screen_width'], 0))
                self.screen.blit(self.agent_surface, offset)
        # 真实车辆后绘制，不能被计划线替换成预测的自车状态。
        RoadGraphics.display_traffic(self.env.road, self.sim_surface,
            simulation_frequency=self.env.config['simulation_frequency'], offscreen=self.offscreen)
        ObservationGraphics.display(self.env.observation_type, self.sim_surface)
        if plan_drawn:
            self._draw_plan_label()
        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            if self.env.config['real_time_rendering']:
                self.clock.tick(self.env.config['simulation_frequency'])
            pygame.display.flip()
        if self.SAVE_IMAGES and self.directory:
            pygame.image.save(self.sim_surface, str(self.directory / f'highway-env_{self.frame}.png'))
            self.frame += 1


class NativeVideoRecorder:
    def __init__(self,adapter,path,fps):
        self.adapter=adapter
        self.path=filesystem_path(path)
        self.frames_per_sec=float(fps)
        self.frames=0
        self.writer=None
        self.size=None
        self.closed=False
        self.frames_with_plan=0

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
        self.frames_with_plan+=int(getattr(self.adapter.env,'_planned_trajectory',None) is not None)

    def close(self):
        if self.writer is not None:self.writer.release()
        self.closed=True

    def metadata(self):
        return {'path':str(self.path),'frames':self.frames,'fps':self.frames_per_sec,
                'size':self.size,'simulated_duration_s':max(0,self.frames-1)/self.frames_per_sec,
                'encoded_duration_s':self.frames/self.frames_per_sec,
                'source':'highway_env_native_physics_substeps','plan_overlay':self.frames_with_plan>0,
                'frames_with_plan':self.frames_with_plan,
                'plan_overlay_description':'青色线为最近一次MPC计划轨迹，车辆仍显示真实当前状态'}
