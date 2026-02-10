
from scipy.signal import find_peaks
import numpy as np
from scipy.stats import norm
from obspy import Stream, Trace,read

import matplotlib.pyplot as plt
from scipy import signal
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from scipy.signal import find_peaks
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from datetime import datetime
from pathlib import Path
from copy import deepcopy


def interpolate_middle_nans(veh_state):
    """
    只对中间的 NaN 进行插值，保留头尾的 NaN。
    """
    veh_state = np.asarray(veh_state)
    isnan = np.isnan(veh_state)

    if not np.any(~isnan):
        # 全是 nan，返回原数组
        return veh_state.copy()

    # 找到第一个非nan和最后一个非nan的索引
    first_valid = np.argmax(~isnan)
    last_valid = len(veh_state) - np.argmax(~isnan[::-1]) - 1

    # 中间部分进行插值
    interp_vals = veh_state.copy()
    x = np.arange(len(veh_state))
    x_valid = x[first_valid:last_valid + 1][~isnan[first_valid:last_valid + 1]]
    y_valid = veh_state[first_valid:last_valid + 1][~isnan[first_valid:last_valid + 1]]

    interp_vals[first_valid:last_valid + 1] = np.interp(
        x[first_valid:last_valid + 1], x_valid, y_valid
    )

    return interp_vals

def likelihood_1d(peak_loc, das_time_ds, sigma):
    """
    估计峰值的概率分布（似然性）
    """
    likelihood = np.zeros(len(das_time_ds))
    for j in range(len(peak_loc)):
        likelihood += norm.pdf(das_time_ds, loc=das_time_ds[peak_loc[j]], scale=sigma)
    return likelihood

class KF_tracking:
    def __init__(self, data, t_axis, x_axis, args):
        self.data = data
        self.t_axis = t_axis
        self.x_axis = x_axis
        self.dx = x_axis[1] - x_axis[0]
        self.args = args


    def detect_in_one_section(self, start_x, nx=1, pick_args=None, sigma=0.1, pclip=98, show_plot=False, plt_xlim=1000):
        """
        在某一区域内通过多个传感器检测车辆初始位置
        """
        if pick_args is None:
            pick_args = self.args["detect"]

        prominence = pick_args["prominence"]
        distance   = pick_args["distance"]
        wlen       = pick_args["wlen"]
        height     = pick_args.get("height", None)

        peak_erode = np.zeros(len(self.t_axis))
        start_x_idx = np.argmin(np.abs(start_x - self.x_axis))

        all_peaks = []
        for i in range(nx):
            signal = self.data[start_x_idx + i]
            peaks = find_peaks(signal, prominence=prominence, wlen=wlen, height=height, distance=distance)[0]
            all_peaks.extend(peaks)

        if len(all_peaks) == 0:
            return np.array([])

        # 对峰值进行排序和非极大值抑制，确保不重叠
        all_peaks = np.array(sorted(all_peaks))
        selected_peaks = [all_peaks[0]]
        min_interval = 3000  # 以采样点为单位，避免过密重复车（可以调）

        for p in all_peaks[1:]:
            if np.min(np.abs(np.array(selected_peaks) - p)) > min_interval:
                selected_peaks.append(p)

        return np.array(selected_peaks)


    def tracking_visualization_one_section(self, start_x, tracked_v, trace=None, weak_states=None, fig=None, ax=None):
        """
        绘制跟踪结果的剖面图（水平section图）
        """
        start_x_idx = np.argmin(np.abs(start_x - self.x_axis))
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 6))

        st_visual = Stream()
        for i, d in enumerate(self.data):
            tr = Trace(data=d)
            tr.stats.distance  = self.x_axis[i]
            tr.stats.starttime = self.t_axis[0]
            tr.stats.delta     = self.t_axis[1] - self.t_axis[0]
            st_visual.append(tr)

        st_visual.plot(type='section', scale=1.0, orientation='horizontal', fig=fig)

        if trace is not None:
            v = trace
            t_values = tracked_v[v][~np.isnan(tracked_v[v])].astype(int)
            x_indices = np.where(~np.isnan(tracked_v[v]))[0] + start_x_idx
            ax.plot(self.t_axis[t_values], self.x_axis[x_indices] * 1e-3, 'x', markersize=5, color='red')
        else:
            for v in range(tracked_v.shape[0]):
                t_values = tracked_v[v][~np.isnan(tracked_v[v])].astype(int)
                x_indices = np.where(~np.isnan(tracked_v[v]))[0] + start_x_idx
                ax.plot(self.t_axis[t_values], self.x_axis[x_indices] * 1e-3, 'o', markersize=5, label='Observation' if v == 0 else "")

                # 绘制弱观测（灰色圈圈）
                if weak_states is not None:
                    weak_mask = (np.isnan(tracked_v[v])) & (~np.isnan(weak_states[v]))

                    weak_t = weak_states[v][weak_mask].astype(int)
                    weak_t = np.clip(weak_t, 0, len(self.t_axis) - 1)   # 🔧 防止越界
                    weak_x = np.where(weak_mask)[0] + start_x_idx
                    ax.plot(self.t_axis[weak_t], self.x_axis[weak_x] * 1e-3, '+', markersize=4, color='k', label='Weak Prediction' if v == 0 else "")


    def _init_state(self, veh_states, start_x_idx, veh_base):
        """
        初始化每辆车的状态向量 Tkk 和协方差矩阵 Pkk
        """
        n_veh = len(veh_base)
        Tkk   = np.full((2, n_veh), np.nan)
        Pkk   = np.full((2, 2, n_veh), np.nan)
        Xv    = np.full(n_veh, np.nan)

        for v in range(n_veh):
            val = veh_states[v, ~np.isnan(veh_states[v])]
            if len(val) > 0:
                Tkk[:, v] = [val[0], 25.0]
                # Pkk[:, :, v] = np.array([[625, 0], [0, 25]])
                Pkk[:, :, v] = np.array([[100,   0],    # 位置方差：允许轻微调整（10个采样点级别）
                                          [  0,  4]])      # 速度方差：非常自信（±2 m/s）范围内不会轻易调整
                Xv[v] = self.x_axis[start_x_idx]
        return Tkk, Pkk, Xv

    def _predict_state(self, Tkk_v, Pkk_v, dx, sigma_a):
        """
        使用状态转移矩阵 A 和过程噪声 Q 进行状态预测
        """
        A = np.array([[1, dx], [0, 1]])
        # Q = sigma_a * np.array([[0.25 * dx**4, 0.5 * dx**3],
        #                         [0.5 * dx**3, dx**2]])
        Q = sigma_a * np.array([[0.25 * dx**4, 0.5 * dx**3],
                        [0.5 * dx**3,  0.1 * dx**2]])  # ❗️速度项更保守
        Tk1k = A @ Tkk_v
        Pk1k = A @ Pkk_v @ A.T + Q
        return Tk1k, Pk1k

    def _select_peak(self, peak_loc, pred_pos, tmin=-4000, tmax=6000):
        """
        选择最接近预测位置的合法峰值
        """
        dist_tmp = peak_loc - pred_pos
        idx_valid = np.where((dist_tmp > tmin) & (dist_tmp <= tmax))[0]
        if len(idx_valid) == 0:
            return None
        best_idx = idx_valid[np.argmin(np.abs(peak_loc[idx_valid] - pred_pos))]
        return peak_loc[best_idx]

    def _select_center(self, data_row, pred_pos, tmin=-4000, tmax=6000, mode="energy_center"):
        """
        在预测附近时间窗内选择中心点。
        mode:
          - energy_center: 能量加权质心
          - main_peak: 窗内最大幅值点
        """
        n = len(data_row)
        left = max(0, int(np.floor(pred_pos + tmin)))
        right = min(n - 1, int(np.ceil(pred_pos + tmax)))
        if left >= right:
            return None
        seg = data_row[left:right + 1]
        if mode == "main_peak":
            idx = int(np.argmax(np.abs(seg)))
            return left + idx
        # energy_center (default)
        w = np.abs(seg) ** 2
        wsum = np.sum(w)
        if wsum == 0:
            return None
        center = np.sum(w * np.arange(left, right + 1)) / wsum
        return int(np.round(center))

    def _update_state(self, Tk1k, Pk1k, obs, C, R, vmin=20.0, vmax=30.0):
        """
        卡尔曼滤波更新步骤
        """
        K      = Pk1k @ C / (R + C @ Pk1k @ C)
        tkk    = Tk1k + K * (obs - C @ Tk1k)
        tkk[1] = np.clip(tkk[1], vmin, vmax)        # 限制速度在 [20, 30] 之间
        Pkk    = Pk1k - np.outer(K, C) @ Pk1k
        return tkk, Pkk

    def tracking_with_veh_base(self, start_x, end_x, veh_base, sigma_a=0.01, pick_args=None, veh_args=None):
        """
        使用空间域贝叶斯滤波方法对车辆进行轨迹跟踪。
        """
        if pick_args is None:
            pick_args = self.args['detect']
        if veh_args is None:
            veh_args = self.args['veh']

        start_x_idx = np.argmin(np.abs(start_x - self.x_axis))
        end_x_idx   = np.argmin(np.abs(end_x - self.x_axis))
        n_veh       = len(veh_base)
        n_stat      = end_x_idx - start_x_idx + 1

        veh_weak_states  = np.full((n_veh, n_stat), np.nan)  # 储存弱观测值
        veh_states       = np.full((n_veh, n_stat), np.nan)
        veh_base_state   = veh_base.copy()
        nan_streak_count = np.zeros(n_veh, dtype=int)
        terminate_flag   = np.zeros(n_veh, dtype=bool)
        Tkk, Pkk, Xv     = self._init_state(veh_states, start_x_idx, veh_base)

        prominence = pick_args.get("prominence", None)
        distance   = pick_args.get("distance", None)
        wlen       = pick_args.get("wlen", None)
        height     = pick_args.get("height", None)
        center_mode = pick_args.get("center_mode", "energy_center")
        vel_init   = veh_args.get("vel_init", 25.0)
        vmin       = veh_args.get("vmin", 20.0)
        vmax       = veh_args.get("vmax", 30.0)
        tmin       = round(veh_args.get("tmin", -4.0) / veh_args.get("dt", 0.001))
        tmax       = round(veh_args.get("tmax", 6.0) / veh_args.get("dt", 0.001))

        C, R = np.array([1, 0]), 10
        Tkk[1, :] = 25.0
        for i in range(start_x_idx, end_x_idx + 1):
            xi = self.x_axis[i]
            peak_loc = find_peaks(self.data[i], prominence=prominence,
                                  wlen=wlen, height=height, distance=distance)[0]

            for v in range(n_veh):
                if terminate_flag[v]:
                    continue

                if i == start_x_idx:
                    Tkk[:, v] = [veh_base[v], vel_init]
                    # Pkk[:, :, v] = np.array([[100, 0], [0, 4]])
                    Pkk[:, :, v] = np.array([[625, 0], [0, 25]])
                    Xv[v] = xi
                    veh_base_state[v] = veh_base[v]
                    veh_states[v, 0] = veh_base[v]


                valid = veh_states[v, ~np.isnan(veh_states[v])]
                if len(valid) == 1:
                    Tkk[:, v] = [valid[0], vel_init]
                    Pkk[:, :, v] = np.array([[625, 0], [0, 25]])
                    # Pkk[:, :, v] = np.array([[100, 0], [0, 4]])
                    Xv[v] = self.x_axis[start_x_idx]
                    # veh_base_state[v] = veh_base[v]

                    dx = xi - Xv[v]
                    Tk1k_v, Pk1k_v = self._predict_state(Tkk[:, v], Pkk[:, :, v], dx, sigma_a)
                    veh_base_state[v] = Tk1k_v[0]
                    Tkk[:, v], Pkk[:, :, v] = Tk1k_v, Pk1k_v

                elif len(valid) > 1:
                    dx = xi - Xv[v]
                    Tk1k_v, Pk1k_v = self._predict_state(Tkk[:, v], Pkk[:, :, v], dx, sigma_a)
                    veh_base_state[v] = Tk1k_v[0]
                    Tkk[:, v], Pkk[:, :, v] = Tk1k_v, Pk1k_v

                pred_pos = veh_base_state[v]
                center_idx = self._select_center(self.data[i], pred_pos, tmin, tmax, mode=center_mode)
                if center_idx is not None:
                    veh_states[v, i - start_x_idx] = center_idx
                    nan_streak_count[v] = 0
                else:
                    # veh_states[v, i - start_x_idx] = pred_pos
                    veh_states[v, i - start_x_idx] = np.nan
                    # veh_weak_states[v, i - start_x_idx] = veh_base_state[v]  # 记录弱观测值
                    nan_streak_count[v] += 1
                    if nan_streak_count[v] >= 4:
                        terminate_flag[v] = True

                veh_weak_states[v, i - start_x_idx] = veh_base_state[v]  # 记录弱观测值
                # 状态更新
                if not np.isnan(veh_states[v, i - start_x_idx]) and np.sum(~np.isnan(veh_states[v])) > 2:
                    obs = veh_states[v, i - start_x_idx]
                    Tkk[:, v], Pkk[:, :, v] = self._update_state(Tkk[:, v], Pkk[:, :, v], obs, C, R, vmin, vmax)
                    Xv[v] = xi

        # 去除轨迹点少于5个的车辆（异常轨迹）
        # min_track_points = 10
        # valid_mask = np.sum(~np.isnan(veh_states), axis=1) >= min_track_points
        # veh_states = veh_states[valid_mask]
        # veh_weak_states = veh_weak_states[valid_mask]

        return veh_states, veh_weak_states



def fit_and_fill_nans(arr, deg=2):
    """
    对1D数组中的非NaN值进行多项式拟合，用于填充所有NaN（包括前后两端）。
    
    参数：
        arr: 1D numpy 数组
        deg: 拟合多项式阶数（建议 2 或 3）
    返回：
        填充后的数组
    """
    arr = np.asarray(arr, dtype=float)
    x_all = np.arange(len(arr))
    mask = ~np.isnan(arr)

    if mask.sum() < deg + 1:
        return arr  # 非NaN点太少，无法拟合

    x_known = x_all[mask]
    y_known = arr[mask]

    coeffs = np.polyfit(x_known, y_known, deg)
    poly_func = np.poly1d(coeffs)
    arr_filled = poly_func(x_all)
    arr[~mask] = arr_filled[~mask]
    arr[arr < 0] = np.nan
    arr[arr > 1000*60*15] = np.nan

    return arr




class TrackingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("车辆轨迹识别系统（手动标注版）")
        
        # Matplotlib 画布
        self.fig    = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        self.ax     = self.fig.add_subplot(111)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        w = QWidget(); w.setLayout(layout)
        self.setCentralWidget(w)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        # self.files = '/Volumes/SanDisk2T/BaFang/Track_data/ltasta/2025-05-19/03/'
        # self.files = '/Volumes/SanDisk2T4/Track_data/agc/2025-05-19/00/'
        self.files = '/Volumes/SanDisk2T4/BaFang/Track_data/agc/2025-05-19/00/'
        parts = Path(self.files).parts
        # 提取倒数两级目录
        self.npz_name  = 'peaks_' + parts[-2].replace("-", "") + parts[-1]# 组合为路径对象
        self.next_select = 0
        
        cfts = read(self.files+'*.sac')
        self.data_all      = np.array([tr.data for tr in cfts])[::-1]
        self.current_start = 1000 * 60 * 0
        self.window_size   = 1000 * 400
        self.default_window_size = self.window_size
        self.scroll_step_s = 30.0
        self.center_mode = "energy_center"
        self.right_click_mode = "fill"  # fill | cut
        self.update_tracking_data()


    def redraw(self):
        self.ax.clear()

        self.st_visual.plot(type='section', scale=3.0, orientation='horizontal', fig=self.fig)
        if self.fig.axes:
            self.ax = self.fig.axes[0]
        # 让图在窗口内尽量铺满
        self.ax.set_xlim(self.tracking.t_axis[0], self.tracking.t_axis[-1])
        self.ax.set_ylim(self.tracking.x_axis[0] * 1e-3, self.tracking.x_axis[-1] * 1e-3)
        self.ax.margins(x=0, y=0)
        # 以全局时间（1小时文件坐标）显示刻度
        t_offset = getattr(self, "t_offset", 0.0)
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x + t_offset:.1f}"))

        # 绘制当前车辆轨迹
        if not np.isnan(self.veh_state).all():
            cols  = np.where(~np.isnan(self.veh_state))[0]
            t_idx_global = self.veh_state[cols].astype(int)
            t_idx_local = t_idx_global - self.current_start
            in_view = (t_idx_local >= 0) & (t_idx_local < len(self.tracking.t_axis))
            if np.any(in_view):
                cols_in = cols[in_view]
                t_idx_local = t_idx_local[in_view]
                ts    = (self.tracking.t_axis[t_idx_local])
                xs    = (self.tracking.x_axis[cols_in] * 1e-3)
                # self.ax.plot(ts, xs, linewidth=1, color='orange', marker='+') # , ls='--'
                self.ax.scatter(ts, xs, marker='+', color='k', s=50)
        # 绘制所有车辆轨迹
        if self.veh_states:
            for veh_state in self.veh_states:
                t_values_global = veh_state[~np.isnan(veh_state)].astype(int)
                x_indices = np.where(~np.isnan(veh_state))[0]
                t_values_local = t_values_global - self.current_start
                in_view = (t_values_local >= 0) & (t_values_local < len(self.tracking.t_axis))
                if np.any(in_view):
                    self.ax.scatter(
                        self.tracking.t_axis[t_values_local[in_view]],
                        self.tracking.x_axis[x_indices[in_view]] * 1e-3,
                        marker='x',
                        s=50
                    )

        self.canvas.draw()

    def adjust_window_size(self, scale):
        new_size = int(self.window_size * scale)
        min_size = 1000  # 最小窗口长度（样点）
        max_size = self.data_all.shape[1]
        new_size = max(min_size, min(new_size, max_size))
        if new_size == self.window_size:
            return
        self.window_size = new_size
        # 确保窗口不越界
        self.current_start = min(self.current_start, max(0, self.data_all.shape[1] - self.window_size))
        self.update_tracking_data(reset_states=False)

    def adjust_window_size_at_cursor(self, scale, cursor_x):
        if cursor_x is None:
            self.adjust_window_size(scale)
            return
        old_size = self.window_size
        new_size = int(self.window_size * scale)
        min_size = 1000  # 最小窗口长度（样点）
        max_size = self.data_all.shape[1]
        new_size = max(min_size, min(new_size, max_size))
        if new_size == old_size:
            return
        # cursor_x 是当前窗口内的秒数（0~window_s）
        t_offset = getattr(self, "t_offset", 0.0)
        cursor_global_s = cursor_x + t_offset
        old_window_s = old_size * self.dt
        rel = 0.0 if old_window_s == 0 else (cursor_x / old_window_s)
        new_start_s = cursor_global_s - rel * (new_size * self.dt)
        new_start = int(round(new_start_s / self.dt))
        new_start = max(0, min(new_start, self.data_all.shape[1] - new_size))
        self.window_size = new_size
        self.current_start = new_start
        self.update_tracking_data(reset_states=False)

    def on_scroll(self, event):
        if event is None:
            return
        step = getattr(event, "step", 0)
        if step == 0:
            # 兼容老版本：scroll up/down 使用 button
            step = 1 if getattr(event, "button", None) == "up" else -1
        step = 1 if step > 0 else -1
        # 优先读取 Qt 修饰键，避免 event.key 在 mac 上为空
        modifiers = None
        if hasattr(event, "guiEvent") and event.guiEvent is not None:
            try:
                modifiers = event.guiEvent.modifiers()
            except Exception:
                modifiers = None

        use_shift = False
        use_cmd = False
        if modifiers is not None:
            use_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
            use_cmd = bool(modifiers & (Qt.KeyboardModifier.MetaModifier | Qt.KeyboardModifier.ControlModifier))
        else:
            key = str(getattr(event, "key", "")).lower()
            use_shift = "shift" in key
            use_cmd = ("control" in key) or ("cmd" in key) or ("meta" in key)

        if use_shift:
            # Shift + 滚轮：缩放
            scale = 0.8 if step > 0 else 1.25
            self.adjust_window_size_at_cursor(scale, event.xdata)
        elif use_cmd:
            # Cmd/Ctrl + 滚轮：缩放（以鼠标位置为中心）
            scale = 0.8 if step > 0 else 1.25
            self.adjust_window_size_at_cursor(scale, event.xdata)
        else:
            # 滚轮：平移
            shift = int(self.scroll_step_samples) * (-step)
            new_start = self.current_start + shift
            new_start = max(0, min(new_start, self.data_all.shape[1] - self.window_size))
            if new_start == self.current_start:
                return
            self.current_start = new_start
            self.update_tracking_data(reset_states=False)

    def on_click(self, event):
        if not event.inaxes:
            return

        clicked_t = event.xdata
        clicked_x = event.ydata * 1e3
        t_approx  = np.argmin(np.abs(self.tracking.t_axis - clicked_t))
        x_idx     = np.argmin(np.abs(self.tracking.x_axis - clicked_x))

        pick = self.tracking.args['detect']
        peaks = find_peaks(
            self.tracking.data[x_idx],
            prominence=pick['prominence'],
            distance=pick['distance'],
            wlen=pick['wlen'],
            height=pick.get('height', None)
        )[0]
        peak_times = self.tracking.t_axis[peaks]
        i_best = np.argmin(np.abs(peak_times - clicked_t))
        t_idx = peaks[i_best] if peaks.size > 0 else t_approx

        # —— 左键：对应点卡尔曼滤波 ——
        if event.button == 1:
            if peaks.size == 0:
                print("⚠️ 该位置无检测到峰值！")
                return
            self.temp_points.append((x_idx, int(t_idx)))
            print(f"🚀 标注点：x_idx={x_idx}, t_idx={t_idx}")
            try:
                new_states, _ = self.tracking.tracking_with_veh_base(
                                        start_x  = self.tracking.x_axis[x_idx],
                                        end_x    = self.tracking.x_axis[-1],
                                        veh_base = np.array([t_idx]),
                                        sigma_a  = 0.0001
                                    )
                
                track_len = new_states.shape[1]
                new_states = interpolate_middle_nans(new_states[0])
                new_states_global = new_states.copy()
                valid = ~np.isnan(new_states_global)
                new_states_global[valid] = new_states_global[valid] + self.current_start
                self.veh_state[x_idx:x_idx+track_len] = new_states_global
            except Exception as e:
                print("❌ 跟踪失败:", e)
            self.redraw()

        # —— 中键：寻找最近峰值并写入当前轨迹 ——
        elif event.button == 2:
            mid_click_pick = {
                "prominence": 0.2,    # 峰值显著性（用于峰值检测）
                "distance"  : 100,    # 相邻峰值间的最小距离
                "wlen"      : 1000,   # 检测窗口长度
                "height"    : 1.0,    # 峰值的最低高度
            }

            peaks = find_peaks(
                self.tracking.data[x_idx],
                prominence = mid_click_pick['prominence'],
                distance   = mid_click_pick['distance'],
                wlen       = mid_click_pick['wlen'],
                height     = mid_click_pick.get('height', None)
            )[0]

            if peaks.size == 0:
                print("⚠️ 该位置无检测到峰值！")
                return

            peak_times = self.tracking.t_axis[peaks]
            i_best = np.argmin(np.abs(peak_times - clicked_t))
            t_idx = int(peaks[i_best])
            self.veh_state[x_idx] = t_idx + self.current_start
            print(f"✅ 中键吸附最近峰值: x_idx={x_idx}, t_idx={t_idx + self.current_start}")
            self.redraw()


        # —— 右键：从veh_state中选择起止点并补全中间轨迹 ——
        elif event.button == 3:
            if self.right_click_mode == "cut":
                self.veh_state[x_idx:] = np.nan
                print(f"✂️ 已裁剪 veh_state，从 x_idx={x_idx} 之后全部清空")
                self.redraw()
                return
            if not hasattr(self, 'right_click_tmp'):
                self.right_click_tmp = []

            self.right_click_tmp.append((x_idx, t_idx))
            print(f"🟠 右键选点 {len(self.right_click_tmp)}：x_idx={x_idx}, t_idx={t_idx}")

            if len(self.right_click_tmp) == 1:      # 第一次右键，选择起点
                # 寻找离点击最近的veh_state点
                dist = np.abs(self.tracking.x_axis - clicked_x)
                idx = np.nanargmin(dist + np.isnan(self.veh_state) * 1e10)
                if np.isnan(self.veh_state[idx]):
                    print("⚠️ 当前无可用veh_state点")
                    self.right_click_tmp = []
                    return
                t0_local = int(self.veh_state[idx]) - self.current_start
                if t0_local < 0 or t0_local >= len(self.tracking.t_axis):
                    print("⚠️ 起点不在当前窗口内")
                    self.right_click_tmp = []
                    return
                self.right_click_tmp[0] = (idx, t0_local)
                print(f"✅ 设置起点为 veh_state 最近点: x_idx={idx}, t_idx={t0_local}")

            elif len(self.right_click_tmp) == 2:    # 第二次右键，选择终点
                x0, t0 = self.right_click_tmp[0]
                x1, t1 = self.right_click_tmp[1]
                if x0 > x1:
                    x0, x1 = x1, x0
                    t0, t1 = t1, t0

                print(f"🧩 补全轨迹区间: x[{x0}:{x1}]")
                for xi in range(x0, x1 + 1):
                    # 初始插值
                    interp_t = t0 + (t1 - t0) * (xi - x0) / max((x1 - x0), 1)

                    # 在附近查找最接近峰值
                    search_range = 20
                    ti_low = max(0, int(interp_t - search_range))
                    ti_high = min(len(self.tracking.t_axis), int(interp_t + search_range))


                    Right_click_pick = {
                        "prominence": 0.2,    # 峰值显著性（用于峰值检测）
                        "distance"  : 100,    # 相邻峰值间的最小距离
                        "wlen"      : 1000,   # 检测窗口长度
                        "height"    : 1.0,    # 峰值的最低高度
                    }

                    peaks = find_peaks(
                        self.tracking.data[xi][ti_low:ti_high],
                        prominence = Right_click_pick['prominence'],
                        distance   = Right_click_pick['distance'],
                        wlen       = Right_click_pick['wlen'],
                        height     = Right_click_pick.get('height', None)
                    )[0]
                    if peaks.size > 0:
                        t_peak_idx = peaks[np.argmin(np.abs(peaks - (interp_t - ti_low)))] + ti_low
                        self.veh_state[xi] = t_peak_idx + self.current_start
                        print(f"✅ Find peak at t_idx={t_peak_idx + self.current_start}")
                    else:
                        print(f"❌ No peak found, use fallback interpolation at t_idx={interp_t}")
                        self.veh_state[xi] = interp_t + self.current_start  # fallback 插值
                self.right_click_tmp = []
                print(f"✅ 已更新veh_state轨迹区间 [{x0}–{x1}]")
                self.redraw()

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key.Key_C:
            self.center_mode = "main_peak" if self.center_mode == "energy_center" else "energy_center"
            print(f"🔄 center_mode = {self.center_mode}")
            self.update_tracking_data(reset_states=False)
            return
        elif key == Qt.Key.Key_E:
            self.right_click_mode = "cut"
            print("✂️ 右键裁剪模式已开启（点击后方全部清空）")
            return
        elif key == Qt.Key.Key_R:
            self.right_click_mode = "fill"
            print("✅ 右键补全模式已恢复")
            return

        if key == Qt.Key.Key_X:
            if not self.temp_points:
                print("❗️ 没有临时点可提交。")
                return

            self.veh_state = np.full((len(self.tracking.x_axis)), np.nan)
            self.temp_points = []
            self.redraw()

        if key == Qt.Key.Key_S:
            if not np.isnan(self.veh_state).all():

                # self.veh_state = fit_and_fill_nans(self.veh_state, deg=2) # 是否自动补全

                self.veh_states.append(self.veh_state.copy())
                print(f"💾 已保存手动轨迹 #{len(self.veh_states)}，点数 {np.sum(~np.isnan(self.veh_state))}")
                self.veh_state = np.full((len(self.tracking.x_axis)), np.nan)
                self.temp_points = []
                self.redraw()
            else:
                print("⚠️ 当前 veh_state 是空的，未保存。")

        elif key == Qt.Key.Key_Z:
            self.veh_state = fit_and_fill_nans(self.veh_state, deg=2)
            self.redraw()

        elif key == Qt.Key.Key_D:
            # 取消 D 写入功能
            print("⚠️ D 键已取消保存功能，请使用 F 键保存并前进。")

        elif key == Qt.Key.Key_Backspace:
            if self.veh_states:
                self.veh_states.pop()
                print("🗑️ 已删除最后一条veh_states轨迹")
                self.redraw()
            else:
                print("⚠️ 没有轨迹可删除")

        elif key == Qt.Key.Key_A:

            self.st_visual_bak = deepcopy(self.st_visual)

            parts = list(Path(self.files).parts)
            parts[parts.index('ltasta')] = 'agc'
            files = str(Path(*parts))
            
            cfts = read(files+'/*.sac')
            data_all = np.array([tr.data[self.current_start:self.current_start+self.window_size] for tr in cfts])[::-1]
            self.st_visual = Stream()
            for i, d in enumerate(data_all):
                tr = Trace(data=d)
                tr.stats.distance  = self.tracking.x_axis[i]
                tr.stats.starttime = self.tracking.t_axis[0] - 1000 * 60 * 15
                tr.stats.delta     = self.tracking.t_axis[1] - self.tracking.t_axis[0]
                self.st_visual.append(tr)
            self.redraw()
        elif key == Qt.Key.Key_Q:
            self.st_visual = self.st_visual_bak
            self.redraw()

        elif key == Qt.Key.Key_F:
            # 先保存当前窗口轨迹
            if self.veh_states:
                save_path = f"{self.npz_name}_{self.next_select:02d}.npz"
                np.savez(save_path, veh_states=self.veh_states)
                print(f"👺 已保存所有轨迹到 {save_path}")
            else:
                print("⚠️ 当前没有轨迹可保存")

            # 然后进入下一个文件
            parts = Path(self.files).parts
            self.files = str(Path(*parts[:-1], f"{int(parts[-1])+1:02d}"))
            parts = Path(self.files).parts
            # 提取倒数两级目录
            self.npz_name  = 'peaks_' + parts[-2].replace("-", "") + parts[-1]# 组合为路径对象
            self.next_select = 0
            
            cfts = read(self.files+'/*.sac')
            self.data_all      = np.array([tr.data for tr in cfts])[::-1]
            self.current_start = 1000 * 60 * 0
            self.window_size   = self.default_window_size
            self.update_tracking_data()




    def update_tracking_data(self, reset_states=True):
        end = self.current_start + self.window_size
        if end > self.data_all.shape[1]:
            print("⚠️ 已经到达数据末尾")
            return

        data = self.data_all[:, self.current_start:end]
        x_axis = np.arange(len(data)) * 100.
        t_axis = np.arange(len(data[0])) * 0.001
        self.t_offset = self.current_start * 0.001
        self.dt = t_axis[1] - t_axis[0]
        self.scroll_step_samples = max(1, int(self.scroll_step_s / self.dt))

        self.tracking = KF_tracking(
            data=data,
            t_axis=t_axis,
            x_axis=x_axis,
            args={
                "detect": {
                    "prominence": 0.4,    # 峰值显著性（用于峰值检测）
                    "distance"  : 500,    # 相邻峰值间的最小距离
                    "wlen"      : 1000,   # 检测窗口长度
                    "height"    : 2.0,    # 峰值的最低高度
                    "center_mode": self.center_mode,  # energy_center | main_peak
                },
                "veh": {
                    "vel_init": 25.0,    # 初始速度
                    "vmin"    : 22.0,    # 速度下限
                    "vmax"    : 28.0,    # 速度上限
                    "tmin"    : -3.0,    # 时间偏移下限
                    "tmax"    : 3.0,     # 时间偏移上限
                    "dt"      : 0.001,   # 时间步长
                    "dx"      : 100,     # 空间步长
                }
            }
        )

        self.st_visual = Stream()
        for i, d in enumerate(self.tracking.data):
            tr = Trace(data=d)
            tr.stats.distance  = self.tracking.x_axis[i]
            tr.stats.starttime = self.tracking.t_axis[0]
            tr.stats.delta     = self.tracking.t_axis[1] - self.tracking.t_axis[0]
            self.st_visual.append(tr)


        # 清空状态（非滚轮/缩放时）
        if reset_states:
            self.manual_states = []
            self.temp_points = []
            self.base_states = []
            self.veh_states = []
            self.veh_state = np.full((len(self.tracking.x_axis)), np.nan)
        self.redraw()



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    gui = TrackingGUI()
    gui.show()
    sys.exit(app.exec())
