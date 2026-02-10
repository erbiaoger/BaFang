
from scipy.signal import find_peaks
import numpy as np
from scipy.stats import norm
from obspy import Stream, Trace
# from modules.tracking_1 import likelihood_1d
import matplotlib.pyplot as plt
from scipy import signal

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
        min_interval = 1800  # 以采样点为单位，避免过密重复车（可以调）

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
                best_peak = self._select_peak(peak_loc, pred_pos, tmin, tmax)
                if best_peak is not None:
                    veh_states[v, i - start_x_idx] = best_peak
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
        min_track_points = 3
        valid_mask = np.sum(~np.isnan(veh_states), axis=1) >= min_track_points
        veh_states = veh_states[valid_mask]
        veh_weak_states = veh_weak_states[valid_mask]


        return veh_states, veh_weak_states







from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TrackingGUI(QMainWindow):
    def __init__(self, tracking_obj, veh_states, veh_weak_states):
        super().__init__()
        self.setWindowTitle("车辆轨迹识别系统")

        self.tracking = tracking_obj
        self.veh_states = veh_states
        self.veh_weak_states = veh_weak_states

        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.redraw_plot()
        self.canvas.mpl_connect("button_press_event", self.on_click)


    def redraw_plot(self):
        self.ax.clear()
        self.tracking.tracking_visualization_one_section(
            start_x=start_x,
            tracked_v=self.veh_states,
            weak_states=None,
            fig=self.fig,
            ax=self.ax
        )
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes:
            clicked_t = event.xdata
            clicked_x = event.ydata * 1e3
            print(f"点击位置: 时间={clicked_t:.2f}s, 位置={clicked_x:.1f}m")

            t_idx = np.argmin(np.abs(self.tracking.t_axis - clicked_t))
            x_idx = np.argmin(np.abs(self.tracking.x_axis - clicked_x))

            new_states, new_weak = self.tracking.tracking_with_veh_base(
                start_x=self.tracking.x_axis[x_idx],
                end_x=end_x,
                veh_base=np.array([t_idx]),
                sigma_a=0.0001
            )


            v_idx = np.argmin(np.abs(self.veh_states[:, x_idx] - clicked_t*1e3))
            
            self.veh_states[v_idx, x_idx:] = new_states.reshape(-1)

            self.redraw_plot()
    


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)




    cfts = np.load('/Volumes/SanDisk2T4/vehicle_track/save_data/agc_20250728_153911.npz')['data']
    data = cfts.T[:, 1000*60*20:1000*60*30]

    x_axis = np.arange(len(data)) * 100.
    t_axis = np.arange(len(data[0])) * 0.001


    tracking_args = {
        "detect":{
                "prominence": 0.3,  # 突起程度
                "distance"  : 500,  # 相邻峰之间的最小水平距离
                "wlen"      : 2000, # 突起窗口长度
                "height"    : 2.0,  # 突起高度
        },
        "veh":{
            "vel_init": 25.0,
            "vmin"    : 22.0,
            "vmax"    : 28.0,
            "tmin"    : -3.0,
            "tmax"    : 3.0,
            "dt"      : 0.001,
            "dx"      : 100,
        }
    }

    start_x, end_x = 0, 5000
    tracking = KF_tracking(data=data, t_axis=t_axis,
                        x_axis=x_axis, 
                        args=tracking_args)

    veh_base = tracking.detect_in_one_section(start_x=start_x, nx=1, sigma=0.0001)
    veh_states, veh_weak_states = tracking.tracking_with_veh_base(start_x=start_x, end_x=end_x,
                                                veh_base=veh_base, sigma_a=0.0001)

    # tracking.tracking_visualization_one_section(start_x, veh_states, trace=None, weak_states=None, fig=None, ax=None)


    gui = TrackingGUI(tracking, veh_states, veh_weak_states)
    gui.show()
    sys.exit(app.exec())