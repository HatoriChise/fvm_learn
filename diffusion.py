import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class OneDHeatSolver:
    """
    一维导热方程求解器类
    方程：∂φ/∂t = α * ∂²φ/∂x²
    """

    def __init__(self, L=1.0, N=100, alpha=1.0, total_time=1.0, dt=0.01):
        """
        初始化求解器参数

        参数:
            L: 空间长度 (m)
            N: 网格数量
            alpha: 热扩散系数 (m²/s)
            total_time: 总模拟时间 (s)
            dt: 时间步长 (s)
        """
        # 设置字体为 SimHei 以正常显示中文标签
        plt.rcParams["font.family"] = [
            "STFangsong",
            "Arial",
            "Helvetica",
            "Times New Roman",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        self.L = L
        self.N = N
        self.alpha = alpha
        self.total_time = total_time
        self.dt = dt

        # 计算空间步长
        self.dx = L / N

        # 检查稳定性条件
        self._check_stability()

        # 初始化网格和时间
        self.x = np.linspace(0, L, N + 1)  # 包括边界点
        self.time_steps = int(total_time / dt)

        # 初始化温度场
        self.temperature = np.zeros((self.time_steps + 1, N + 1))

    def _check_stability(self):
        """检查CFL稳定性条件"""
        stability_condition = self.alpha * self.dt / (self.dx**2)
        if stability_condition > 0.5:
            raise ValueError(
                f"CFL stability condition failed: {stability_condition:.3f} > 0.5"
            )
        else:
            print("CFL stability condition passed.")

    def set_initial_condition(self, initial_func=None):
        """
        设置初始条件

        参数:
            initial_func: 初始温度分布函数，默认为正弦分布
        """
        if initial_func is None:
            # 默认初始条件: 正弦分布
            self.temperature[0, :] = np.sin(np.pi * self.x / self.L)
        else:
            self.temperature[0, :] = initial_func(self.x)

    def set_boundary_conditions(self, left_bc=0.0, right_bc=0.0):
        """
        设置边界条件

        参数:
            left_bc: 左边界温度
            right_bc: 右边界温度
        """
        self.left_bc = left_bc
        self.right_bc = right_bc

        # 设置时间序列上的边界条件
        for n in range(self.time_steps + 1):
            self.temperature[n, 0] = left_bc
            self.temperature[n, -1] = right_bc

    def solve(self):
        """求解一维导热方程"""
        # 计算扩散项系数
        r = self.alpha * self.dt / (self.dx**2)

        # 时间步进
        for n in range(self.time_steps):
            # 更新内部点
            for i in range(1, self.N):
                # 中心差分公式
                self.temperature[n + 1, i] = self.temperature[n, i] + r * (
                    self.temperature[n, i + 1]
                    - 2 * self.temperature[n, i]
                    + self.temperature[n, i - 1]
                )

            # 应用边界条件
            self.temperature[n + 1, 0] = self.left_bc
            self.temperature[n + 1, -1] = self.right_bc

    def get_solution(self, time_index=None):
        """获取指定时间步的解"""
        if time_index is None:
            time_index = self.time_steps  # 返回最终解
        return self.temperature[time_index, :]

    def plot_solution(self, time_indices=None):
        """绘制不同时间步的温度分布"""
        if time_indices is None:
            time_indices = [
                0,
                self.time_steps // 4,
                self.time_steps // 2,
                self.time_steps,
            ]

        plt.figure(figsize=(10, 6))
        for idx in time_indices:
            if idx <= self.time_steps:
                time = idx * self.dt
                plt.plot(self.x, self.temperature[idx, :], label=f"t = {time:.2f}s")

        plt.xlabel("location x (m)")
        plt.ylabel("temperature φ")
        plt.title("1D heat diffusion term numerical solution")
        plt.legend()
        plt.grid(True)
        plt.show()

    def animate_solution(self, interval=50):
        """创建温度分布随时间演化的动画"""
        fig, ax = plt.subplots(figsize=(10, 6))
        (line,) = ax.plot(self.x, self.temperature[0, :])
        ax.set_xlabel("location x (m)")
        ax.set_ylabel("temperature φ")
        ax.set_title("一维导热方程数值解随时间演化")
        ax.set_ylim(0, 1.1)
        ax.grid(True)

        def update(frame):
            line.set_ydata(self.temperature[frame, :])
            ax.set_title(f"1D heat diffusion term (t = {frame*self.dt:.2f}s)")
            return (line,)

        anim = FuncAnimation(
            fig, update, frames=self.time_steps + 1, interval=interval, blit=True
        )
        plt.show()
        return anim


# 使用示例
if __name__ == "__main__":
    # 创建求解器实例
    solver = OneDHeatSolver(L=1.0, N=50, alpha=0.01, total_time=10.0, dt=0.001)

    # 设置初始条件 - 自定义函数
    def initial_condition(x):
        # 在中间区域设置高温区域
        return np.where((x > 0.4) & (x < 0.6), 1.0, 0.0)

    solver.set_initial_condition(initial_condition)

    # 设置边界条件 (两端固定为0)
    solver.set_boundary_conditions(left_bc=0.0, right_bc=0.0)

    # 求解方程
    solver.solve()

    # 绘制结果
    solver.plot_solution(time_indices=[0, 10, 25, 50, 75, 100])

    # 创建动画 (可选，可能在某些环境中需要单独保存)
    solver.animate_solution()

    # 获取最终解
    final_solution = solver.get_solution()
    print(
        f"最终解的统计: 最小值={final_solution.min():.4f}, 最大值={final_solution.max():.4f}"
    )
