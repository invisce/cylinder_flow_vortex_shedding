"""2-D incompressible cylinder-flow solver on a staggered MAC grid."""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


class CylinderFlowSolver:
    def __init__(self, nx=401, ny=101, Lx=8.0, Ly=2.0, D=0.4, Re=100.0,
                 U_inf=1.0, rho=1.0, xc=2.0, wall_type="no-slip"):
        if min(nx, ny) < 9 or min(Lx, Ly, D, Re, U_inf, rho) <= 0:
            raise ValueError("Positive physical parameters and >=9 grid points are required")
        if wall_type not in ("no-slip", "slip"):
            raise ValueError("wall_type must be 'no-slip' or 'slip'")
        if not (D < Ly and D / 2 < xc < Lx - D / 2):
            raise ValueError("Cylinder must lie fully inside the channel")

        self.nx, self.ny = nx - 1, ny - 1
        self.Lx, self.Ly, self.D = Lx, Ly, D
        self.dx, self.dy = Lx / self.nx, Ly / self.ny
        self.Re, self.U_inf, self.rho, self.xc = Re, U_inf, rho, xc
        self.yc, self.nu, self.mu = Ly / 2, U_inf * D / Re, rho * U_inf * D / Re
        self.wall_type = wall_type
        self.x = (np.arange(self.nx) + .5) * self.dx
        self.y = (np.arange(self.ny) + .5) * self.dy
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.mask = (self.X - xc) ** 2 + (self.Y - self.yc) ** 2 <= (D / 2) ** 2
        if not np.any(self.mask) or np.any(self.mask[:, (0, -1)]):
            raise ValueError("Cylinder is unresolved or intersects inlet/outlet cells")

        fluid = ~self.mask
        self.u_open = np.zeros((self.ny, self.nx + 1), dtype=bool)
        self.v_open = np.zeros((self.ny + 1, self.nx), dtype=bool)
        self.u_open[:, 1:-1] = fluid[:, :-1] & fluid[:, 1:]
        self.u_open[:, 0] = fluid[:, 0]
        self.u_open[:, -1] = fluid[:, -1]
        self.v_open[1:-1, :] = fluid[:-1, :] & fluid[1:, :]
        self._make_pressure_solver()
        self.q = .5 * rho * U_inf ** 2 * D
        self.probe_i = int(np.clip(np.searchsorted(self.x, xc + 1.5 * D), 0, self.nx - 1))
        self.probe_j = int(np.clip(np.searchsorted(self.y, self.yc + .1), 0, self.ny - 1))
        if self.mask[self.probe_j, self.probe_i]:
            raise ValueError("Wake probe lies inside the cylinder")
        self.reset()

    def _make_pressure_solver(self):
        """Assemble the pressure operator with zero-gradient walls and p=0 at the outlet."""
        ny, nx = self.ny, self.nx
        ids = np.full((ny, nx), -1, dtype=np.int32)
        ids[~self.mask] = np.arange(np.count_nonzero(~self.mask))
        self.fluid_ids = ids
        rows, cols, vals = [], [], []
        ax, ay = 1 / self.dx ** 2, 1 / self.dy ** 2
        for j, i in zip(*np.where(~self.mask)):
            k = int(ids[j, i])
            diag = 0.0
            for jj, ii, coeff in ((j, i - 1, ax), (j, i + 1, ax),
                                   (j - 1, i, ay), (j + 1, i, ay)):
                if ii == nx:
                    diag += 2 * coeff
                elif 0 <= jj < ny and 0 <= ii < nx and ids[jj, ii] >= 0:
                    diag += coeff
                    rows.append(k); cols.append(int(ids[jj, ii])); vals.append(-coeff)
            rows.append(k); cols.append(k); vals.append(diag)
        self.pressure_matrix = coo_matrix(
            (vals, (rows, cols)), shape=(ids.max() + 1, ids.max() + 1)
        ).tocsc()
        self.pressure_lu = splu(self.pressure_matrix)

    def reset(self):
        self.u = self.U_inf * self.u_open.astype(float)
        self.v = np.zeros_like(self.v_open, dtype=float)
        self.p = np.zeros_like(self.mask, dtype=float)

    def cell_velocity(self):
        return .5 * (self.u[:, :-1] + self.u[:, 1:]), .5 * (self.v[:-1] + self.v[1:])

    def divergence(self, u=None, v=None):
        if u is None:
            u, v = self.u, self.v
        return (u[:, 1:] - u[:, :-1]) / self.dx + (v[1:] - v[:-1]) / self.dy

    def _predict(self, dt):
        u, v = self.u, self.v
        us, vs = u.copy(), v.copy()

        c = u[:, 1:-1]
        north = np.empty_like(c); south = np.empty_like(c)
        north[:-1] = np.where(self.u_open[1:, 1:-1], u[1:, 1:-1], -c[:-1])
        south[1:] = np.where(self.u_open[:-1, 1:-1], u[:-1, 1:-1], -c[1:])
        wall_sign = 1 if self.wall_type == "slip" else -1
        north[-1] = wall_sign * c[-1]
        south[0] = wall_sign * c[0]
        across_v = .25 * (v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:])
        ux = (u[:, 2:] - u[:, :-2]) / (2 * self.dx)
        uy = (north - south) / (2 * self.dy)
        lap_u = (u[:, 2:] - 2*c + u[:, :-2]) / self.dx**2 + (north - 2*c + south) / self.dy**2
        us[:, 1:-1] = np.where(self.u_open[:, 1:-1],
                                c + dt * (-c * ux - across_v * uy + self.nu * lap_u), 0.0)

        c = v[1:-1]
        east = np.empty_like(c); west = np.empty_like(c)
        east[:, :-1] = np.where(self.v_open[1:-1, 1:], v[1:-1, 1:], -c[:, :-1])
        west[:, 1:] = np.where(self.v_open[1:-1, :-1], v[1:-1, :-1], -c[:, 1:])
        east[:, -1] = c[:, -1]
        west[:, 0] = -c[:, 0]
        across_u = .25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])
        vx = (east - west) / (2 * self.dx)
        vy = (v[2:] - v[:-2]) / (2 * self.dy)
        lap_v = (east - 2*c + west) / self.dx**2 + (v[2:] - 2*c + v[:-2]) / self.dy**2
        vs[1:-1] = np.where(self.v_open[1:-1],
                             c + dt * (-across_u * vx - c * vy + self.nu * lap_v), 0.0)
        us[:, 0] = self.U_inf
        us[:, -1] = us[:, -2]
        vs[0] = 0.0; vs[-1] = 0.0
        return us, vs

    def step(self, dt):
        us, vs = self._predict(dt)
        rhs = -self.rho / dt * self.divergence(us, vs)[~self.mask]
        sol = self.pressure_lu.solve(rhs)
        self.p.fill(0.0)
        self.p[~self.mask] = sol
        a = dt / self.rho

        self.u[:, 1:-1] = np.where(
            self.u_open[:, 1:-1],
            us[:, 1:-1] - a * (self.p[:, 1:] - self.p[:, :-1]) / self.dx, 0.0
        )
        self.u[:, 0] = self.U_inf
        self.u[:, -1] = us[:, -1] + a * 2 * self.p[:, -1] / self.dx
        self.v[1:-1] = np.where(
            self.v_open[1:-1],
            vs[1:-1] - a * (self.p[1:] - self.p[:-1]) / self.dy, 0.0
        )
        self.v[0] = 0.0; self.v[-1] = 0.0
        div = self.divergence()
        max_div = float(np.max(np.abs(div[~self.mask])))
        pres_res = float(np.max(np.abs(self.pressure_matrix @ sol - rhs)) /
                         max(np.max(np.abs(rhs)), 1.0))
        return max_div, pres_res

    def forces(self):
        """Estimate pressure and viscous forces on the staircase boundary."""
        uc, vc = self.cell_velocity()
        f, dx, dy, mu = ~self.mask, self.dx, self.dy, self.mu
        left = np.zeros_like(f); right = np.zeros_like(f)
        below = np.zeros_like(f); above = np.zeros_like(f)
        left[:, :-1] = f[:, :-1] & self.mask[:, 1:]
        right[:, 1:] = f[:, 1:] & self.mask[:, :-1]
        below[:-1] = f[:-1] & self.mask[1:]
        above[1:] = f[1:] & self.mask[:-1]
        fx_p = dy * (self.p[left].sum() - self.p[right].sum())
        fy_p = dx * (self.p[below].sum() - self.p[above].sum())
        fx_s = 2 * mu * dx / dy * (uc[below].sum() + uc[above].sum())
        fy_s = 2 * mu * dy / dx * (vc[left].sum() + vc[right].sum())
        return (fx_p + fx_s) / self.q, (fy_p + fy_s) / self.q

    def report_stability(self, dt):
        diff = .5 / (self.nu * (1 / self.dx**2 + 1 / self.dy**2))
        print(f"Re_D={self.Re:g}, blockage D/Ly={self.D / self.Ly:.3f}, "
              f"dx={self.dx:.4f}, dy={self.dy:.4f}, dt={dt:.4f}")
        print(f"Initial advective CFL ~ {dt * self.U_inf / min(self.dx, self.dy):.3f}; "
              f"explicit diffusion bound dt <= {diff:.4f}")
        if dt >= diff:
            raise ValueError("Time step exceeds the explicit diffusion stability bound")

    def solve(self, nsteps=25000, dt=.002, trigger=True, save_every=50, verbose=True):
        if nsteps < 1 or dt <= 0 or save_every < 0:
            raise ValueError("nsteps, dt must be positive and save_every nonnegative")
        self.reset()
        self.report_stability(dt)
        self.dt, self.nsteps = dt, nsteps
        self.probe_v = np.empty(nsteps)
        self.Cd_hist = np.empty(nsteps)
        self.Cl_hist = np.empty(nsteps)
        self.div_hist = np.empty(nsteps)
        self.p_res_hist = np.empty(nsteps)
        self.time_arr = (np.arange(nsteps) + 1) * dt
        self.frames, self.frame_t = [], []
        t0 = time.time()
        if trigger:
            x0, y0 = self.xc + 1.5*self.D, self.yc + .1
            px = np.arange(self.nx)[None, :]
            py = np.arange(self.ny + 1)[:, None]
            i0, j0 = int(x0 / self.dx), int(y0 / self.dy)
            perturb = (np.abs(px-i0) <= 2) & (np.abs(py-j0) <= 2)
            self.v[perturb & self.v_open] = .05 * self.U_inf
        print(f"Starting MAC projection: {nsteps} steps, t_end={nsteps*dt:g}")
        for s in range(nsteps):
            div, pres = self.step(dt)
            self.div_hist[s], self.p_res_hist[s] = div, pres
            self.probe_v[s] = .5*(self.v[self.probe_j, self.probe_i] +
                                   self.v[self.probe_j+1, self.probe_i])
            self.Cd_hist[s], self.Cl_hist[s] = self.forces()
            if save_every and s % save_every == 0:
                self.frames.append(self.vorticity().astype(np.float32))
                self.frame_t.append(self.time_arr[s])
            if verbose and (s % max(1, min(2000, nsteps // 5)) == 0 or s == nsteps-1):
                print(f"step {s+1:6d} t={self.time_arr[s]:7.3f} "
                      f"Cd={self.Cd_hist[s]:8.3f} Cl={self.Cl_hist[s]:+8.3f} "
                      f"Div_Linf={div:.2e} P_relres={pres:.2e} "
                      f"wall={time.time()-t0:.1f}s")
            if not np.isfinite(div) or not np.isfinite(self.Cd_hist[s]):
                raise RuntimeError(f"Numerical blow-up at step {s+1}; reduce dt or inspect BCs")
            if div > max(1e-9, 1e-7 * self.U_inf / self.D) or pres > 1e-8:
                raise RuntimeError(f"Pressure projection failed at step {s+1}: "
                                   f"max divergence={div:.2e}, relative residual={pres:.2e}")
        return dict(probe_v=self.probe_v, Cd=self.Cd_hist, Cl=self.Cl_hist,
                    time=self.time_arr, div_linf=self.div_hist, p_relres=self.p_res_hist)

    def vorticity(self):
        uc, vc = self.cell_velocity()
        vort = np.gradient(vc, self.dx, axis=1) - np.gradient(uc, self.dy, axis=0)
        vort[self.mask] = np.nan
        return vort

    def _spectrum(self, signal):
        signal = signal - signal.mean()
        win = np.hanning(len(signal))
        amp = 2 * np.abs(np.fft.rfft(win * signal)) / max(win.sum(), 1)
        freq = np.fft.rfftfreq(len(signal), self.dt)
        if len(amp) < 3:
            return freq, amp, np.nan
        peak = 1 + int(np.argmax(amp[1:]))
        f = freq[peak]
        if 1 <= peak < len(amp) - 1 and np.all(amp[peak-1:peak+2] > 0):
            a, b, c = np.log(amp[peak-1:peak+2])
            offset = .5 * (a-c) / (a-2*b+c) if (a-2*b+c) != 0 else 0
            f += np.clip(offset, -.5, .5) * (freq[1]-freq[0])
        return freq, amp, f

    def validate_strouhal(self, tail_fraction=.4):
        if not hasattr(self, "probe_v"):
            raise ValueError("Call solve() first")
        if not (0 < tail_fraction <= 1):
            raise ValueError("tail_fraction must lie between zero and one")
        start = int((1-tail_fraction) * self.nsteps)
        t = self.time_arr[start:]
        cl = self.Cl_hist[start:]
        if len(cl) < 8:
            raise ValueError("Too few samples in analysis window")
        centered = cl - cl.mean()
        up = np.flatnonzero((centered[:-1] <= 0) & (centered[1:] > 0))
        crossings = t[up] + self.dt * (-centered[up]) / (centered[up+1]-centered[up])
        periods = np.diff(crossings)
        period = periods.mean() if len(periods) >= 3 else np.nan
        st_zc = self.D / (period * self.U_inf)
        freqs, amp, peak = self._spectrum(cl)
        st_fft = peak * self.D / self.U_inf
        regular = len(periods) >= 3 and periods.std()/period < .15
        shedding = bool(regular and cl.std() > .005 and
                        abs(st_zc-st_fft)/st_zc < .1)
        result = dict(sheds=shedding, St_zc=st_zc, St_fft=st_fft, f_fft=peak,
                      mean_Cd=self.Cd_hist[start:].mean(),
                      Cl_amp=.5 * np.ptp(cl), div_linf=np.max(self.div_hist[start:]),
                      f_resolution=1/(len(cl)*self.dt), cycles=len(periods))
        print(f"Analysis window: {t[0]:.2f}..{t[-1]:.2f}s, "
              f"FFT bin width={result['f_resolution']:.4f} Hz, "
              f"blockage={self.D/self.Ly:.3f}")
        print(f"Periodic shedding={shedding}, cycles={len(periods)}; "
              f"St(zero-cross)={st_zc:.4f}, St(FFT)={st_fft:.4f}")
        print(f"Post-transient <Cd>={result['mean_Cd']:.4f}, "
              f"Cl amplitude={result['Cl_amp']:.4f}, "
              f"max |div|={result['div_linf']:.2e}")
        print("Two frequency estimators agree internally; an external benchmark "
              "with matching blockage, grid study and force verification are still needed.")
        return result

    def plot_diagnostics(self, save_path="cfd_validation_dashboard.png", tail_fraction=.4):
        if not hasattr(self, "probe_v"):
            raise ValueError("Call solve() first")
        start = int((1-tail_fraction) * self.nsteps)
        freqs, amp, peak = self._spectrum(self.Cl_hist[start:])
        vort = self.vorticity()
        vmax = max(float(np.nanpercentile(np.abs(vort), 98)), 1e-10)
        fig, axs = plt.subplots(2, 2, figsize=(14, 8), dpi=130)
        m = axs[0, 0].contourf(self.X/self.D, self.Y/self.D, vort,
                                levels=np.linspace(-vmax, vmax, 41),
                                cmap="RdBu_r", extend="both")
        axs[0, 0].add_patch(plt.Circle((self.xc/self.D, self.yc/self.D),
                                        .5, color="dimgrey", ec="black"))
        axs[0, 0].set(aspect="equal", xlabel="x / D", ylabel="y / D",
                      title=f"Vorticity | Re={self.Re:g}, blockage={self.D/self.Ly:.2f}")
        fig.colorbar(m, ax=axs[0, 0], fraction=.03, pad=.04, label="Vorticity (1/s)")
        axs[0, 1].plot(self.time_arr, self.probe_v, lw=1)
        axs[0, 1].set(xlabel="Time (s)", ylabel="Transverse velocity",
                      title="Wake probe v(t)")
        axs[1, 0].plot(self.time_arr[start:], self.Cl_hist[start:], label="Cl")
        axs[1, 0].plot(self.time_arr[start:], self.Cd_hist[start:], label="Cd")
        axs[1, 0].set(xlabel="Time (s)", ylabel="Coefficient",
                      title="Staircase-approximate forces (after transient)")
        axs[1, 0].legend()
        axs[1, 1].plot(freqs, amp, color="crimson")
        if np.isfinite(peak):
            axs[1, 1].axvline(peak, color="black", ls="--",
                              label=f"f={peak:.3f} Hz, St={peak*self.D/self.U_inf:.3f}")
        axs[1, 1].set(xlim=(0, min(2, freqs[-1])), xlabel="Frequency (Hz)",
                      ylabel="Windowed amplitude", title="Lift FFT amplitude spectrum")
        axs[1, 1].legend()
        for ax in axs.flat:
            ax.grid(alpha=.25)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved dashboard: {save_path}")
        plt.close(fig)

    def animate_wake(self, filename="vortex_street.gif", fps=15, max_frames=180):
        from matplotlib import animation
        if not self.frames:
            raise ValueError("Call solve(save_every>0) first")
        indices = np.unique(np.linspace(0, len(self.frames)-1,
                                        min(len(self.frames), max_frames)).astype(int))
        vmax = max(float(np.nanpercentile(np.abs(self.frames[-1]), 98)), 1e-10)
        fig, ax = plt.subplots(figsize=(11, 3.2), dpi=100)

        def draw(k):
            ax.clear()
            idx = indices[k]
            ax.contourf(self.X/self.D, self.Y/self.D, self.frames[idx],
                        levels=np.linspace(-vmax, vmax, 41), cmap="RdBu_r", extend="both")
            ax.add_patch(plt.Circle((self.xc/self.D, self.yc/self.D), .5,
                                    color="dimgrey", ec="black"))
            ax.set(aspect="equal", xlabel="x / D", ylabel="y / D",
                   title=f"Vorticity | Re={self.Re:g}, t={self.frame_t[idx]:.2f}s")
        anim = animation.FuncAnimation(fig, draw, frames=len(indices),
                                       interval=1000/fps)
        anim.save(filename, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        print(f"Saved animation: {filename}")


if __name__ == "__main__":
    solver = CylinderFlowSolver()
    solver.solve(nsteps=25000, dt=.002, save_every=50)
    solver.validate_strouhal()
    solver.plot_diagnostics()
    solver.animate_wake()
