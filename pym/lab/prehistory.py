from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from pym.core.engine import PYMEngine
from pym.core.state import PYMState
from pym.physics.pym_forces import memory_force
from pym.physics.standard import harmonic_force
from pym.physics.multimode_bath import BathParameters, MultiModeState, step_velocity_verlet


@dataclass(frozen=True)
class PrepProtocol:
    amplitude: float = 0.5
    frequency: float = 2.0
    phase: float = 0.0

    def force(self, t: float) -> float:
        return self.amplitude * np.sin(self.frequency * t + self.phase)


def prepare_pym(q0, p0, lambda_pym, dt, prep_steps, protocol):
    state = PYMState(
        position=np.asarray(q0, float), momentum=np.asarray(p0, float),
        memory=0.0, lambda_pym=lambda_pym, alpha=float(np.exp(-dt / 1.0)),
    )
    engine = PYMEngine()
    for k in range(prep_steps):
        t = (k - prep_steps) * dt
        u = protocol.force(t)
        def forced_standard(position, momentum, u=u):
            return harmonic_force(position, momentum) + np.full_like(position, u)
        state = engine.step(state, dt, forced_standard, memory_force)
    return state


def prepare_bath(params, q0, p0, dt, prep_steps, protocol):
    n = params.n_modes
    state = MultiModeState(
        np.asarray(q0, float).copy(), np.asarray(p0, float).copy(),
        np.zeros((n, len(q0))), np.zeros((n, len(q0))), 1.0,
    )
    # Same external preparation force acts on the observable particle.
    for k in range(prep_steps):
        t = (k - prep_steps) * dt
        u = protocol.force(t)
        # Velocity-Verlet with the external preparation force added to q-force.
        from pym.physics.multimode_bath import forces
        fq, fy = forces(state, params)
        fq = fq + u
        p_half = state.p + 0.5 * dt * fq
        py_half = state.py + 0.5 * dt * fy
        q_new = state.q + dt * p_half / state.particle_mass
        y_new = state.y + dt * py_half / params.mass[:, None]
        trial = MultiModeState(q_new, p_half, y_new, py_half, state.particle_mass)
        fq2, fy2 = forces(trial, params)
        fq2 = fq2 + protocol.force(t + dt)
        state = MultiModeState(
            q_new, p_half + 0.5 * dt * fq2, y_new,
            py_half + 0.5 * dt * fy2, state.particle_mass,
        )
    return state


def release_pym(state, steps, dt):
    engine = PYMEngine(); qs=[]; ps=[]
    for _ in range(steps):
        state = engine.step(state, dt, harmonic_force, memory_force)
        qs.append(state.position.copy()); ps.append(state.momentum.copy())
    return np.asarray(qs), np.asarray(ps)


def release_bath(state, params, steps, dt):
    qs=[]; ps=[]
    for _ in range(steps):
        state = step_velocity_verlet(state, params, dt)
        qs.append(state.q.copy()); ps.append(state.p.copy())
    return np.asarray(qs), np.asarray(ps)
