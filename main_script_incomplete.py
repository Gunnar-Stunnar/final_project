import os
import sys
import time
import collections
from echoState import EchoStateNetwork

# Real-time error plot — import with a fallback so the sim still runs if Tk is absent.
try:
    import matplotlib
    matplotlib.use("TkAgg")          # macOS-friendly; falls back automatically
    import matplotlib.pyplot as plt
    _MPLOT_OK = True
except Exception:
    _MPLOT_OK = False

# Simbody (macOS) searches each PATH entry D for D/simbody-visualizer.app/Contents/MacOS/simbody-visualizer.
# So PATH must list the folder that *contains* simbody-visualizer.app (e.g. .../out/build/default).
# Putting .../Contents/MacOS on PATH makes Simbody look for .../MacOS/simbody-visualizer.app/... (wrong).
#
# Option A: export PATH="/path/to/parent/of/simbody-visualizer.app:$PATH"
# Option B: SIMBODY_VISUALIZER_APP=/path/to/simbody-visualizer.app  (we prepend parent of the .app)
# Option C: SIMBODY_VISUALIZER_SEARCH_ROOT=/path/to/parent/folder
# Option D: set DEFAULT_SIMBODY_VISUALIZER_APP below to the .app path

# Example: "/Users/.../simbody-Simbody-3.8/out/build/default/simbody-visualizer.app"
DEFAULT_SIMBODY_VISUALIZER_APP = "/Users/gunnarenserro/simbody-Simbody-3.8/out/build/default/simbody-visualizer.app"


def _path_parent_containing_simbody_visualizer_app(path: str) -> str | None:
	"""Directory that must be on PATH: parent of simbody-visualizer.app."""
	if not path:
		return None
	p = os.path.normpath(os.path.expanduser(path))
	if os.path.basename(p) == "simbody-visualizer.app":
		return os.path.dirname(p)
	cur = p
	for _ in range(24):
		if os.path.basename(cur) == "simbody-visualizer.app":
			return os.path.dirname(cur)
		parent = os.path.dirname(cur)
		if parent == cur:
			break
		cur = parent
	if os.path.isdir(os.path.join(p, "simbody-visualizer.app")):
		return p
	return None


def _ensure_simbody_visualizer_on_path() -> None:
	search_root = os.environ.get("SIMBODY_VISUALIZER_SEARCH_ROOT", "").strip()
	app = os.environ.get("SIMBODY_VISUALIZER_APP", "").strip() or DEFAULT_SIMBODY_VISUALIZER_APP.strip()
	legacy_bindir = os.environ.get("SIMBODY_VISUALIZER_BINDIR", "").strip()

	parent: str | None = None
	if search_root:
		parent = os.path.normpath(os.path.expanduser(search_root))
	elif legacy_bindir:
		parent = _path_parent_containing_simbody_visualizer_app(legacy_bindir)
		if parent is None:
			parent = os.path.normpath(os.path.expanduser(legacy_bindir))
	elif app:
		parent = _path_parent_containing_simbody_visualizer_app(app)

	if parent:
		os.environ["PATH"] = parent + os.pathsep + os.environ.get("PATH", "")


_ensure_simbody_visualizer_on_path()

import opensim as osim
import math
import numpy as np

model_filename = 'ue_torque.osim'

# --- Live demo / visualization ---
# Requires a working `simbody-visualizer` binary (bundled with many OpenSim installs; conda envs may omit it).
# If initSystem fails to spawn it, the script falls back to headless and sets USE_VISUALIZER False.
# Set True to try the Simbody window, or run with: OPENSIM_USE_VISUALIZER=1 python main_script_incomplete.py
TRY_VISUALIZER = True
_viz_env = os.environ.get("OPENSIM_USE_VISUALIZER", "").strip().lower()
if _viz_env in ("1", "true", "yes"):
	USE_VISUALIZER_REQUESTED = True
elif _viz_env in ("0", "false", "no"):
	USE_VISUALIZER_REQUESTED = False
else:
	USE_VISUALIZER_REQUESTED = TRY_VISUALIZER

USE_VISUALIZER = False  # set True below only when initSystem succeeds with the visualizer

# Reach-based targets: arm blends toward IK goal; when the hand is close enough to the target, a new random target is chosen.
MOVE_BLEND_DURATION_S = 0.9  # nominal time for cubic blend toward current goal (s)
REACH_TOL_M = 0.09  # hand-to-target distance (m) to count as "reached"
MIN_TIME_BEFORE_REACH_S = 0.12  # avoid instant retarget at segment start
BALL_RADIUS_M = 0.035  # red target sphere radius for API visualizer
# Ghost stick-figure: how far the "lag ghost" trails the real arm (seconds).
GHOST_LAG_SECONDS = 0.35
GHOST_SPHERE_RADIUS_M = 0.018  # joint-marker sphere radius
GHOST_COLOR = (0.0, 0.9, 0.9)  # cyan
GHOST_OPACITY = 0.35
# Simbody Ground mobilized body index for fixed-world decorations
GROUND_MOBOD_IX = 0
# ── ESN forward predictive model ──────────────────────────────────────────────
# Input per step : q[2] + qd[2] + qdd[2] + tau[2] + q_goal[2]  = 10 features
# Output          : predicted q_next[2] + qd_next[2]             =  4 features
ESN_N_RESERVOIR   = 1_000         # reservoir nodes
ESN_WASHOUT_STEPS = 100          # skip updates while reservoir settles
# ── Online RLS (Recursive Least Squares) parameters ───────────────────────────
# W_out is updated after EVERY step via the Sherman-Morrison rank-1 update on
# P = (ΦᵀΦ)⁻¹, giving the exact optimal readout at all times with no batch solve.
ESN_RLS_LAMBDA    = 0.999   # forgetting factor  (1 = no forgetting, <1 = exponential decay)
ESN_RLS_DELTA     = 1e4     # initial P = delta·I  (high → fast early learning)
ESN_RLS_WARMUP    = 200     # show ghost at real arm for this many RLS steps before trusting W_out
# Per-target washout: skip RLS updates for this many steps after a new target so
# the high-velocity transition spike doesn't corrupt P.
ESN_TARGET_WASHOUT_STEPS  = 30     # ~0.15 s at dt=0.005 s
ESN_LEARN_STOP_REACHES    = 10    # freeze W_out / P after this many successful reaches
# Threshold-gated learning: RLS update only fires when ||Δq|| exceeds this value.
# Below the threshold the model runs pure inference (no weight change).
# Maps to inferior-olive / climbing-fiber signal in the cerebellum.
ESN_ERROR_LEARN_THRESH    = 0.05  # rad (~3°) — tune lower to learn more, higher for less
# Plot update rate (green milestone lines are now replaced by convergence line only).
ESN_PLOT_INTERVAL = 40            # redraw error plot every N steps
# Convergence: mark a milestone once EMA‖Δq‖ stays below threshold for enough steps.
ESN_CONVERGENCE_Q_THRESH = 0.03   # rad (~1.7°)
ESN_CONVERGENCE_WINDOW   = 400    # consecutive steps below threshold
ESN_EMA_ALPHA_CONV       = 0.05   # EMA smoothing for convergence detection
# Ghost arm colour when showing ESN predictions (blue).
GHOST_COLOR = (0.15, 0.45, 1.0)
# Maximum allowed deviation (rad) between predicted q and actual q.
# Clamps wild FK inputs when W_out is still near-zero early in training.
# Keep targets in an "easier" region so the arm doesn't have to reach extreme high poses.
# Bounds are in Ground coordinates (meters).
TARGET_X_RANGE_M = (0.05, 0.26)
TARGET_Y_RANGE_M = (-0.25, 0.25)
TARGET_Z_RANGE_M = (0.15, 0.65)
TARGET_SAMPLE_MAX_TRIES = 200
MIN_TARGET_SEPARATION_M = 0.12  # keep ball from respawning too close to current hand

# CoordinateActuator: actual torque = control_value × optimal_force.
# Keep optimal_force=1.0 so control_value IS the torque in Nm — no hidden scaling.
COORD_ACTUATOR_OPTIMAL_FORCE = {
	"shoulder_elv": 1.0,
	"elbow_flexion": 1.0,
}
MAX_SIM_TIME_S = 45.0
RNG_SEED = 1

# Hard torque limits (Nm). Prevents numerical blow-up.
# shoulder optimal_force=1.0 → control IS torque in Nm.
# elbow optimal_force overridden to 1.0 → same.
TAU_LIMIT = {
	"shoulder_elv": 60.0,
	"elbow_flexion": 30.0,
}

# If True and the visualizer is enabled, throttle the integration loop to real time
# so the animation doesn't finish instantly (and doesn't look "stuck").
REALTIME_WHEN_VISUALIZING = True

# Print a progress line every N seconds of simulated time (helps distinguish "running" vs "hung").
PROGRESS_EVERY_SIM_S = 1.0


def _attach_prescribed_torque_controller(model: osim.Model) -> None:
	muscleController = osim.PrescribedController()
	torqueSet = model.updActuators()
	# Scale coordinate actuator strengths so tracking can lift to targets.
	for name, opt_f in COORD_ACTUATOR_OPTIMAL_FORCE.items():
		try:
			act = osim.CoordinateActuator.safeDownCast(torqueSet.get(name))
			if act:
				act.setOptimalForce(float(opt_f))
		except Exception:
			pass
	for j in range(torqueSet.getSize()):
		func = osim.Constant(0.0)
		muscleController.addActuator(torqueSet.get(j))
		actu = torqueSet.get(j)
		muscleController.prescribeControlForActuator(actu.getName(), func)
	model.addController(muscleController)


def _load_model_init_system(request_visualizer: bool) -> tuple[osim.Model, osim.State]:
	"""
	Load model, attach controller, initSystem. On macOS/Linux conda installs, simbody-visualizer may be missing;
	if so, caller should retry with request_visualizer=False.
	"""
	model = osim.Model(model_filename)
	_attach_prescribed_torque_controller(model)
	if request_visualizer:
		model.setUseVisualizer(True)
	state = model.initSystem()
	return model, state


def _visualizer_init_failed(err: RuntimeError) -> bool:
	msg = str(err)
	return "simbody-visualizer" in msg or "VisualizerProtocol" in msg


try:
	model, state = _load_model_init_system(USE_VISUALIZER_REQUESTED)
	USE_VISUALIZER = USE_VISUALIZER_REQUESTED
except RuntimeError as exc:
	if not USE_VISUALIZER_REQUESTED:
		raise
	# Any failure with the visualizer on (including generic std::exception) — retry once headless.
	hint = ""
	if _visualizer_init_failed(exc):
		hint = " (simbody-visualizer missing or not on PATH?)"
	print(f"[OpenSim] initSystem failed with visualizer{hint} Retrying headless. First error: {exc}", file=sys.stderr)
	try:
		model, state = _load_model_init_system(False)
		USE_VISUALIZER = False
	except RuntimeError as exc2:
		print(f"[OpenSim] Headless initSystem also failed: {exc2}", file=sys.stderr)
		raise exc2 from exc

stepsize = 0.005


# ── Simulation step log (populated by on_simulation_step, saved at end) ──────
_step_log: list[dict] = []

# ── Ghost lag buffer (kept for reference, ESN ghost replaces its use) ─────────
_lag_buffer: collections.deque = collections.deque()



def on_simulation_step(t: float, state_snapshot: dict, torque_commands: dict) -> dict:
	"""
	Called every simulation timestep (stepsize s).

	Parameters
	----------
	t : float
	    Current simulation time (seconds).
	state_snapshot : dict
	    "q"                        – joint positions  {coord_name: rad}
	    "qd"                       – joint velocities {coord_name: rad/s}
	    "qdd"                      – joint accelerations {coord_name: rad/s²}
	    "q_goal"                   – PID setpoint joint angles {coord_name: rad}
	    "integral_err"             – PID integral accumulator {coord_name: rad·s}
	    "end_effector_pos_ground"  – hand XYZ in Ground frame (m),  np.ndarray
	    "shoulder_pos_ground"      – GH joint XYZ in Ground frame (m), np.ndarray
	    "elbow_pos_ground"         – elbow joint XYZ in Ground frame (m), np.ndarray
	    "target_pos_ground"        – active target ball XYZ (m),    np.ndarray
	    "dist_to_target_m"         – Euclidean hand-to-target (m)
	    "targets_completed"        – targets reached so far (int)
	torque_commands : dict
	    "shoulder_elv_torque"  – torque at shoulder elevation (Nm)
	    "elbow_flexion_torque" – torque at elbow flexion (Nm)

	Returns
	-------
	dict with keys:
	    "ghost_shoulder", "ghost_elbow", "ghost_hand"  – np.ndarray[3] positions
	    that drive the blue ESN-predicted ghost stick figure.
	"""
	global _esn_r, _esn_r_at_pred, _esn_last_pred, _esn_step_count

	q      = state_snapshot["q"]
	qd     = state_snapshot["qd"]
	qdd    = state_snapshot.get("qdd", {cn: 0.0 for cn in coord_names})
	q_goal = state_snapshot["q_goal"]
	tau    = {cn: torque_commands.get(f"{cn}_torque", 0.0) for cn in coord_names}
	n_coords = len(coord_names)

	# ── Cerebellum forward model ──────────────────────────────────────────────
	# Input: current real state (q, qd, qdd) + efference copy (tau) + goal.
	# Output: one-step-ahead prediction of next (q, qd).
	# The prediction is the ghost arm position — no coupling to the real arm
	# except when W_out is re-solved (climbing-fiber correction, green lines).
	u      = _build_esn_input_vec(q, qd, qdd, tau, q_goal)
	_esn_r = _esn.step(_esn_r, u)
	_esn_r_at_pred = _esn_r.copy()
	_esn_last_pred = _esn._readout(_esn_r).copy()
	_esn_step_count += 1

	# ── Ghost arm FK ──────────────────────────────────────────────────────────
	# Before the first W_out solve (buffer still filling) W_out ≈ 0 so pred ≈ 0.
	# Show ghost at real arm until training has begun so the figure is sensible.
	q_actual_arr = np.array([q.get(cn, 0.0) for cn in coord_names])
	if _esn_rls_n < ESN_RLS_WARMUP or _esn_target_washout_rem > 0:
		# Not yet trained, OR reservoir is settling into new target context —
		# pin ghost to real arm so there is no wild jump in the visualizer.
		pred_q_arr = q_actual_arr
	else:
		pred_q_arr = _esn_last_pred[:n_coords]

	ghost_shoulder, ghost_elbow, ghost_hand = _esn_fk(pred_q_arr)

	return {
		"ghost_shoulder": ghost_shoulder,
		"ghost_elbow":    ghost_elbow,
		"ghost_hand":     ghost_hand,
	}


def after_simulation_step(
	t_prev: float,
	t_now: float,
	prev_state_snapshot: dict,
	prev_torque_commands: dict,
	current_state_snapshot: dict,
) -> None:
	"""
	Called immediately after each integration step (t_prev → t_now).

	Compares the ESN's one-step-ahead prediction (made in on_simulation_step at
	t_prev) against the actual observed state at t_now, then applies a delta-rule
	weight update to reduce the error.

	Parameters
	----------
	t_prev               : simulation time at start of step
	t_now                : simulation time at end of step (= t_prev + stepsize)
	prev_state_snapshot  : full state dict at t_prev
	prev_torque_commands : torques applied during this step (Nm)
	current_state_snapshot : observed ground-truth state at t_now
	    "q"   – {coord_name: rad}
	    "qd"  – {coord_name: rad/s}
	    "end_effector_pos_ground" – np.ndarray
	"""
	global _esn_frozen, _esn_conv_count, _esn_err_ema_q, _esn_P, _esn_rls_n, _esn_target_washout_rem, _esn_learning_stopped

	# Skip everything during the initial washout (reservoir settling period).
	if _esn_step_count <= ESN_WASHOUT_STEPS:
		return

	# Ground-truth next state
	q_now  = current_state_snapshot["q"]
	qd_now = current_state_snapshot["qd"]
	actual = _build_esn_target_vec(q_now, qd_now)   # shape (4,)

	# Prediction error: positive → ESN undershot, negative → overshot.
	error = actual - _esn_last_pred                  # shape (4,)

	n          = len(coord_names)
	q_err_mag  = float(np.linalg.norm(error[:n]))                      # ||Δq||  (rad)
	qd_err_mag = float(np.linalg.norm(error[n:]) / _ESN_QD_SCALE)     # ||Δqd|| (rad/s, unscaled for display)

	# ── Threshold-gated RLS update (climbing-fiber model) ────────────────────
	# Mirrors the inferior olive → climbing fiber pathway: the update only fires
	# when the prediction error exceeds ESN_ERROR_LEARN_THRESH.  Below that the
	# cerebellum runs pure inference with no weight change.
	if _esn_target_washout_rem > 0:
		_esn_target_washout_rem -= 1   # suppress spike data near target transitions
	elif not _esn_learning_stopped and q_err_mag > ESN_ERROR_LEARN_THRESH:
		phi  = _esn._augment(_esn_r_at_pred)          # (N_aug,)
		Pp   = _esn_P @ phi                           # (N_aug,)
		denom = ESN_RLS_LAMBDA + float(phi @ Pp)      # scalar
		gain  = Pp / denom                            # (N_aug,)
		_esn.W_out += np.outer(error, gain)           # (n_out, N_aug) rank-1 correction
		_esn_P = (_esn_P - np.outer(gain, Pp)) / ESN_RLS_LAMBDA
		_esn_rls_n += 1
		# Mark this learning event on the error plot (climbing-fiber tick).
		if _MPLOT_OK:
			try:
				_esn_err_ax.axvline(t_now, color="cyan", lw=0.5, alpha=0.25)
				_esn_err_fig.canvas.draw_idle()
			except Exception:
				pass

	# ── Convergence milestone (diagnostic only — does NOT stop learning) ───────
	# When EMA‖Δq‖ stays below the threshold long enough, mark the milestone on
	# the plot.  RLS continues updating W_out every step regardless.
	_esn_err_ema_q = (ESN_EMA_ALPHA_CONV * q_err_mag
	                  + (1.0 - ESN_EMA_ALPHA_CONV) * _esn_err_ema_q)

	if not _esn_frozen:                        # _esn_frozen now means "milestone printed"
		if _esn_err_ema_q < ESN_CONVERGENCE_Q_THRESH:
			_esn_conv_count += 1
			if _esn_conv_count >= ESN_CONVERGENCE_WINDOW:
				_esn_frozen = True             # just marks that the milestone fired once
				print(
					f"\n[ESN] ★ CONVERGED at t={t_now:.2f}s  "
					f"EMA‖Δq‖={_esn_err_ema_q:.4f} rad  "
					f"(continual learning continues — W_out still updating)\n",
					flush=True,
				)
				if _MPLOT_OK:
					try:
						_esn_err_ax.axvline(
							t_now, color="lime", ls="--", lw=1.5,
							label=f"Converged t={t_now:.1f}s",
						)
						_esn_err_ax.legend(loc="upper right")
						_esn_err_fig.canvas.draw_idle()
						_esn_err_fig.canvas.flush_events()
					except Exception:
						pass
		else:
			_esn_conv_count = 0   # error spiked — reset counter, keep learning

	# ── Real-time error plot ──────────────────────────────────────────────────
	_esn_err_t.append(t_now)
	_esn_err_q.append(q_err_mag)
	_esn_err_qd.append(qd_err_mag)

	if _MPLOT_OK and _esn_step_count % ESN_PLOT_INTERVAL == 0:
		_esn_line_q.set_xdata(_esn_err_t)
		_esn_line_q.set_ydata(_esn_err_q)
		_esn_line_qd.set_xdata(_esn_err_t)
		_esn_line_qd.set_ydata(_esn_err_qd)
		_esn_err_ax.relim()
		_esn_err_ax.autoscale_view(scalex=True, scaley=True)
		try:
			_esn_err_fig.canvas.draw_idle()
			_esn_err_fig.canvas.flush_events()
		except Exception:
			pass


def _sync_red_target_ball(model: osim.Model, target_xyz: np.ndarray, viz_ball: dict) -> None:
	"""
	Draw/update a red DecorativeSphere at the IK target (Simbody API visualizer only).

	Simbody's Python SWIG binding may return a *copy* from updDecoration(), meaning
	setTransform() on the returned object modifies the copy and not the stored decoration.
	To work around this we:
	  (a) store the Python handle returned at creation time (viz_ball["ref"]) and call
	      setTransform on it every frame — if SWIG wraps a pointer this is sufficient;
	  (b) ALSO call updDecoration() fresh every frame — belt-and-suspenders in case (a)
	      is stale but fresh calls give live access.
	"""
	if not USE_VISUALIZER:
		return
	try:
		viz = model.getVisualizer().getSimbodyVisualizer()
	except RuntimeError:
		return
	x, y, z = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])
	xf = osim.Transform()
	xf.setP(osim.Vec3(x, y, z))
	if viz_ball["idx"] is None:
		sph = osim.DecorativeSphere(float(viz_ball["radius"]))
		sph.setColor(osim.Vec3(1.0, 0.0, 0.0))
		sph.setOpacity(1.0)
		idx = int(viz.getNumDecorations())
		viz.addDecoration(GROUND_MOBOD_IX, xf, sph)
		viz_ball["idx"] = idx
		# Keep the handle; if SWIG wraps a live C++ pointer this suffices for future updates.
		viz_ball["ref"] = viz.updDecoration(idx)
		print(f"[Ball] decoration added at idx={idx}  pos=({x:.3f},{y:.3f},{z:.3f})", flush=True)
	else:
		# (a) update via stored handle
		try:
			viz_ball["ref"].setTransform(xf)
		except Exception:
			pass
		# (b) update via fresh lookup — redundant if (a) worked, essential if (a) was a copy
		try:
			viz.updDecoration(int(viz_ball["idx"])).setTransform(xf)
		except Exception:
			pass


def _bone_transform(p1: np.ndarray, p2: np.ndarray) -> tuple[osim.Transform, float]:
	"""
	Return (transform, half_length) to place a Y-axis-aligned Simbody cylinder
	so that it runs from p1 to p2 in Ground frame.

	Simbody cylinders are aligned with their local Y axis by default.  We build a
	rotation that maps Y → (p2−p1)/|p2−p1| and translate to the segment midpoint.
	Only setTransform (base-class method) is needed for updates, so this approach
	works even when updDecoration() returns the opaque DecorativeGeometry base type.
	"""
	d       = p2 - p1
	length  = float(np.linalg.norm(d))
	half_h  = max(length / 2.0, 1e-6)

	if length < 1e-6:
		return osim.Transform(), half_h

	d_hat = d / length
	y     = np.array([0.0, 1.0, 0.0])
	cross = np.cross(y, d_hat)
	cross_n = float(np.linalg.norm(cross))
	dot     = float(np.dot(y, d_hat))

	if cross_n < 1e-6:
		# Parallel or anti-parallel to Y
		rot = osim.Rotation() if dot > 0 else osim.Rotation(math.pi, osim.Vec3(1, 0, 0))
	else:
		angle = math.acos(max(-1.0, min(1.0, dot)))
		ax    = cross / cross_n
		rot   = osim.Rotation(angle, osim.Vec3(float(ax[0]), float(ax[1]), float(ax[2])))

	mid = (p1 + p2) / 2.0
	xf  = osim.Transform(rot, osim.Vec3(float(mid[0]), float(mid[1]), float(mid[2])))
	return xf, half_h


# Radius (m) for the thin bone cylinders on the ghost stick figure.
_GHOST_BONE_RADIUS_M = 0.007


def _sync_ghost_stick_figure(
	model: osim.Model,
	shoulder_pos: np.ndarray,
	elbow_pos: np.ndarray,
	hand_pos: np.ndarray,
	viz_ghost: dict,
) -> None:
	"""
	Draw/update the cyan lag-ghost stick figure.

	Geometry
	--------
	• 3 DecorativeSpheres at the shoulder / elbow / hand joint positions.
	• 2 thin DecorativeCylinders connecting shoulder→elbow and elbow→hand.

	Why cylinders instead of DecorativeLines
	-----------------------------------------
	updDecoration() returns the *base* DecorativeGeometry type.  setTransform() is
	defined on the base class and reliably updates position + orientation every frame.
	DecorativeLine.setPoint1/setPoint2 are *derived-class* methods and are not
	reachable through the base pointer, so line endpoint updates silently fail.
	A cylinder needs only setTransform — the same mechanism that already works for
	the red target ball.  Bone lengths are constant (rigid arm), so we compute the
	half-height once at creation and only refresh the transform each frame.

	viz_ghost keys (populated on first call)
	-----------------------------------------
	idxs      : dict[str, int]               – decoration indices
	refs      : dict[str, DecorativeGeometry] – stored handles at creation
	half_lens : dict[str, float]             – fixed bone half-lengths (m)
	sph_r     : float                        – sphere radius (m)
	"""
	if not USE_VISUALIZER:
		return
	try:
		viz = model.getVisualizer().getSimbodyVisualizer()
	except RuntimeError:
		return

	color   = osim.Vec3(*GHOST_COLOR)
	opacity = GHOST_OPACITY
	sph_r   = float(viz_ghost["sph_r"])

	def _sphere_xf(pos: np.ndarray) -> osim.Transform:
		xf = osim.Transform()
		xf.setP(osim.Vec3(float(pos[0]), float(pos[1]), float(pos[2])))
		return xf

	joint_items = [
		("shoulder", shoulder_pos),
		("elbow",    elbow_pos),
		("hand",     hand_pos),
	]
	bone_items = [
		("bone_se", shoulder_pos, elbow_pos),
		("bone_eh", elbow_pos,    hand_pos),
	]

	def _add_and_store(geo, xf, idxs, refs, name):
		idx = int(viz.getNumDecorations())
		viz.addDecoration(GROUND_MOBOD_IX, xf, geo)
		idxs[name] = idx
		refs[name] = viz.updDecoration(idx)

	def _update(name, xf, idxs, refs):
		try:
			refs[name].setTransform(xf)
		except Exception:
			pass
		try:
			viz.updDecoration(idxs[name]).setTransform(xf)
		except Exception:
			pass

	if viz_ghost["idxs"] is None:
		idxs: dict[str, int] = {}
		refs: dict = {}
		half_lens: dict[str, float] = {}

		for name, pos in joint_items:
			sph = osim.DecorativeSphere(sph_r)
			sph.setColor(color)
			sph.setOpacity(opacity)
			_add_and_store(sph, _sphere_xf(pos), idxs, refs, name)

		for name, p1, p2 in bone_items:
			xf, half_h = _bone_transform(p1, p2)
			half_lens[name] = half_h
			cyl = osim.DecorativeCylinder(_GHOST_BONE_RADIUS_M, half_h)
			cyl.setColor(color)
			cyl.setOpacity(opacity)
			_add_and_store(cyl, xf, idxs, refs, name)

		viz_ghost["idxs"]      = idxs
		viz_ghost["refs"]      = refs
		viz_ghost["half_lens"] = half_lens

	else:
		idxs      = viz_ghost["idxs"]
		refs      = viz_ghost["refs"]

		for name, pos in joint_items:
			_update(name, _sphere_xf(pos), idxs, refs)

		for name, p1, p2 in bone_items:
			xf, _ = _bone_transform(p1, p2)
			_update(name, xf, idxs, refs)


def _pick_end_effector_body(model: osim.Model) -> osim.Body:
	bodies = model.getBodySet()
	# Prefer something that looks like a hand/wrist/end-effector.
	preferred_substrings = ["hand", "wrist", "radius", "ulna"]
	for sub in preferred_substrings:
		for i in range(bodies.getSize()):
			b = bodies.get(i)
			if sub in b.getName().lower():
				return b
	# Fallback: last non-ground body.
	return bodies.get(bodies.getSize() - 1)


def _find_body_containing(model: osim.Model, substrings: list[str]) -> osim.Body | None:
	"""Return the first Body whose name contains any of the given substrings (case-insensitive)."""
	bodies = model.getBodySet()
	for sub in substrings:
		for i in range(bodies.getSize()):
			b = bodies.get(i)
			if sub.lower() in b.getName().lower():
				return b
	return None


def _get_end_effector_position_in_ground(model: osim.Model, state: osim.State, ee_body: osim.Body) -> np.ndarray:
	# Station at end-effector body origin, expressed in Ground.
	model.realizePosition(state)
	p = ee_body.findStationLocationInGround(state, osim.Vec3(0, 0, 0))
	return np.array([p.get(0), p.get(1), p.get(2)], dtype=float)


def _get_coord_ranges(coords: osim.CoordinateSet, coord_names: list[str]) -> dict[str, tuple[float, float]]:
	ranges: dict[str, tuple[float, float]] = {}
	for name in coord_names:
		c = coords.get(name)
		ranges[name] = (c.getRangeMin(), c.getRangeMax())
	return ranges


def _copy_state_for_kinematics(state: osim.State) -> osim.State:
	"""
	Create a copy of State for kinematic calculations (IK/target sampling) so we never
	mutate the live integrator state (which can cause visible 'jumps').
	"""
	try:
		return osim.State(state)
	except Exception:
		# Fallback: return same handle if copy ctor unavailable (older bindings).
		return state


def sample_reachable_targets(
	model: osim.Model,
	state: osim.State,
	coords: osim.CoordinateSet,
	coord_names: list[str],
	n_targets: int,
	seed: int = 0,
) -> list[np.ndarray]:
	"""
	Generates reachable 3D targets by sampling random joint angles within coordinate bounds
	and forward-evaluating the model's end-effector position.
	"""
	rng = np.random.default_rng(seed)
	ee_body = _pick_end_effector_body(model)
	coord_ranges = _get_coord_ranges(coords, coord_names)

	targets: list[np.ndarray] = []
	s_tmp = _copy_state_for_kinematics(state)
	needs_restore = s_tmp is state
	for _ in range(n_targets):
		if needs_restore:
			q_save = {cn: coords.get(cn).getValue(state) for cn in coord_names}
		for name in coord_names:
			lo, hi = coord_ranges[name]
			coords.get(name).setValue(s_tmp, float(rng.uniform(lo, hi)))
		targets.append(_get_end_effector_position_in_ground(model, s_tmp, ee_body))
		if needs_restore:
			for cn in coord_names:
				coords.get(cn).setValue(state, q_save[cn])
	return targets


def sample_one_reachable_target(
	model: osim.Model,
	state: osim.State,
	coords: osim.CoordinateSet,
	coord_names: list[str],
	rng: np.random.Generator,
	current_ee_pos: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
	"""
	Returns (target_pos, q_goal) where:
	  • target_pos  – 3-D end-effector position in Ground (m)
	  • q_goal      – the exact joint angles that produce target_pos via FK

	Because q_goal comes directly from FK (not from a separate IK solve), the
	target is *guaranteed* reachable.  The caller can use q_goal as the PID
	setpoint immediately without running solve_ik_2dof_numeric — which avoids
	the failure mode where IK starts from a distant warm-start and diverges.

	Fallback order when no fully-valid candidate is found within MAX_TRIES:
	  1. Best in-box position (even if too close to current EE).
	  2. The last sampled position (prevents returning None).
	"""
	ee_body = _pick_end_effector_body(model)
	coord_ranges = _get_coord_ranges(coords, coord_names)
	s_tmp = _copy_state_for_kinematics(state)
	needs_restore = s_tmp is state
	if needs_restore:
		q_save = {cn: coords.get(cn).getValue(state) for cn in coord_names}

	best_pos: np.ndarray | None = None
	best_q:   dict[str, float] | None = None
	last_pos: np.ndarray | None = None
	last_q:   dict[str, float] = {}

	for _ in range(TARGET_SAMPLE_MAX_TRIES):
		q_sample: dict[str, float] = {}
		for name in coord_names:
			lo, hi = coord_ranges[name]
			v = float(rng.uniform(lo, hi))
			q_sample[name] = v
			coords.get(name).setValue(s_tmp, v)

		p = _get_end_effector_position_in_ground(model, s_tmp, ee_body)
		last_pos = p
		last_q   = q_sample

		x_ok = TARGET_X_RANGE_M[0] <= float(p[0]) <= TARGET_X_RANGE_M[1]
		y_ok = TARGET_Y_RANGE_M[0] <= float(p[1]) <= TARGET_Y_RANGE_M[1]
		z_ok = TARGET_Z_RANGE_M[0] <= float(p[2]) <= TARGET_Z_RANGE_M[1]

		if x_ok and y_ok and z_ok:
			# Always keep the latest in-box candidate as a fallback.
			best_pos = p
			best_q   = dict(q_sample)
			sep_ok = (
				current_ee_pos is None
				or float(np.linalg.norm(p - current_ee_pos)) >= MIN_TARGET_SEPARATION_M
			)
			if sep_ok:
				# Found a well-separated, in-box, guaranteed-reachable target.
				break

	if needs_restore:
		for cn in coord_names:
			coords.get(cn).setValue(state, q_save[cn])

	# Return best candidate found (prefer separated in-box > any in-box > last sample).
	if best_pos is not None:
		return np.array(best_pos, dtype=float), best_q  # type: ignore[return-value]
	return np.array(last_pos, dtype=float), last_q


def solve_ik_2dof_numeric(
	model: osim.Model,
	state: osim.State,
	coords: osim.CoordinateSet,
	coord_names: list[str],
	target_pos_ground: np.ndarray,
	max_iters: int = 200,
	tol_m: float = 5e-4,
	alpha: float = 0.5,
	fd_eps: float = 1e-4,
	apply_to_state: bool = True,
) -> dict[str, float]:
	"""
	Simple numeric IK for 2 coordinates using Jacobian-transpose with finite-difference Jacobian.
	Returns a dict coord_name -> solved_value (clamped to coordinate bounds).
	If apply_to_state is False, joint coordinates are restored to their pre-IK values after solving
	(so the simulation state is not jumped to the goal pose).
	"""
	if len(coord_names) != 2:
		raise ValueError("solve_ik_2dof_numeric expects exactly 2 coordinates")

	# Never iterate IK by repeatedly changing the live integrator State; use a copy unless
	# the caller explicitly wants the solution applied.
	s_ik = state if apply_to_state else _copy_state_for_kinematics(state)
	needs_restore = (not apply_to_state) and (s_ik is state)

	ee_body = _pick_end_effector_body(model)
	coord_ranges = _get_coord_ranges(coords, coord_names)
	q_before = {cn: coords.get(cn).getValue(s_ik) for cn in coord_names}

	def clamp(name: str, v: float) -> float:
		lo, hi = coord_ranges[name]
		return float(min(max(v, lo), hi))

	# Initialize with current coordinate values.
	q = np.array([coords.get(coord_names[0]).getValue(s_ik), coords.get(coord_names[1]).getValue(s_ik)], dtype=float)

	for _ in range(max_iters):
		coords.get(coord_names[0]).setValue(s_ik, clamp(coord_names[0], float(q[0])))
		coords.get(coord_names[1]).setValue(s_ik, clamp(coord_names[1], float(q[1])))

		p = _get_end_effector_position_in_ground(model, s_ik, ee_body)
		e = (target_pos_ground - p)  # meters
		if float(np.linalg.norm(e)) < tol_m:
			break

		# Finite-difference Jacobian: dp/dq (3x2)
		J = np.zeros((3, 2), dtype=float)
		for j in range(2):
			q_pert = q.copy()
			q_pert[j] += fd_eps
			coords.get(coord_names[0]).setValue(s_ik, clamp(coord_names[0], float(q_pert[0])))
			coords.get(coord_names[1]).setValue(s_ik, clamp(coord_names[1], float(q_pert[1])))
			p_pert = _get_end_effector_position_in_ground(model, s_ik, ee_body)
			J[:, j] = (p_pert - p) / fd_eps

		# Jacobian-transpose update in joint space.
		dq = alpha * (J.T @ e)
		q = q + dq

	# Final clamp + write back.
	q0 = clamp(coord_names[0], float(q[0]))
	q1 = clamp(coord_names[1], float(q[1]))
	coords.get(coord_names[0]).setValue(s_ik, q0)
	coords.get(coord_names[1]).setValue(s_ik, q1)
	result = {coord_names[0]: q0, coord_names[1]: q1}
	if needs_restore:
		for cn, v in q_before.items():
			coords.get(cn).setValue(state, v)
	return result


def _cubic_blend(q0: float, q1: float, T: float, t: float) -> tuple[float, float, float]:
	"""
	Cubic with zero endpoint velocities.
	Returns (q, qd, qdd) at time t in [0, T].
	"""
	if T <= 0:
		return (q1, 0.0, 0.0)
	s = float(np.clip(t / T, 0.0, 1.0))
	h00 = 2 * s**3 - 3 * s**2 + 1
	h01 = -2 * s**3 + 3 * s**2
	q = h00 * q0 + h01 * q1

	# Time derivatives
	ds_dt = 1.0 / T
	dh00_ds = 6 * s**2 - 6 * s
	dh01_ds = -6 * s**2 + 6 * s
	qd = (dh00_ds * q0 + dh01_ds * q1) * ds_dt

	d2h00_ds2 = 12 * s - 6
	d2h01_ds2 = -12 * s + 6
	qdd = (d2h00_ds2 * q0 + d2h01_ds2 * q1) * (ds_dt**2)
	return (float(q), float(qd), float(qdd))


def _actuator_name_to_function_index_map(function_set: osim.FunctionSet) -> dict[int, str]:
	# PrescribedController stores functions aligned to actuators added.
	# We'll expose a best-effort mapping: index -> label for debugging/logging.
	m: dict[int, str] = {}
	for i in range(function_set.getSize()):
		m[i] = f"actuator_{i}"
	return m


def _choose_actuator_indices_for_coords(model: osim.Model, torque_set: osim.SetActuators, coord_names: list[str]) -> dict[str, int]:
	"""
	Best-effort mapping from coordinate name to an actuator index in torque_set.
	- Prefers actuators whose name contains a coordinate substring.
	- Falls back to the indices used in your original script (1 for shoulder, 3 for elbow) if present.
	"""
	by_name = {torque_set.get(i).getName().lower(): i for i in range(torque_set.getSize())}

	def find_index(coord_name: str) -> int | None:
		key = coord_name.lower().replace("_", "")
		for name, idx in by_name.items():
			if key in name.replace("_", ""):
				return idx
		# also try partial tokens
		tokens = coord_name.lower().split("_")
		for name, idx in by_name.items():
			if any(tok in name for tok in tokens):
				return idx
		return None

	mapping: dict[str, int] = {}
	for cn in coord_names:
		idx = find_index(cn)
		if idx is not None:
			mapping[cn] = idx

	# Fallback to the original script's actuator indices if we couldn't find matches.
	if "shoulder_elv" in coord_names and "shoulder_elv" not in mapping and torque_set.getSize() > 1:
		mapping["shoulder_elv"] = 1
	if "elbow_flexion" in coord_names and "elbow_flexion" not in mapping and torque_set.getSize() > 3:
		mapping["elbow_flexion"] = 3

	# Final sanity: ensure all requested coords are mapped.
	for cn in coord_names:
		if cn not in mapping:
			raise RuntimeError(f"Could not map coordinate '{cn}' to a torque actuator in the model.")
	return mapping


torqueSet = model.getActuators()
coords = model.updCoordinateSet()

# The two DOFs we actively control with PID.
coord_names = ["shoulder_elv", "elbow_flexion"]
_controlled_set = set(coord_names)

# Lock ONLY the three actuated-but-uncontrolled shoulder DOFs so they don't
# drift under gravity.  We must NOT lock the 10 scapular/clavicular coordinates
# that are coupled to shoulder_elv via CoordinateCouplerConstraints — locking
# them would conflict with the couplers and freeze shoulder_elv completely.
# (sternoclavicular_r*, acromioclavicular_r*, unrotscap_r*, unrothum_r* are all
#  coupling-dependent and must stay free.)
_DRIFT_DOFS_TO_LOCK = {"elv_angle", "shoulder_rot", "pro_sup"}
_locked_count = 0
for _i in range(coords.getSize()):
	_c = coords.get(_i)
	if _c.getName() in _DRIFT_DOFS_TO_LOCK:
		try:
			_c.setLocked(state, True)
			_locked_count += 1
		except Exception:
			pass
print(f"[Init] Locked {_locked_count} drift DOFs {_DRIFT_DOFS_TO_LOCK}; "
	  f"actively controlling: {coord_names}", flush=True)

# Print the initial hand position so we know where the arm starts in world space.
_ee_body_init = _pick_end_effector_body(model)
model.realizePosition(state)
_ee_init = _get_end_effector_position_in_ground(model, state, _ee_body_init)
print(f"[Init] Default hand position (Ground frame): "
	  f"x={_ee_init[0]:.3f}  y={_ee_init[1]:.3f}  z={_ee_init[2]:.3f}", flush=True)

model.equilibrateMuscles(state)
manager = osim.Manager(model)
manager.initialize(state)

# ── Dedicated kinematics-only model ──────────────────────────────────────────
# IK and target sampling MUST NOT touch the simulation state – any external
# coordinate mutation corrupts the integrator's cached derivatives and causes
# the arm to visibly jump/teleport each frame.  Using a completely separate
# model+state guarantees the simulation state is never touched.
_ik_model = osim.Model(model_filename)
_ik_state = _ik_model.initSystem()
_ik_coords = _ik_model.updCoordinateSet()
# Lock the same three drift DOFs in the IK model for consistency.
for _i in range(_ik_coords.getSize()):
	_c = _ik_coords.get(_i)
	if _c.getName() in _DRIFT_DOFS_TO_LOCK:
		try:
			_c.setLocked(_ik_state, True)
		except Exception:
			pass
# ─────────────────────────────────────────────────────────────────────────────

if USE_VISUALIZER:
	try:
		model.getVisualizer().show(state)
	except RuntimeError:
		USE_VISUALIZER = False

# Remove the brown ground plane so the skeleton floats against a clean backdrop.
# Simbody BackgroundType: 0 = GroundAndSky (default, shows wood floor), 1 = SolidColor.
if USE_VISUALIZER:
	try:
		_viz = model.getVisualizer().getSimbodyVisualizer()
		# Try the named enum attribute first; fall back to integer 1 for older bindings.
		_bg_solid = getattr(_viz, "SolidColor", 1)
		_viz.setBackgroundType(_bg_solid)
		_viz.setBackgroundColor(osim.Vec3(0.08, 0.08, 0.12))  # dark navy — arm pops visually
	except Exception:
		pass

actuator_index_for_coord = _choose_actuator_indices_for_coords(model, torqueSet, coord_names)

rng = np.random.default_rng(RNG_SEED)

# PID gains — with optimal_force=1.0 these are the actual torques applied (Nm).
#
# Why these values:
#   From the observed 2s oscillation, effective shoulder inertia I ≈ 3 kg·m².
#   Critical damping requires Kd_crit = 2·√(Kp·I).
#   → At Kp=100: Kd_crit = 2·√300 ≈ 34.6  → Kd=28 gives ζ ≈ 0.81 (well-damped).
#   Gravity torque ≈ 10 Nm → steady-state PD error ≈ 10/100 = 0.1 rad → ~4 cm Cartesian.
#   Ki eliminates that remaining offset so the arm actually arrives at the target.
#
#   Elbow I ≈ 0.5 kg·m²; at Kp=60: Kd_crit = 2·√30 ≈ 11 → Kd=14 (overdamped, smooth).
Kp = {"shoulder_elv": 100.0, "elbow_flexion": 60.0}
Kd = {"shoulder_elv": 28.0,  "elbow_flexion": 14.0}
Ki = {"shoulder_elv": 15.0,  "elbow_flexion": 8.0}   # integral: kills gravity offset
# Anti-windup: clamp the accumulated integral (in rad·s) to this value.
KI_MAX = {"shoulder_elv": 3.0, "elbow_flexion": 2.0}

# ── Initial target ────────────────────────────────────────────────────────────
ee_body = _pick_end_effector_body(model)
ee_pos  = _get_end_effector_position_in_ground(model, state, ee_body)

# ALL IK / target sampling goes through the dedicated _ik_* objects so the
# simulation state is never externally modified.
# q_goal_seg comes directly from the FK-generating joint angles — guaranteed reachable.
target_pos_seg, q_goal_seg = sample_one_reachable_target(
	_ik_model, _ik_state, _ik_coords, coord_names, rng, current_ee_pos=ee_pos
)
for _cn in coord_names:
	_ik_coords.get(_cn).setValue(_ik_state, q_goal_seg[_cn])

viz_ball: dict  = {"idx": None, "ref": None, "radius": BALL_RADIUS_M}
viz_ghost: dict = {"idxs": None, "refs": None, "sph_r": GHOST_SPHERE_RADIUS_M}

# Bodies used to compute ghost joint positions each frame.
# humerus origin  ≈ GH / shoulder joint centre.
# ulna origin     ≈ elbow joint centre (radial head / trochlear notch).
# ee_body already found above  ≈ hand / wrist.
_ghost_shoulder_body = _find_body_containing(model, ["humer"])
_ghost_elbow_body    = _find_body_containing(model, ["ulna", "radius"])
# Diagnostic: let the user know which bodies were found.
print(
	f"[Ghost] shoulder body = {_ghost_shoulder_body.getName() if _ghost_shoulder_body else 'NOT FOUND'}, "
	f"elbow body = {_ghost_elbow_body.getName() if _ghost_elbow_body else 'NOT FOUND'}",
	flush=True,
)

# ── ESN input / target vector helpers ────────────────────────────────────────
# These reference coord_names (defined above) at call time — no forward-ref issue.

def _build_esn_input_vec(
	q: dict, qd: dict, qdd: dict, tau: dict, q_goal: dict,
) -> np.ndarray:
	"""Flatten (q, qd, qdd, tau, q_goal) for each coord → length-10 ESN input."""
	return np.array(
		[q.get(cn, 0.0)      for cn in coord_names] +
		[qd.get(cn, 0.0)     for cn in coord_names] +
		[qdd.get(cn, 0.0)    for cn in coord_names] +
		[tau.get(cn, 0.0)    for cn in coord_names] +
		[q_goal.get(cn, 0.0) for cn in coord_names],
		dtype=float,
	)


_ESN_QD_SCALE = stepsize          # convert rad/s → rad/step  (≈ 0.005)
_ESN_QD_CLIP  = 5.0 * _ESN_QD_SCALE  # clip target qd to ±5 steps-worth of motion

def _build_esn_target_vec(q: dict, qd: dict) -> np.ndarray:
	"""Flatten actual (q_next, qd_next) → length-4 ESN target.

	Velocity is scaled by dt (rad/s → rad/step) so that q and qd live
	on the same numerical scale.  Scaled qd is also clipped to suppress
	spike corruption of the replay buffer at target transitions.
	"""
	q_arr  = np.array([q.get(cn, 0.0)  for cn in coord_names], dtype=float)
	qd_arr = np.array([qd.get(cn, 0.0) for cn in coord_names], dtype=float)
	qd_scaled = np.clip(qd_arr * _ESN_QD_SCALE, -_ESN_QD_CLIP, _ESN_QD_CLIP)
	return np.concatenate([q_arr, qd_scaled])


# ── ESN instance ──────────────────────────────────────────────────────────────
_ESN_N_IN  = len(coord_names) * 5   # q + qd + qdd + tau + q_goal = 10
_ESN_N_OUT = len(coord_names) * 2   # q_next + qd_next = 4

_esn = EchoStateNetwork(
	n_inputs        = _ESN_N_IN,
	n_reservoir     = ESN_N_RESERVOIR,
	n_outputs       = _ESN_N_OUT,
	spectral_radius = 0.95,
	connectivity    = 0.1,
	leak_rate       = 0.3,
	input_scaling   = 1.0,
	bias_scaling    = 0.1,
	regularization  = 1e-6,
	seed            = 42,
)
# W_out starts at zero — online_update builds it up from scratch each episode.
_esn.W_out = np.zeros((_ESN_N_OUT, 2 * ESN_N_RESERVOIR + 1))

# ESN runtime state shared between on_simulation_step and after_simulation_step.
_esn_r:          np.ndarray = np.zeros(ESN_N_RESERVOIR)  # current reservoir state
_esn_r_at_pred:  np.ndarray = np.zeros(ESN_N_RESERVOIR)  # state at last prediction
_esn_last_pred:  np.ndarray = np.zeros(_ESN_N_OUT)        # last prediction vector
_esn_step_count: int        = 0                            # steps elapsed
# Convergence tracking
_esn_frozen:     bool  = False
_esn_conv_count: int   = 0
_esn_err_ema_q:  float = 1.0
# Per-target washout counter: counts down from ESN_TARGET_WASHOUT_STEPS after each
# new target.  RLS updates are suppressed while this is > 0.
_esn_target_washout_rem: int = 0

# ── Online RLS state ──────────────────────────────────────────────────────────
# P  : running estimate of (ΦᵀΦ)⁻¹, shape (N_aug, N_aug).
#      Updated every step via Sherman-Morrison rank-1 formula — O(N²) per step.
# _esn_rls_n : number of RLS updates performed (used for warmup gating).
_N_aug    = 2 * ESN_N_RESERVOIR + 1
_esn_P:   np.ndarray = ESN_RLS_DELTA * np.eye(_N_aug)  # high initial uncertainty
_esn_rls_n: int          = 0
_esn_learning_stopped: bool = False   # set True after ESN_LEARN_STOP_REACHES reaches

# ── Dedicated FK state for ESN ghost arm ──────────────────────────────────────
# We need a state separate from _ik_state so FK evaluation (predicted q → 3D)
# does not clobber the IK warm-start values used by solve_ik_2dof_numeric.
_esn_fk_state  = _ik_model.initSystem()
_esn_fk_coords = _ik_model.updCoordinateSet()
for _i in range(_esn_fk_coords.getSize()):
	_c = _esn_fk_coords.get(_i)
	if _c.getName() in _DRIFT_DOFS_TO_LOCK:
		try:
			_c.setLocked(_esn_fk_state, True)
		except Exception:
			pass

# Bodies in the IK model used to convert predicted joint angles → Ground positions.
_esn_fk_shoulder_body = _find_body_containing(_ik_model, ["humer"])
_esn_fk_elbow_body    = _find_body_containing(_ik_model, ["ulna", "radius"])
_esn_fk_hand_body     = _pick_end_effector_body(_ik_model)


def _esn_fk(q_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Forward kinematics for the ESN ghost arm.

	Sets the predicted joint angles on the dedicated FK state, realizes position,
	and returns (shoulder_pos, elbow_pos, hand_pos) in Ground frame (metres).
	"""
	for i, cn in enumerate(coord_names):
		try:
			_esn_fk_coords.get(cn).setValue(_esn_fk_state, float(q_pred[i]))
		except Exception:
			pass
	try:
		_ik_model.realizePosition(_esn_fk_state)
	except Exception:
		return np.zeros(3), np.zeros(3), np.zeros(3)

	def _body_pos(body: osim.Body | None) -> np.ndarray:
		if body is None:
			return np.zeros(3)
		p = body.findStationLocationInGround(_esn_fk_state, osim.Vec3(0, 0, 0))
		return np.array([p.get(0), p.get(1), p.get(2)], dtype=float)

	return (
		_body_pos(_esn_fk_shoulder_body),
		_body_pos(_esn_fk_elbow_body),
		_body_pos(_esn_fk_hand_body),
	)


print(
	f"[ESN] reservoir={ESN_N_RESERVOIR}  n_in={_ESN_N_IN}  n_out={_ESN_N_OUT}  "
	f"RLS λ={ESN_RLS_LAMBDA}  δ={ESN_RLS_DELTA}  warmup={ESN_RLS_WARMUP}  "
	f"washout={ESN_WASHOUT_STEPS} steps",
	flush=True,
)

# ── Real-time error plot ───────────────────────────────────────────────────────
_esn_err_t:   list[float] = []
_esn_err_q:   list[float] = []
_esn_err_qd:  list[float] = []

if _MPLOT_OK:
	plt.ion()
	_esn_err_fig, _esn_err_ax = plt.subplots(figsize=(9, 4))
	_esn_err_fig.canvas.manager.set_window_title("ESN Online Learning — Prediction Error")
	_esn_line_q,  = _esn_err_ax.plot([], [], lw=1.2, color="royalblue",  label="||Δq||  (rad)")
	_esn_line_qd, = _esn_err_ax.plot([], [], lw=1.2, color="darkorange", label="||Δqd|| (rad/s)", alpha=0.7)
	# Horizontal line showing the learning threshold — RLS fires only above this.
	_esn_err_ax.axhline(ESN_ERROR_LEARN_THRESH, color="cyan", lw=1.0, ls=":",
	                    alpha=0.7, label=f"Learn thresh ({ESN_ERROR_LEARN_THRESH} rad)")
	_esn_err_ax.set_xlabel("Simulation time (s)")
	_esn_err_ax.set_ylabel("Prediction error magnitude")
	_esn_err_ax.set_title("ESN one-step-ahead error  (↓ = learning)  |  cyan ticks = RLS update fired")
	_esn_err_ax.legend(loc="upper right", fontsize=7)
	_esn_err_ax.set_xlim(0, MAX_SIM_TIME_S)
	_esn_err_fig.tight_layout()
	try:
		_esn_err_fig.canvas.draw()
		_esn_err_fig.canvas.flush_events()
	except Exception:
		pass
	print("[ESN] Matplotlib error window opened.", flush=True)
else:
	# Placeholders so after_simulation_step references don't raise NameError.
	_esn_err_fig = _esn_err_ax = _esn_line_q = _esn_line_qd = None
	print("[ESN] Matplotlib unavailable — error plot disabled.", flush=True)

targets_completed = 0
dist = float("inf")
move_t0 = float(state.getTime())
last_progress_t = float(state.getTime())
wall_t0 = time.time()
# Integral accumulator for PID gravity compensation (rad·s, reset on new target).
integral_err: dict[str, float] = {cn: 0.0 for cn in coord_names}

print(
	f"[Run] visualizer={'on' if USE_VISUALIZER else 'off'}  "
	f"stepsize={stepsize}  max_sim_time={MAX_SIM_TIME_S}s  "
	f"Kp={Kp}  Kd={Kd}",
	file=sys.stdout,
)
print(
	f"[Run] initial target=({target_pos_seg[0]:.3f},{target_pos_seg[1]:.3f},{target_pos_seg[2]:.3f})  "
	f"IK goal: {q_goal_seg}",
	file=sys.stdout,
)

brain = osim.PrescribedController.safeDownCast(model.getControllerSet().get(0))
functionSet = brain.get_ControlFunctions()

while float(state.getTime()) + stepsize <= MAX_SIM_TIME_S + 1e-9:
	t_cur  = float(state.getTime())          # time at start of this step (= t_prev for the after-hook)
	t_next = t_cur + stepsize

	# ── Read current joint state ──────────────────────────────────────────────
	q_cur  = {cn: coords.get(cn).getValue(state)      for cn in coord_names}
	qd_cur = {cn: coords.get(cn).getSpeedValue(state) for cn in coord_names}

	# ── PID controller ────────────────────────────────────────────────────────
	# τ = Kp·(q_goal−q) + Ki·∫(q_goal−q)dt − Kd·q̇
	# The integral term accumulates the gravity-induced steady-state offset so
	# the arm actually arrives at the Cartesian target rather than settling short.
	tau_cmd: dict[str, float] = {}
	for cn in coord_names:
		pos_err = float(q_goal_seg[cn] - q_cur[cn])
		integral_err[cn] = float(np.clip(
			integral_err[cn] + pos_err * stepsize,
			-KI_MAX[cn], KI_MAX[cn],
		))
		tau = (Kp[cn] * pos_err
			   + Ki[cn] * integral_err[cn]
			   - Kd[cn] * float(qd_cur[cn]))
		lim = float(TAU_LIMIT.get(cn, 1e9))
		tau_cmd[cn] = float(np.clip(tau, -lim, lim))

	# ── Write controls ────────────────────────────────────────────────────────
	for cn in coord_names:
		func = osim.Constant.safeDownCast(functionSet.get(actuator_index_for_coord[cn]))
		func.setValue(tau_cmd[cn])

	# ── User hook ─────────────────────────────────────────────────────────────
	ee_pos = _get_end_effector_position_in_ground(model, state, ee_body)

	# Shoulder and elbow joint positions for the ghost stick figure.
	# _get_end_effector_position_in_ground works on any body, not just the EE.
	shoulder_pos = (
		_get_end_effector_position_in_ground(model, state, _ghost_shoulder_body)
		if _ghost_shoulder_body is not None else ee_pos
	)
	elbow_pos = (
		_get_end_effector_position_in_ground(model, state, _ghost_elbow_body)
		if _ghost_elbow_body is not None else ee_pos
	)

	# Compute joint accelerations: realizeAcceleration fills in qdd given the
	# current position, velocity, and already-applied torques (tau_cmd above).
	qdd_cur: dict[str, float] = {}
	try:
		model.realizeAcceleration(state)
		for cn in coord_names:
			qdd_cur[cn] = float(coords.get(cn).getAccelerationValue(state))
	except Exception:
		qdd_cur = {cn: float("nan") for cn in coord_names}

	state_snapshot = {
		"q":                       q_cur,
		"qd":                      qd_cur,
		"qdd":                     qdd_cur,
		"q_goal":                  dict(q_goal_seg),
		"integral_err":            dict(integral_err),
		"end_effector_pos_ground": ee_pos,
		"shoulder_pos_ground":     shoulder_pos,
		"elbow_pos_ground":        elbow_pos,
		"target_pos_ground":       target_pos_seg.copy(),
		"targets_completed":       targets_completed,
		"dist_to_target_m":        dist,
	}
	ghost_positions = on_simulation_step(
		float(state.getTime()), state_snapshot,
		{f"{cn}_torque": tau_cmd[cn] for cn in coord_names},
	)

	# ── Integrate one step ────────────────────────────────────────────────────
	state = manager.integrate(t_next)

	# ── Post-step: build observed state at t_now, call after_simulation_step ──
	q_now  = {cn: coords.get(cn).getValue(state)      for cn in coord_names}
	qd_now = {cn: coords.get(cn).getSpeedValue(state) for cn in coord_names}
	ee_pos = _get_end_effector_position_in_ground(model, state, ee_body)

	after_simulation_step(
		t_cur,
		float(state.getTime()),
		state_snapshot,
		{f"{cn}_torque": tau_cmd[cn] for cn in coord_names},
		{
			"q":                       q_now,
			"qd":                      qd_now,
			"end_effector_pos_ground": ee_pos,
		},
	)

	# ── Post-step reach check (reuses ee_pos computed above) ─────────────────
	dist   = float(np.linalg.norm(ee_pos - target_pos_seg))
	t_rel_done = float(state.getTime()) - move_t0
	arm_settled = all(abs(qd_cur[cn]) < 0.6 for cn in coord_names)
	if t_rel_done >= MIN_TIME_BEFORE_REACH_S and dist < REACH_TOL_M and arm_settled:
		targets_completed += 1
		old_target = target_pos_seg.copy()
		# q_goal_seg from sampling = exact FK-generating angles → always reachable, residual = 0.
		target_pos_seg, q_goal_seg = sample_one_reachable_target(
			_ik_model, _ik_state, _ik_coords, coord_names, rng, current_ee_pos=ee_pos
		)
		for _cn in coord_names:
			_ik_coords.get(_cn).setValue(_ik_state, q_goal_seg[_cn])
		move_t0 = float(state.getTime())
		integral_err = {cn: 0.0 for cn in coord_names}   # reset to avoid windup fighting new goal
		# Reset reservoir so old-goal context doesn't corrupt predictions for the
		# new target.  The washout phase (ESN_TARGET_WASHOUT_STEPS steps of real
		# arm data + new q_goal) then re-fills _esn_r with fresh context before
		# we trust the readout again.
		_esn_r[:] = 0.0
		_esn_target_washout_rem = ESN_TARGET_WASHOUT_STEPS
		new_dist = float(np.linalg.norm(ee_pos - target_pos_seg))
		t_reach  = float(state.getTime())

		# ── Mark reach on error plot ──────────────────────────────────────────
		if _MPLOT_OK:
			try:
				_esn_err_ax.axvline(t_reach, color="orange", lw=0.8, alpha=0.6,
				                    label=f"reach" if targets_completed == 1 else None)
				_esn_err_fig.canvas.draw_idle()
				_esn_err_fig.canvas.flush_events()
			except Exception:
				pass

		# ── Freeze learning after ESN_LEARN_STOP_REACHES reaches ─────────────
		if targets_completed >= ESN_LEARN_STOP_REACHES and not _esn_learning_stopped:
			_esn_learning_stopped = True
			print(
				f"\n[ESN] ◼ LEARNING STOPPED after {targets_completed} reaches "
				f"at t={t_reach:.2f}s — W_out frozen, pure inference from here\n",
				flush=True,
			)
			if _MPLOT_OK:
				try:
					_esn_err_ax.axvline(t_reach, color="red", lw=2.0, ls="--",
					                    label=f"Learning stopped (reach {targets_completed})")
					_esn_err_ax.legend(loc="upper right", fontsize=7)
					_esn_err_fig.canvas.draw_idle()
					_esn_err_fig.canvas.flush_events()
				except Exception:
					pass

		print(
			f"  ✓ target {targets_completed} reached at t={t_reach:.2f}s "
			f"(dist={dist:.3f}m to old ball)  "
			f"→ NEW target ({target_pos_seg[0]:.2f},{target_pos_seg[1]:.2f},{target_pos_seg[2]:.2f})  "
			f"q_goal={{{', '.join(f'{k}: {v:.3f}' for k,v in q_goal_seg.items())}}}  "
			f"dist_to_new={new_dist:.3f}m",
			flush=True,
		)

	# ── Red ball + ghost stick figure: sync EVERY frame ─────────────────────
	# Both are called before show() so the visualizer renders updated positions.
	_sync_red_target_ball(model, target_pos_seg, viz_ball)

	if ghost_positions is not None:
		_sync_ghost_stick_figure(
			model,
			ghost_positions["ghost_shoulder"],
			ghost_positions["ghost_elbow"],
			ghost_positions["ghost_hand"],
			viz_ghost,
		)

	# ── Visualizer show ───────────────────────────────────────────────────────
	if USE_VISUALIZER:
		try:
			model.getVisualizer().show(state)
		except RuntimeError:
			pass

	# ── Real-time pacing ──────────────────────────────────────────────────────
	if USE_VISUALIZER and REALTIME_WHEN_VISUALIZING:
		sleep_s = float(state.getTime()) - (time.time() - wall_t0)
		if sleep_s > 0:
			time.sleep(min(sleep_s, 0.03))

	# ── Progress print ────────────────────────────────────────────────────────
	if float(state.getTime()) - last_progress_t >= PROGRESS_EVERY_SIM_S:
		last_progress_t = float(state.getTime())
		print(
			f"[t={state.getTime():.1f}s] reached={targets_completed}  "
			f"dist={dist:.3f}m  target=({target_pos_seg[0]:.2f},{target_pos_seg[1]:.2f},{target_pos_seg[2]:.2f})",
			flush=True,
		)

statesDegrees = manager.getStateStorage()
statesDegrees.printToFile("IK_Torque_Trajectory.sto", "w")

# ── Save step log to CSV ──────────────────────────────────────────────────────
import csv as _csv

_LOG_FILE = "simulation_log.csv"
if _step_log:
	_fields = list(_step_log[0].keys())
	with open(_LOG_FILE, "w", newline="") as _fh:
		_writer = _csv.DictWriter(_fh, fieldnames=_fields)
		_writer.writeheader()
		_writer.writerows(_step_log)
	print(
		f"[Log] Wrote {len(_step_log):,} rows × {len(_fields)} columns → {_LOG_FILE}",
		flush=True,
	)

print(
	f"Simulation complete: final time = {state.getTime():.4f} s, targets reached = {targets_completed} "
	f"→ wrote IK_Torque_Trajectory.sto + {_LOG_FILE} "
	f"(visualizer {'enabled' if USE_VISUALIZER else 'off; set TRY_VISUALIZER=True or OPENSIM_USE_VISUALIZER=1'})",
	file=sys.stdout,
)
