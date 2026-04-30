import os
import sys
import time

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
# Simbody Ground mobilized body index for fixed-world decorations
GROUND_MOBOD_IX = 0
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


def on_simulation_step(t: float, state_snapshot: dict, torque_commands: dict) -> None:
	"""
	Called every simulation timestep (stepsize s).

	Parameters
	----------
	t : float
	    Current simulation time (seconds).
	state_snapshot : dict
	    "q"                       – joint positions  {coord_name: rad}
	    "qd"                      – joint velocities {coord_name: rad/s}
	    "qdd"                     – joint accelerations {coord_name: rad/s²}
	    "q_goal"                  – PID setpoint joint angles {coord_name: rad}
	    "integral_err"            – PID integral accumulator {coord_name: rad·s}
	    "end_effector_pos_ground" – hand XYZ in Ground frame (m),  np.ndarray
	    "target_pos_ground"       – active target ball XYZ (m),    np.ndarray
	    "dist_to_target_m"        – Euclidean hand-to-target (m)
	    "targets_completed"       – targets reached so far (int)
	torque_commands : dict
	    "shoulder_elv_torque"  – torque at shoulder elevation (Nm)
	    "elbow_flexion_torque" – torque at elbow flexion (Nm)
	"""
	q   = state_snapshot["q"]
	qd  = state_snapshot["qd"]
	qdd = state_snapshot.get("qdd", {})
	ee  = state_snapshot["end_effector_pos_ground"]
	tgt = state_snapshot["target_pos_ground"]

	# Next STEP:
	'''
		Take in current shoulder position, velocity, and acceleration with torque commands
		Predict next time step sensory. 

		Use a forward PC, using t to predict t+1, when t+1, run weight update for f(t)-->(t+1)
		measure free energy to determine if the model is learning

		subtract predictions from actual to determine error. 

		after some number of reaches, add a weight to the arm and see how it reacts in free energy
	'''

	...
	# _step_log.append({
	# 	"t_s":                      t,
	# 	# ── Joint positions (rad) ─────────────────────────────────────────────
	# 	"shoulder_elv_rad":         q.get("shoulder_elv",   float("nan")),
	# 	"elbow_flexion_rad":        q.get("elbow_flexion",  float("nan")),
	# 	# ── Joint velocities (rad/s) ──────────────────────────────────────────
	# 	"shoulder_elv_vel_rad_s":   qd.get("shoulder_elv",  float("nan")),
	# 	"elbow_flexion_vel_rad_s":  qd.get("elbow_flexion", float("nan")),
	# 	# ── Joint accelerations (rad/s²) ──────────────────────────────────────
	# 	"shoulder_elv_acc_rad_s2":  qdd.get("shoulder_elv",  float("nan")),
	# 	"elbow_flexion_acc_rad_s2": qdd.get("elbow_flexion", float("nan")),
	# 	# ── PID setpoints (rad) ───────────────────────────────────────────────
	# 	"shoulder_elv_goal_rad":    state_snapshot["q_goal"].get("shoulder_elv",  float("nan")),
	# 	"elbow_flexion_goal_rad":   state_snapshot["q_goal"].get("elbow_flexion", float("nan")),
	# 	# ── Motor commands (Nm) ───────────────────────────────────────────────
	# 	"shoulder_elv_torque_Nm":   torque_commands.get("shoulder_elv_torque",  float("nan")),
	# 	"elbow_flexion_torque_Nm":  torque_commands.get("elbow_flexion_torque", float("nan")),
	# 	# ── End-effector position (m) ─────────────────────────────────────────
	# 	"ee_x_m":                   float(ee[0]),
	# 	"ee_y_m":                   float(ee[1]),
	# 	"ee_z_m":                   float(ee[2]),
	# 	# ── Target position (m) ───────────────────────────────────────────────
	# 	"target_x_m":               float(tgt[0]),
	# 	"target_y_m":               float(tgt[1]),
	# 	"target_z_m":               float(tgt[2]),
	# 	"dist_to_target_m":         state_snapshot.get("dist_to_target_m", float("nan")),
	# 	"targets_completed":        int(state_snapshot.get("targets_completed", 0)),
	# })


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
) -> np.ndarray:
	"""Single random reachable end-effector position (Ground)."""
	ee_body = _pick_end_effector_body(model)
	coord_ranges = _get_coord_ranges(coords, coord_names)
	s_tmp = _copy_state_for_kinematics(state)
	needs_restore = s_tmp is state
	if needs_restore:
		q_save = {cn: coords.get(cn).getValue(state) for cn in coord_names}
	pos = None
	for _ in range(TARGET_SAMPLE_MAX_TRIES):
		for name in coord_names:
			lo, hi = coord_ranges[name]
			coords.get(name).setValue(s_tmp, float(rng.uniform(lo, hi)))
		p = _get_end_effector_position_in_ground(model, s_tmp, ee_body)
		x_ok = TARGET_X_RANGE_M[0] <= float(p[0]) <= TARGET_X_RANGE_M[1]
		y_ok = TARGET_Y_RANGE_M[0] <= float(p[1]) <= TARGET_Y_RANGE_M[1]
		z_ok = TARGET_Z_RANGE_M[0] <= float(p[2]) <= TARGET_Z_RANGE_M[1]
		if x_ok and y_ok and z_ok:
			if current_ee_pos is None or float(np.linalg.norm(p - current_ee_pos)) >= MIN_TARGET_SEPARATION_M:
				pos = p
				break
		pos = p
	if needs_restore:
		for cn in coord_names:
			coords.get(cn).setValue(state, q_save[cn])
	return np.array(pos, dtype=float)


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
target_pos_seg = sample_one_reachable_target(
	_ik_model, _ik_state, _ik_coords, coord_names, rng, current_ee_pos=ee_pos
)
q_goal_seg = solve_ik_2dof_numeric(
	_ik_model, _ik_state, _ik_coords, coord_names, target_pos_seg,
	apply_to_state=False
)
# Warm-start seed so next IK call begins near this solution.
for _cn in coord_names:
	_ik_coords.get(_cn).setValue(_ik_state, q_goal_seg[_cn])

viz_ball: dict = {"idx": None, "ref": None, "radius": BALL_RADIUS_M}
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
	t_next = float(state.getTime()) + stepsize

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
		"target_pos_ground":       target_pos_seg.copy(),
		"targets_completed":       targets_completed,
		"dist_to_target_m":        dist,
	}
	on_simulation_step(float(state.getTime()), state_snapshot, {f"{cn}_torque": tau_cmd[cn] for cn in coord_names})

	# ── Integrate one step ────────────────────────────────────────────────────
	state = manager.integrate(t_next)

	# ── Post-step reach check (uses post-integration state) ───────────────────
	ee_pos = _get_end_effector_position_in_ground(model, state, ee_body)
	dist   = float(np.linalg.norm(ee_pos - target_pos_seg))
	t_rel_done = float(state.getTime()) - move_t0
	arm_settled = all(abs(qd_cur[cn]) < 0.6 for cn in coord_names)
	if t_rel_done >= MIN_TIME_BEFORE_REACH_S and dist < REACH_TOL_M and arm_settled:
		targets_completed += 1
		old_target = target_pos_seg.copy()
		target_pos_seg = sample_one_reachable_target(
			_ik_model, _ik_state, _ik_coords, coord_names, rng, current_ee_pos=ee_pos
		)
		q_goal_seg = solve_ik_2dof_numeric(
			_ik_model, _ik_state, _ik_coords, coord_names, target_pos_seg,
			apply_to_state=False
		)
		# ── Warm-start: seed next IK from this solution so it converges faster ──
		for _cn in coord_names:
			_ik_coords.get(_cn).setValue(_ik_state, q_goal_seg[_cn])
		# ────────────────────────────────────────────────────────────────────────
		move_t0 = float(state.getTime())
		integral_err = {cn: 0.0 for cn in coord_names}   # reset to avoid windup fighting new goal
		new_dist = float(np.linalg.norm(ee_pos - target_pos_seg))
		# Verify IK residual by FK on the IK model.
		_ik_model.realizePosition(_ik_state)
		_ik_ee = _get_end_effector_position_in_ground(_ik_model, _ik_state, _pick_end_effector_body(_ik_model))
		_ik_residual = float(np.linalg.norm(_ik_ee - target_pos_seg))
		print(
			f"  ✓ target {targets_completed} reached at t={state.getTime():.2f}s "
			f"(dist={dist:.3f}m to old ball)  "
			f"→ NEW target ({target_pos_seg[0]:.2f},{target_pos_seg[1]:.2f},{target_pos_seg[2]:.2f})  "
			f"IK_residual={_ik_residual:.4f}m  dist_to_new={new_dist:.3f}m",
			flush=True,
		)

	# ── Red ball: sync EVERY frame so the decoration stays current ───────────
	# Called before show() so the visualizer renders the updated position.
	_sync_red_target_ball(model, target_pos_seg, viz_ball)

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
