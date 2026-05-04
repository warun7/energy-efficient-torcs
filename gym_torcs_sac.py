try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
import numpy as np
# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import shutil
import time


class TorcsEnv:
    termination_limit_progress = 5  # [km/h]
    default_speed = 50
    AALBORG_TRACK_LENGTH = 2598.63  # metres, used for prospective fuel budget signal

    # Distance-based stuck detection: if car hasn't advanced this many metres
    # within STUCK_WINDOW steps it is considered stuck.
    STUCK_WINDOW = 100        # steps between progress checkpoints
    STUCK_MIN_PROGRESS = 8.0  # minimum metres required in that window

    initial_reset = True

    @staticmethod
    def _torcs_bin():
        # Allow explicit binary override, e.g. /usr/games/torcs
        return os.environ.get("TORCS_BIN", "torcs")

    @classmethod
    def _launch_torcs(cls, vision=False):
        torcs_bin = cls._torcs_bin()
        if vision:
            os.system(f"{torcs_bin} -nodamage -vision &")
        else:
            os.system(f"{torcs_bin} &")

    @staticmethod
    def _ensure_scr_server_user_config():
        """
        Some custom TORCS builds don't copy scr_server config into ~/.torcs.
        Without this file TORCS may crash with GfParmGetNum bad handle.
        """
        user_driver_dir = os.path.expanduser("~/.torcs/drivers/scr_server")
        user_driver_xml = os.path.join(user_driver_dir, "scr_server.xml")
        if os.path.exists(user_driver_xml):
            return

        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "gym_torcs", "vtorcs-RL-color", "src", "drivers", "scr_server", "scr_server.xml"),
        ]
        prefix = os.environ.get("TORCS_PREFIX")
        if prefix:
            candidates.append(
                os.path.join(prefix, "share", "games", "torcs", "drivers", "scr_server", "scr_server.xml")
            )
        data = os.environ.get("TORCS_DATADIR")
        if data:
            candidates.append(os.path.join(data, "drivers", "scr_server", "scr_server.xml"))
        candidates.append("/usr/local/share/games/torcs/drivers/scr_server/scr_server.xml")

        source_xml = next((p for p in candidates if p and os.path.exists(p)), None)
        if source_xml is None:
            return
        os.makedirs(user_driver_dir, exist_ok=True)
        shutil.copy2(source_xml, user_driver_xml)

    @staticmethod
    def _force_raceman_track(track_name="aalborg", category="road"):
        """
        Force raceman configs to use a specific track.

        TORCS reads config from ~/.torcs/config/raceman/, but the launcher
        script (setup_linux.sh) can overwrite user configs from the
        system-level copies at /usr/local/share/games/torcs/config/raceman/
        based on file timestamps.  To be safe we patch BOTH locations.
        """
        import re

        config_dirs = [
            os.path.expanduser("~/.torcs/config/raceman"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".torcs-rebuild", "share", "games", "torcs", "config", "raceman"),
            "/usr/local/share/games/torcs/config/raceman",
        ]

        xml_names = (
            "practice.xml",
            "quickrace.xml",
            "ncrace.xml",
            "endrace.xml",
            "dtmrace.xml",
            "champ.xml",
        )

        for race_dir in config_dirs:
            for name in xml_names:
                path = os.path.join(race_dir, name)
                if not os.path.exists(path):
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        txt = f.read()

                    # Use regex to replace the track name and category under the Tracks section
                    txt = re.sub(r'(<section name="Tracks">.*?<attstr name="name" val=")[^"]+("/>)', r'\g<1>' + track_name + r'\2', txt, flags=re.DOTALL, count=1)
                    txt = re.sub(r'(<section name="Tracks">.*?<attstr name="category" val=")[^"]+("/>)', r'\g<1>' + category + r'\2', txt, flags=re.DOTALL, count=1)

                    try:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(txt)
                    except PermissionError:
                        # Root-owned file in a world-writable dir: delete + recreate
                        tmp = path + ".tmp"
                        with open(tmp, "w", encoding="utf-8") as f:
                            f.write(txt)
                        os.remove(path)
                        os.rename(tmp, path)
                except (PermissionError, OSError):
                    pass

    @staticmethod
    def _ensure_raceman_practice_config():
        """
        Keep ~/.torcs practice race config aligned with this project's
        known-good SCR setup to avoid TORCS parser/assert crashes.
        """
        user_race_dir = os.path.expanduser("~/.torcs/config/raceman")
        user_practice_xml = os.path.join(user_race_dir, "practice.xml")

        # Prefer the upstream vtorcs raceman config shape (fewer custom sections),
        # then fall back to local copies.
        local_candidates = [
            os.path.join(os.path.dirname(__file__), "gym_torcs", "vtorcs-RL-color", "src", "raceman", "practice.xml"),
            os.path.join(os.path.dirname(__file__), "gym_torcs", "practice.xml"),
            os.path.join(os.path.dirname(__file__), "practice.xml"),
        ]
        source_xml = next((p for p in local_candidates if os.path.exists(p)), None)
        if source_xml is None:
            return

        os.makedirs(user_race_dir, exist_ok=True)
        shutil.copy2(source_xml, user_practice_xml)

    @staticmethod
    def _normalize_raceman_driver_block():
        """
        Normalize raceman driver blocks to a safe single scr_server entry.

        Previous regex rewrites replaced every driver `module` and `idx` in files
        like ncrace/champ, which created invalid duplicate driver definitions
        (e.g. many `idx=0` entries). TORCS then aborts in GfFatal during startup.
        Here we replace the whole Drivers section with a minimal known-good block.
        """
        import re
        config_dirs = [
            os.path.expanduser("~/.torcs/config/raceman"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".torcs-rebuild", "share", "games", "torcs", "config", "raceman"),
            "/usr/local/share/games/torcs/config/raceman",
        ]
        xml_names = ("practice.xml", "quickrace.xml", "ncrace.xml", "endrace.xml", "dtmrace.xml", "champ.xml")
        drivers_block = (
            "  <section name=\"Drivers\">\n"
            "    <attnum name=\"maximum number\" val=\"1\"/>\n"
            "    <attstr name=\"focused module\" val=\"scr_server\"/>\n"
            "    <attnum name=\"focused idx\" val=\"0\"/>\n"
            "    <section name=\"1\">\n"
            "      <attnum name=\"idx\" val=\"0\"/>\n"
            "      <attstr name=\"module\" val=\"scr_server\"/>\n"
            "    </section>\n"
            "  </section>"
        )

        for race_dir in config_dirs:
            for name in xml_names:
                path = os.path.join(race_dir, name)
                if not os.path.exists(path):
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        txt = f.read()
                    
                    # Replace the complete Drivers section atomically.
                    txt_new = re.sub(
                        r'<section name="Drivers">[\s\S]*?</section>\s*(?=<section name="Configuration">|</params>)',
                        drivers_block,
                        txt,
                        flags=0,
                        count=1,
                    )
                    if txt_new == txt:
                        # If section is missing/malformed, skip rather than writing junk.
                        continue
                    txt = txt_new
                    
                    # Try to write back (handling potential root-owned files)
                    try:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(txt)
                    except PermissionError:
                        tmp = path + ".tmp"
                        with open(tmp, "w", encoding="utf-8") as f:
                            f.write(txt)
                        os.remove(path)
                        os.rename(tmp, path)
                except Exception:
                    pass

    def __init__(self, vision=False, throttle=False, gear_change=False, fuel_lambda=0.0, max_laps=1):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.fuel_lambda = float(fuel_lambda)
        self.max_laps = int(max_laps)

        self.initial_run = True
        self._ensure_scr_server_user_config()
        self._ensure_raceman_practice_config()
        self._normalize_raceman_driver_block()
        self._force_raceman_track(track_name="aalborg", category="road")

        # Lap tracking state
        self._prev_dist_from_start = 0.0
        self._lap_count = 0
        self._lap_start_step = 0
        self._episode_dist_raced_start = 0.0
        self.last_lap_time = None
        self.last_termination_reason = None

        # Distance-based stuck detection state
        self._stuck_checkpoint_dist = 0.0  # distFromStart at the last checkpoint
        self._stuck_window_step = 0        # step counter for the current window

        self.initial_fuel = None
        self.total_fuel_consumed = 0.0
        self.total_fuel_budget = np.random.uniform(3.0, 8.0)
        self.next_waypoint = 500.0
        self.waypoint_start_step = 0
        self.last_fuel_penalty = 0.0
        self.last_fuel_rate_avg = 0.0
        self.last_budget_rate_target = 0.0
        self.last_over_consumption = 0.0

        # Fix 3: 20-step rolling window for smooth fuel consumption rate
        self._fuel_history = col.deque(maxlen=20)

        os.system("pkill -f torcs")
        time.sleep(0.5)
        self._launch_torcs(self.vision)
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if self.throttle:
                if client.S.d['speedX'] > 50:
                    action_torcs['gear'] = 2
                if client.S.d['speedX'] > 80:
                    action_torcs['gear'] = 3
                if client.S.d['speedX'] > 110:
                    action_torcs['gear'] = 4
                if client.S.d['speedX'] > 140:
                    action_torcs['gear'] = 5
                if client.S.d['speedX'] > 170:
                    action_torcs['gear'] = 6
        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        client.get_servers_input()

        # If the race finished (e.g. max laps reached), snakeoil shuts down the socket.
        # We need to catch this and terminate the RL episode cleanly.
        if not client.so:
            client.R.d['meta'] = True
            self.last_termination_reason = "race_finished_socket_closed"
            return self.get_obs(), 0.0, True, self._build_step_info(done=True, reward=0.0, speed_kmh=0.0)

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])

        # Speed-based reward: reward forward progress, penalise sliding.
        # Base reward: linear forward progress gives strong feedback even at low speeds
        progress = sp * np.cos(obs['angle']) - np.abs(sp * np.sin(obs['angle']))
        reward = progress / 5.0  # at 50 km/h ≈ +10.0 per step

        # Budget-Aware Soft Penalty
        current_fuel = obs.get('fuel', 100.0)
        if self.initial_fuel is None:
            self.initial_fuel = current_fuel
        
        FUEL_SCALING = 5.0
        fuel_consumed = (self.initial_fuel - current_fuel) * FUEL_SCALING
        fuel_remaining = self.total_fuel_budget - fuel_consumed
        b_t = fuel_remaining / max(self.total_fuel_budget, 0.001)
        b_t = np.clip(b_t, 0.0, 1.0)

        step_fuel_delta = obs_pre.get('fuel', 100.0) - current_fuel
        fuel_consumed_this_step = max(0.0, step_fuel_delta) * FUEL_SCALING

        depletion_rate = fuel_consumed_this_step / max(b_t, 0.01)
        nominal_depletion_rate = self.total_fuel_budget / 1000.0
        budget_penalty = -0.3 * max(0.0, depletion_rate - nominal_depletion_rate)

        if not hasattr(self, "_printed_fuel_rate") and fuel_consumed_this_step > 0:
            print(f"Scaled step fuel: {fuel_consumed_this_step:.6f}, target rate: {nominal_depletion_rate:.6f}")
            self._printed_fuel_rate = True

        self.last_fuel_penalty = budget_penalty
        self.last_budget_rate_target = nominal_depletion_rate
        reward += budget_penalty

        # Fuel-efficiency incentive on tight budgets.
        # Gives the agent positive signal for keeping fuel in the tank when b_t < 0.5.
        # Without this, tight budget episodes always end at -200 with no intermediate gradient.
        if b_t < 0.5:
            efficiency_bonus = 0.5 * b_t  # scales from 0 (empty) to 0.25 (half tank)
            reward += efficiency_bonus

        # Minimum-speed pressure: constant -0.5/step whenever going slower than 10 km/h.
        # Removes the "sitting still is neutral" equilibrium without being a terminal.
        if sp < 10.0:
            reward -= 0.5

        damage_delta = obs['damage'] - obs_pre['damage']
        if damage_delta > 0:
            reward -= (damage_delta / 10.0)  # Soft collision penalty
            
        # Sector time bonus
        cur_dist = obs.get('distFromStart', 0.0)
        cur_dist_raced = obs.get('distRaced', 0.0)
        episode_dist_raced = max(0.0, cur_dist_raced - self._episode_dist_raced_start)

        # Initialize waypoint correctly on the first step
        if self.time_step == 0:
            self.next_waypoint = cur_dist + 500.0

        if cur_dist >= self.next_waypoint and cur_dist > self._prev_dist_from_start:
            steps_taken = self.time_step - self.waypoint_start_step
            # Target pace: 100 km/h = 27.7 m/s -> 500m = 18 seconds = 900 steps
            target_steps = 900
            bonus = max(0, (target_steps - steps_taken) * 0.1)
            reward += bonus
            self.next_waypoint += 500.0
            self.waypoint_start_step = self.time_step

        # --- Lap completion detection & lap-time reward ---
        # Detect crossing the start/finish line (distance wraps from a large value back to near zero)
        crossed_start_finish = self._prev_dist_from_start > 0 and cur_dist < self._prev_dist_from_start - 500
        completed_required_distance = episode_dist_raced >= ((self._lap_count + 1) * self.AALBORG_TRACK_LENGTH * 0.8)
        if crossed_start_finish and completed_required_distance:
            self._lap_count += 1
            self.next_waypoint = cur_dist + 500.0  # Reset waypoint for the new lap

            # Use the server's precise internal lap timing if available
            server_lap_time = obs.get('lastLapTime', 0.0)
            if server_lap_time > 0:
                lap_time_sec = server_lap_time
            else:
                # Fallback to step-based timing if sensor missing
                steps_this_lap = self.time_step - self._lap_start_step
                lap_time_sec = steps_this_lap / 50.0

            # Prevent fake lap triggers right at the start of the race
            if lap_time_sec > 10.0:
                lap_bonus = max(0, 50000.0 / max(lap_time_sec, 1.0))
                reward += lap_bonus
                print(f"\n*** LAP {self._lap_count} COMPLETE — "
                      f"{lap_time_sec:.1f}s — bonus {lap_bonus:.1f} ***\n")

                # Store the lap data in a log file
                try:
                    log_path = os.path.join(os.path.dirname(__file__), "lap_times.log")
                    with open(log_path, "a") as f:
                        f.write(f"Lap {self._lap_count}: {lap_time_sec:.1f}s (Bonus: {lap_bonus:.1f}, Step: {self.time_step})\n")
                except Exception as e:
                    print(f"Failed to log lap time: {e}")

                self.last_lap_time = float(lap_time_sec)
                if self._lap_count >= self.max_laps:
                    client.R.d['meta'] = True
                    self.last_termination_reason = "lap_complete"

            self._lap_start_step = self.time_step
        self._prev_dist_from_start = cur_dist

        # Termination judgement #########################
        episode_terminate = False

        # Fix 1: Terminal penalties raised to -200 to be meaningful against ~25k max episode reward

        # 1. Out of track termination
        if abs(trackPos) > 0.999:
            print(">>> OFF TRACK TERMINATION <<<")
            reward -= 200.0
            episode_terminate = True
            client.R.d['meta'] = True
            self.last_termination_reason = "off_track"

        # 2. Spin out termination (facing backwards or completely sideways)
        if abs(obs['angle']) > 1.0:
            print(">>> SPIN OUT TERMINATION <<<")
            reward -= 200.0
            episode_terminate = True
            client.R.d['meta'] = True
            self.last_termination_reason = "spin_out"

        # 3. Distance-progress stuck detection.
        # Every STUCK_WINDOW steps check if the car has advanced STUCK_MIN_PROGRESS metres.
        # Replaces the speed-based check which could be gamed by micro-nudging.
        self._stuck_window_step += 1
        if self._stuck_window_step >= self.STUCK_WINDOW:
            progress_this_window = cur_dist - self._stuck_checkpoint_dist
            # Handle track wraparound (lap crossings): progress is always ≥ 0
            if progress_this_window < -500:  # wrapped around the track
                progress_this_window += self.AALBORG_TRACK_LENGTH
            if progress_this_window < self.STUCK_MIN_PROGRESS:
                print(f">>> STUCK TERMINATION <<< (advanced only {progress_this_window:.1f}m in {self.STUCK_WINDOW} steps)")
                reward -= 200.0
                episode_terminate = True
                client.R.d['meta'] = True
                self.last_termination_reason = "stuck"
            # Reset checkpoint for next window regardless of outcome
            self._stuck_checkpoint_dist = cur_dist
            self._stuck_window_step = 0

        # 4. Driving backward
        if np.cos(obs['angle']) < 0:
            print(">>> BACKWARD TERMINATION <<<")
            reward -= 200.0
            episode_terminate = True
            client.R.d['meta'] = True
            self.last_termination_reason = "backward"

        # 5. Budget Exhaustion Terminal
        current_fuel = obs.get('fuel', 100.0)
        FUEL_SCALING = 5.0
        fuel_consumed = (self.initial_fuel - current_fuel) * FUEL_SCALING if self.initial_fuel is not None else 0.0
        fuel_remaining = self.total_fuel_budget - fuel_consumed
        if fuel_remaining <= 0:
            # Progress-proportional partial credit: makes "farther before empty" better than "less far"
            # At 72% progress: -200 + 108 = -92 instead of flat -200
            # This creates a differentiable gradient pushing agent to stretch fuel further each episode
            dist_raced_this_ep = max(0.0, obs.get('distRaced', 0.0) - self._episode_dist_raced_start)
            track_progress_frac = min(1.0, dist_raced_this_ep / self.AALBORG_TRACK_LENGTH)
            print(f">>> FUEL EXHAUSTION TERMINATION <<< (progress: {track_progress_frac:.2%})")
            reward -= (200.0 - 150.0 * track_progress_frac)  # flat -200 at 0%, -50 at 100%
            episode_terminate = True
            client.R.d['meta'] = True
            self.last_termination_reason = "fuel_exhaustion"

        if client.R.d['meta'] is True:  # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        done = bool(client.R.d['meta'])
        if done and self.last_termination_reason is None:
            self.last_termination_reason = "terminated"
        return self.get_obs(), reward, done, self._build_step_info(done=done, reward=reward, speed_kmh=float(sp))

    def reset(self, relaunch=False, budget=None):
        #print("Reset")

        self.time_step = 0
        self._prev_dist_from_start = 0.0
        self._lap_start_step = 0
        self._lap_count = 0
        self._episode_dist_raced_start = 0.0
        self.last_lap_time = None
        self.last_termination_reason = None

        # Reset distance-based stuck detection
        self._stuck_checkpoint_dist = 0.0
        self._stuck_window_step = 0

        self.initial_fuel = None
        self.total_fuel_consumed = 0.0
        self.total_fuel_budget = budget if budget is not None else np.random.uniform(3.0, 8.0)
        self.next_waypoint = 500.0
        self.waypoint_start_step = 0
        self.last_fuel_penalty = 0.0
        self.last_fuel_rate_avg = 0.0
        self.last_budget_rate_target = 0.0
        self.last_over_consumption = 0.0
        self._fuel_history = col.deque(maxlen=20)  # reset rolling window on episode start

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()
            try:
                self.client.shutdown()
            except Exception:
                pass

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self._episode_dist_raced_start = float(obs.get('distRaced', 0.0))
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        os.system("pkill -f torcs")

    def get_obs(self):
        return self.observation

    def _build_step_info(self, done=False, reward=0.0, speed_kmh=0.0):
        current_fuel = 0.0
        fuel_consumed = 0.0
        FUEL_SCALING = 5.0
        if self.initial_fuel is not None and hasattr(self, "client") and hasattr(self.client, "S"):
            current_fuel = float(self.client.S.d.get("fuel", self.initial_fuel))
            fuel_consumed = float((self.initial_fuel - current_fuel) * FUEL_SCALING)
        track_progress = 0.0
        fuel_budget_remaining = 0.0
        if hasattr(self, "observation") and self.observation is not None:
            fuel_budget_remaining = float(np.asarray(self.observation.fuelBudgetRemaining).reshape(-1)[0])
            if hasattr(self.client, "S"):
                dist_from_start = float(self.client.S.d.get("distFromStart", 0.0))
                track_progress = (dist_from_start % self.AALBORG_TRACK_LENGTH) / self.AALBORG_TRACK_LENGTH
        return {
            "reward_step": float(reward),
            "speedX_kmh": float(speed_kmh),
            "lap_count": int(self._lap_count),
            "lap_completed": bool(self._lap_count >= self.max_laps),
            "lap_time_sec": float(self.last_lap_time) if self.last_lap_time is not None else None,
            "fuel_lambda": float(self.fuel_lambda),
            "fuel_budget": float(self.total_fuel_budget),
            "fuel_current": float(current_fuel),
            "fuel_consumed": float(fuel_consumed),
            "fuel_penalty": float(self.last_fuel_penalty),
            "fuel_rate_avg": float(self.last_fuel_rate_avg),
            "fuel_budget_rate_target": float(self.last_budget_rate_target),
            "fuel_over_consumption": float(self.last_over_consumption),
            "fuel_budget_remaining": float(fuel_budget_remaining),
            "track_progress": float(track_progress),
            "termination_reason": self.last_termination_reason if done else None,
            "done": bool(done),
        }

    def reset_torcs(self):
       #print("relaunch torcs")
        os.system("pkill -f torcs")
        time.sleep(1.0)
        self._launch_torcs(self.vision)
        time.sleep(5.0)  # Wait for GUI to load
        os.system('sh autostart.sh')
        time.sleep(2.0)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})
            torcs_action.update({'brake': u[2]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': int(u[3])})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)
        return np.array([r, g, b], dtype=np.uint8)

    def make_observaton(self, raw_obs):
        current_fuel = raw_obs.get('fuel', 100.0)
        if self.initial_fuel is None:
            self.initial_fuel = current_fuel

        FUEL_SCALING = 5.0
        fuel_consumed = (self.initial_fuel - current_fuel) * FUEL_SCALING
        fuel_level = (self.initial_fuel - (fuel_consumed / FUEL_SCALING)) / max(1.0, self.initial_fuel)

        fuel_remaining = self.total_fuel_budget - fuel_consumed
        fuel_budget_remaining = fuel_remaining / max(self.total_fuel_budget, 0.001)
        fuel_budget_remaining = np.clip(fuel_budget_remaining, 0.0, 1.0)
        b_t = fuel_budget_remaining

        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                     'opponents',
                     'rpm',
                     'track', 
                     'trackPos',
                     'wheelSpinVel',
                     'fuelLevel',
                     'fuelConsumed',
                     'fuelBudgetRemaining',
                     'b_t']
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                               angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                               damage=np.array(raw_obs['damage'], dtype=np.float32),
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               fuelLevel=np.array([fuel_level], dtype=np.float32),
                               fuelConsumed=np.array([fuel_consumed], dtype=np.float32),
                               fuelBudgetRemaining=np.array([fuel_budget_remaining], dtype=np.float32),
                               b_t=np.array([b_t], dtype=np.float32))
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)
