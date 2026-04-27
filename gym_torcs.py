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
    terminal_judge_start = 100  # If after 100 timestep still no progress, terminated
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True

    @staticmethod
    def _torcs_bin():
        # Allow explicit binary override, e.g. /usr/games/torcs
        return os.environ.get("TORCS_BIN", "torcs")

    @classmethod
    def _launch_torcs(cls, vision=False):
        torcs_bin = cls._torcs_bin()
        if vision:
            os.system(f"{torcs_bin} -nofuel -nodamage -nolaptime -vision &")
        else:
            os.system(f"{torcs_bin} -nofuel -nolaptime &")

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

    def __init__(self, vision=False, throttle=False, gear_change=False):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change

        self.initial_run = True
        self._ensure_scr_server_user_config()
        self._ensure_raceman_practice_config()
        self._normalize_raceman_driver_block()
        self._force_raceman_track(track_name="alpine-1", category="road")

        # Lap tracking state
        self._prev_dist_from_start = 0.0
        self._lap_count = 0
        self._lap_start_step = 0
        self.stuck_steps = 0

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
            return self.get_obs(), 0.0, True, {}

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
        # Divided by 50 to give a much stronger baseline reward for speed
        progress = sp * np.cos(obs['angle']) - np.abs(sp * np.sin(obs['angle']))
        reward = progress / 50.0  # ~2.0 at 100 km/h, ~4.0 at 200 km/h

        # Massive bonuses for high speed to encourage the agent to push the throttle
        if sp > 130:
            reward += 15.0
        elif sp > 120:
            reward += 10.0
        elif sp > 110:
            reward += 5.0
        elif sp > 90:
            reward += 1.0

        # Penalty for being far from the track centre
        reward -= 0.1 * min(abs(trackPos), 1.0)
        
        # Specific "Wall Hugging" penalty: if the car is on the edge (>0.95), penalize heavily
        if abs(trackPos) > 0.95:
            reward -= 1.0

        # Spin penalty: if the car is pointed more than 30 degrees away from the track direction
        if abs(obs['angle']) > 0.5: # ~30 degrees
            reward -= 2.0
            
        # Going backwards penalty
        if sp < 0:
            reward -= 5.0

        # Collision penalty: heavily increased (from /10.0 to /2.0)
        damage_delta = obs['damage'] - obs_pre['damage']
        if damage_delta > 0:
            # Now a soft scrape (~5 damage) is -2.5
            # A hard hit (~50 damage) is -25.0
            reward -= (damage_delta / 2.0)

        # --- Lap completion detection & lap-time reward ---
        cur_dist = obs.get('distFromStart', 0.0)
        # Detect crossing the start/finish line (distance wraps from
        # a large value back to near zero)
        if self._prev_dist_from_start > 0 and cur_dist < self._prev_dist_from_start - 500:
            self._lap_count += 1
            
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
                # Reward: faster laps → bigger bonus
                # Increased constant from 1000 to 50000 to provide a much stronger signal
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
                
                # Terminate episode after 3 laps as requested
                if self._lap_count >= 3:
                    client.R.d['meta'] = True
            
            self._lap_start_step = self.time_step
        self._prev_dist_from_start = cur_dist

        # Termination judgement #########################
        episode_terminate = False

        # 1. Stuck detection: Speed below 5km/h for too long
        if sp < 5:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        if self.stuck_steps > 100:
            print(">>> STUCK TERMINATION <<<")
            reward -= 500.0
            episode_terminate = True
            client.R.d['meta'] = True

        # 2. Driving backward
        if np.cos(obs['angle']) < 0:
            print(">>> BACKWARD TERMINATION <<<")
            episode_terminate = True
            client.R.d['meta'] = True

        if client.R.d['meta'] is True:  # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0
        self.stuck_steps = 0
        self._prev_dist_from_start = 0.0
        self._lap_start_step = 0
        self._lap_count = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

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
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        os.system("pkill -f torcs")

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
       #print("relaunch torcs")
        os.system("pkill -f torcs")
        time.sleep(1.0)
        self._launch_torcs(self.vision)
        time.sleep(3.0)  # Wait for GUI to load
        os.system('sh autostart.sh')
        time.sleep(1.0)

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
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                     'opponents',
                     'rpm',
                     'track', 
                     'trackPos',
                     'wheelSpinVel']
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
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
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
