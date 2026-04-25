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
        Force raceman files to use scr_server with idx=0.
        Some custom TORCS builds assert if configured idx/module mismatch.
        """
        race_dir = os.path.expanduser("~/.torcs/config/raceman")
        for name in ("practice.xml", "quickrace.xml"):
            path = os.path.join(race_dir, name)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
            txt = txt.replace('<attstr name="focused module" val="human"/>', '<attstr name="focused module" val="scr_server"/>')
            txt = txt.replace('<attstr name="module" val="human"/>', '<attstr name="module" val="scr_server"/>')
            txt = txt.replace('<attstr name="module" val="chenyi"/>', '<attstr name="module" val="scr_server"/>')
            txt = txt.replace('<attstr name="module" val="inferno"/>', '<attstr name="module" val="scr_server"/>')
            txt = txt.replace('<attstr name="module" val="championship2011server"/>', '<attstr name="module" val="scr_server"/>')
            # Fix driver idx values
            txt = txt.replace('<attnum name="idx" val="1"/>', '<attnum name="idx" val="0"/>')
            txt = txt.replace('<attnum name="idx" val="5"/>', '<attnum name="idx" val="0"/>')
            # Fix focused idx (separate attribute name)
            txt = txt.replace('<attnum name="focused idx" val="1"/>', '<attnum name="focused idx" val="0"/>')
            txt = txt.replace('<attnum name="focused idx" val="5"/>', '<attnum name="focused idx" val="0"/>')
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)

    def __init__(self, vision=False, throttle=False, gear_change=False):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change

        self.initial_run = True
        self._ensure_scr_server_user_config()
        self._ensure_raceman_practice_config()
        self._normalize_raceman_driver_block()

        ##print("launch torcs")
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
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])

        progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        reward = progress

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1

        # Termination judgement #########################
        episode_terminate = False
        #if (abs(track.any()) > 1 or abs(trackPos) > 1):  # Episode is terminated if the car is out of track
        #    reward = -200
        #    episode_terminate = True
        #    client.R.d['meta'] = True

        #if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
        #    if progress < self.termination_limit_progress:
        #        print("No progress")
        #        episode_terminate = True
        #        client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True


        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0

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
        time.sleep(0.5)
        self._launch_torcs(self.vision)
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

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
