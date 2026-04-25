"""
gym_torcs_laptime.py
Extends TorcsEnv with a reward function and observation space shaped
specifically for minimising lap time.

Reward design (based on Ben Lau / yanpanlau DDPG-TORCS + arXiv 2506.06077):
──────────────────────────────────────────────────────────────────────────────
  r_t = Vx * cos(angle)              # forward speed along track axis
      - |Vx * sin(angle)|            # penalise sideways sliding
      - Vx * |trackPos|              # penalise off-center driving

  This reward naturally teaches cornering: sideways velocity and off-center
  position are penalised every step.  Going fast only helps when aligned
  with the track.

  Terminal penalties (episode ends):
    +100  – lap completed
    -(accumulated_reward + 10)  – off-track  (return-canceling, clipped)
     −10  – backwards, stuck, crawling

Observation (31-dim)
────────────────────
  Same 29-element base layout PLUS:
    [29] curLapTime   / 300.0
    [30] distFromStart/ 6000.0
"""

import numpy as np
import collections as col
import copy
import os
import time

from gym_torcs import TorcsEnv
import snakeoil3_gym as snakeoil3


_MAX_LAP_TIME_S   = 300.0
_MAX_TRACK_LENGTH = 6000.0


class TorcsEnvLapTime(TorcsEnv):
    """
    TORCS environment optimised for fastest-lap training.
    """

    STATE_DIM = 31

    def __init__(self, vision=False, throttle=True, gear_change=False):
        if not throttle:
            throttle = True
        super().__init__(vision=vision, throttle=throttle, gear_change=gear_change)
        self.best_lap_time   = float('inf')
        self.lap_times       = []
        self._prev_dist      = 0.0
        self._episode_reward = 0.0
        self._stuck_ticks    = 0

    # ------------------------------------------------------------------ #
    #  Observation                                                         #
    # ------------------------------------------------------------------ #

    def make_observaton(self, raw_obs):
        names = [
            'focus',
            'speedX', 'speedY', 'speedZ', 'angle', 'damage',
            'opponents',
            'rpm',
            'track',
            'trackPos',
            'wheelSpinVel',
            'curLapTime',
            'distFromStart',
        ]
        Observation = col.namedtuple('Observation', names)
        return Observation(
            focus=np.array(raw_obs['focus'],        dtype=np.float32) / 200.,
            speedX=np.array(raw_obs['speedX'],      dtype=np.float32) / 300.0,
            speedY=np.array(raw_obs['speedY'],      dtype=np.float32) / 300.0,
            speedZ=np.array(raw_obs['speedZ'],      dtype=np.float32) / 300.0,
            angle=np.array(raw_obs['angle'],        dtype=np.float32) / 3.1416,
            damage=np.array(raw_obs['damage'],      dtype=np.float32),
            opponents=np.array(raw_obs['opponents'],dtype=np.float32) / 200.,
            rpm=np.array(raw_obs['rpm'],            dtype=np.float32) / 10000.,
            track=np.array(raw_obs['track'],        dtype=np.float32) / 200.,
            trackPos=np.array(raw_obs['trackPos'],  dtype=np.float32),
            wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
            curLapTime=np.float32(
                np.clip(raw_obs.get('curLapTime', 0.0) / _MAX_LAP_TIME_S, 0., 1.)
            ),
            distFromStart=np.float32(
                np.clip(raw_obs.get('distFromStart', 0.0) / _MAX_TRACK_LENGTH, 0., 1.)
            ),
        )

    def get_obs_vector(self):
        """
        Returns the flat 31-element numpy array used as the DDPG state input.
        """
        ob = self.observation
        return np.hstack((
            ob.angle,
            ob.track,
            ob.trackPos,
            ob.speedX,
            ob.speedY,
            ob.speedZ,
            ob.wheelSpinVel / 100.0,
            ob.rpm,
            ob.curLapTime,
            ob.distFromStart,
        ))

    # ------------------------------------------------------------------ #
    #  Step                                                                #
    # ------------------------------------------------------------------ #

    def step(self, u):
        client = self.client

        this_action  = self.agent_to_torcs(u)
        action_torcs = client.R.d

        steer = float(np.clip(this_action['steer'], -1.0, 1.0))
        accel = float(np.clip(this_action['accel'],  0.0, 1.0))
        brake = float(np.clip(this_action['brake'],  0.0, 1.0))

        action_torcs['steer'] = steer
        action_torcs['accel'] = accel
        action_torcs['brake'] = brake

        if self.gear_change:
            action_torcs['gear'] = this_action['gear']
        else:
            sp = client.S.d['speedX']
            action_torcs['gear'] = (
                6 if sp > 170 else
                5 if sp > 140 else
                4 if sp > 110 else
                3 if sp >  80 else
                2 if sp >  50 else
                1
            )

        obs_pre = copy.deepcopy(client.S.d)

        client.respond_to_server()
        client.get_servers_input()

        obs = client.S.d
        self.observation = self.make_observaton(obs)

        # Raw sensor values for reward computation
        angle    = obs['angle']
        trackPos = obs['trackPos']
        damage   = obs['damage']
        speedX   = obs.get('speedX', 0.0)   # km/h
        cur_lap  = obs.get('curLapTime',    0.0)
        last_lap = obs.get('lastLapTime',   0.0)
        pre_cur  = obs_pre.get('curLapTime', 0.0)
        dist_now = obs.get('distFromStart', 0.0)

        # Track dist_delta for lap detection
        dist_delta = dist_now - self._prev_dist
        if dist_delta < -500:
            dist_delta += _MAX_TRACK_LENGTH
        self._prev_dist = dist_now

        # ── Ben Lau reward: proven to teach cornering ────────────────── #
        # Vx * cos(angle)  = forward speed along track axis (good)
        # |Vx * sin(angle)| = sideways sliding speed (bad)
        # Vx * |trackPos|  = off-center penalty scaled by speed (bad)
        sp = speedX / 300.0   # normalise to ~[0,1] range
        reward = (sp * np.cos(angle)
                  - abs(sp * np.sin(angle))
                  - sp * abs(trackPos))

        self._episode_reward += reward

        # ── Lap completion detection ─────────────────────────────────── #
        lap_bonus = 0.0
        if pre_cur > cur_lap and last_lap > 0.0:
            lap_bonus  = 100.0
            reward    += lap_bonus
            self.lap_times.append(last_lap)
            if last_lap < self.best_lap_time:
                self.best_lap_time = last_lap
                print("  [LapTime] NEW BEST: %.3f s" % last_lap)
            else:
                print("  [LapTime] lap: %.3f s  (best: %.3f s)"
                      % (last_lap, self.best_lap_time))

        # ── Termination conditions ───────────────────────────────────── #
        episode_terminate = False

        if abs(trackPos) > 1.0:                       # off track
            penalty = -(self._episode_reward + 10.0)
            reward = float(np.clip(penalty, -200.0, -5.0))
            episode_terminate = True
            client.R.d['meta'] = True

        elif np.cos(angle) < 0:                       # driving backwards
            reward = -10.0
            episode_terminate = True
            client.R.d['meta'] = True

        elif damage > 0 and obs_pre.get('damage', 0.0) == 0.0:
            reward -= 10.0

        elif self.time_step > 500 and self._episode_reward < 0:
            reward = -10.0
            episode_terminate = True
            client.R.d['meta'] = True

        elif self.time_step > 200 and speedX < 3.0:
            reward = -10.0
            episode_terminate = True
            client.R.d['meta'] = True

        # Stuck detection: large angle + no progress for 25 ticks
        if abs(angle) > np.deg2rad(45.0) and abs(dist_delta) < 0.01:
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0
        if self._stuck_ticks > 25:
            reward = -10.0
            episode_terminate = True
            client.R.d['meta'] = True

        if client.R.d['meta']:
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        info = {
            'curLapTime':    cur_lap,
            'lastLapTime':   last_lap,
            'lap_bonus':     lap_bonus,
            'best_lap_time': self.best_lap_time,
            'dist_progress': dist_delta,
        }
        return self.get_obs_vector(), reward, client.R.d['meta'], info

    # ------------------------------------------------------------------ #
    #  Reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(self, relaunch=False):
        self.lap_times       = []
        self.time_step       = 0
        self._prev_dist      = 0.0
        self._episode_reward = 0.0
        self._stuck_ticks    = 0

        if not self.initial_reset:
            try:
                self.client.R.d['meta'] = True
                self.client.respond_to_server()
            except Exception:
                pass
            if relaunch:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        for attempt in range(1, 4):
            try:
                self.client = snakeoil3.Client(p=3101, vision=self.vision)
                self.client.MAX_STEPS = np.inf
                self.client.get_servers_input()
                break
            except TimeoutError as e:
                print("\n[reset] TimeoutError on attempt %d: %s" % (attempt, e))
                print("[reset] Force-relaunching TORCS …")
                self.reset_torcs()
                time.sleep(2)
        else:
            raise RuntimeError("TORCS failed to respond after 3 relaunch attempts.")

        obs = self.client.S.d
        self.observation = self.make_observaton(obs)
        self._prev_dist  = obs.get('distFromStart', 0.0)
        self.last_u      = None
        self.initial_reset = False

        return self.get_obs_vector()
