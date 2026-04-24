import os
# Force CPU-only execution to avoid CUDA init crashes on systems
# with partial/broken NVIDIA runtime libraries.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
import tensorflow as tf
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit

OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=0, episode_count=2000, max_steps=100000, artifact_dir=".", run_tag="run"):    #1 means Train, 0 means simply Run
    artifact_dir = os.path.abspath(artifact_dir)
    os.makedirs(artifact_dir, exist_ok=True)
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    # Use TF1-compatible graph mode under modern TensorFlow.
    tf.compat.v1.disable_eager_execution()
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        # Safe fallback when no GPU devices are registered.
        pass
    config = tf.compat.v1.ConfigProto()
    config.device_count["GPU"] = 0
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)

    actor_weights = os.path.join(artifact_dir, "actormodel.h5")
    critic_weights = os.path.join(artifact_dir, "criticmodel.h5")

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights(actor_weights)
        critic.model.load_weights(critic_weights)
        actor.target_model.load_weights(actor_weights)
        critic.target_model.load_weights(critic_weights)
        print("Weight load successfully")
    except Exception:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")
    episode_stats = []
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
     
        total_reward = 0.
        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]), verbose=0)
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])

            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_actions = actor.target_model.predict(new_states, verbose=0)
            target_q_values = critic.target_model.predict([new_states, target_actions], verbose=0)
           
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states, verbose=0)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights(actor_weights, overwrite=True)
                with open(os.path.join(artifact_dir, "actormodel.json"), "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights(critic_weights, overwrite=True)
                with open(os.path.join(artifact_dir, "criticmodel.json"), "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")
        episode_stats.append({"episode": int(i), "total_reward": float(total_reward), "train": int(train_indicator)})

    env.end()  # This is for shutting down TORCS
    stats_path = os.path.join(artifact_dir, "episode_stats_%s.json" % run_tag)
    with open(stats_path, "w") as sf:
        json.dump(episode_stats, sf, indent=2)
    print("Wrote %s" % stats_path)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [e["episode"] for e in episode_stats]
        ys = [e["total_reward"] for e in episode_stats]
        if xs:
            plt.figure(figsize=(8, 4))
            plt.plot(xs, ys, marker="o", linewidth=1)
            plt.xlabel("Episode")
            plt.ylabel("Total reward")
            plt.title("DDPG TORCS (%s)" % run_tag)
            plt.grid(True, alpha=0.3)
            plot_path = os.path.join(artifact_dir, "reward_curve_%s.png" % run_tag)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=120)
            plt.close()
            print("Wrote %s" % plot_path)
    except Exception as ex:
        print("Skipping plot:", ex)
    print("Finish.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPG agent for TORCS")
    parser.add_argument("--train", type=int, default=0, choices=[0, 1], help="1=train, 0=evaluate")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=100000, dest="max_steps", help="Max steps per episode")
    parser.add_argument("--artifact-dir", type=str, default=".", help="Directory for weights, logs, plots")
    parser.add_argument("--run-tag", type=str, default="run", help="Suffix for stats/plot filenames")
    args = parser.parse_args()
    playGame(
        train_indicator=args.train,
        episode_count=args.episodes,
        max_steps=args.max_steps,
        artifact_dir=args.artifact_dir,
        run_tag=args.run_tag,
    )
