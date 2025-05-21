import time
import numpy as np
from agent import TransformerAgent
from env.wb_env import *
import cv2  # Add this import


class MyAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = np.zeros_like(self.env.action_space.sample())
        
        # Define the action mapping
        directions = {
            'w': [1, 0, 0, 0],  # W
            'a': [0, 1, 0, 0],  # A
            's': [0, 0, 1, 0],  # S
            'd': [0, 0, 0, 1],  # D
            'z': [1, 1, 0, 0],  # W + A
            'x': [1, 0, 0, 1],  # W + D
            'c': [0, 1, 1, 0],  # A + S
            'v': [0, 0, 1, 1],  # S + D
        }

        actions = {
            '1': [1, 0, 0, 0, 0, 0],  # Jump
            '2': [0, 1, 0, 0, 0, 0],  # Light Attack
            '3': [0, 0, 1, 0, 0, 0],  # Taunt Attack
            '4': [0, 0, 0, 1, 0, 0],  # Pickup/Throw
            '5': [0, 0, 0, 0, 1, 0],  # Dash/Dodge
            '6': [0, 0, 0, 0, 0, 1],  # Taunt
        }

        specials = {
            '7': [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Jump
        }

        key = cv2.waitKey(1000) & 0xFF
        if key != 255:
            key = chr(key)
            if key in directions:
                action[:4] = directions[key]
            if key in actions:
                action[4:] = actions[key]
            if key in specials:
                action = specials[key]

        return action


def run_match(agent_1: Agent | partial,
              agent_2: Agent | partial,
              max_timesteps=30*90,
              video_path: Optional[str]=None,
              agent_1_name: Optional[str]=None,
              agent_2_name: Optional[str]=None,
              resolution=CameraResolution.LOW,
              reward_manager: Optional[RewardManager]=None
              ) -> MatchStats:
    # Initialize env
    env = WarehouseBrawl(resolution=resolution)
    observations, infos = env.reset()
    obs_1 = observations[0]
    obs_2 = observations[1]

    if reward_manager is not None:
        reward_manager.reset()
        reward_manager.subscribe_signals(env)

    if agent_1_name is None:
        agent_1_name = 'agent_1'
    if agent_2_name is None:
        agent_2_name = 'agent_2'

    env.agent_1_name = agent_1_name
    env.agent_2_name = agent_2_name

    if video_path is None:
        print("video_path=None -> Not rendering")
    else:
        print(f"video_path={video_path} -> Rendering")

    # If partial
    if callable(agent_1):
        agent_1 = agent_1()
    if callable(agent_2):
        agent_2 = agent_2()

    # Initialize agents
    if not agent_1.initialized: agent_1.get_env_info(env)
    if not agent_2.initialized: agent_2.get_env_info(env)
    # 596, 336

    for _ in range(max_timesteps):
        # actions = {agent: agents[agent].predict(None) for agent in range(2)}

        # observations, rewards, terminations, truncations, infos

        full_action = {
            0: agent_1.predict(obs_1),
            1: agent_2.predict(obs_2)
        }

        observations, rewards, terminated, truncated, info = env.step(full_action)
        obs_1 = observations[0]
        obs_2 = observations[1]

        # # MY EDIT HERE
        # reward = reward_manager.process(env, 1 / 30.0)

        # env.logger[0]["reward"] = rewards

        if reward_manager is not None:
            reward_manager.process(env, 1 / env.fps)
            print(env.logger)

        if video_path is not None:
            img = env.render()
            cv2.imshow('Live Match', img)  # Display the frame
            del img

        if terminated or truncated:
            break
        # env.show_image(img)

    if video_path is not None:
        cv2.destroyAllWindows()  # Close the display window

    env.close()

    # visualize
    # Video(video_path, embed=True, width=800) if video_path is not None else None
    player_1_stats = env.get_stats(0)
    player_2_stats = env.get_stats(1)
    match_stats = MatchStats(
        match_time=env.steps / env.fps,
        player1=player_1_stats,
        player2=player_2_stats,
        player1_result=Result.WIN if player_1_stats.lives_left > player_2_stats.lives_left else Result.LOSS
    )

    del env

    return match_stats


if __name__ == "__main__":
    random_agent = MyAgent() # TransformerAgent("Gen0001/model_ivy_defensive/Policy")
    clockwork_agent = TransformerAgent("C:\Files\PythonProjects\ParrallelRL\AI_conpetition_eval_space\EvoRLv4.4\dh2010pc45\Gen0003\SLOT_dh2010pc45/Policy")  #
    reward_manager = RewardManager()
    match_stats = run_match(random_agent, clockwork_agent, max_timesteps=5120, video_path="vid.mp4",
                            reward_manager=reward_manager)
    print(match_stats)
