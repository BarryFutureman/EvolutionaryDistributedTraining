from .wb_env_utils import *

SelfAgent = TypeVar("SelfAgent", bound="Agent")


class Agent(ABC):

    def __init__(
            self,
            file_path: Optional[str] = None
        ):

        # If no supplied file_path, load from gdown (optional file_path returned)
        if file_path is None:
            file_path = self._gdown()

        self.file_path: Optional[str] = file_path
        self.initialized = False

    def get_env_info(self, env):
        # if isinstance(env, Monitor):
        #     self_env = env.env
        # else:
        #     self_env = env
        self_env = env
        self.observation_space = self_env.observation_space
        self.obs_helper = self_env.obs_helper
        self.action_space = self_env.action_space
        self.act_helper = self_env.act_helper
        self.env = env
        self._initialize()
        self.initialized = True

    def get_num_timesteps(self) -> int:
        if hasattr(self, 'model'):
            return self.model.num_timesteps
        else:
            return 0

    def update_num_timesteps(self, num_timesteps: int) -> None:
        if hasattr(self, 'model'):
            self.model.num_timesteps = num_timesteps

    @abstractmethod
    def predict(self, obs) -> spaces.Space:
        pass

    def save(self, file_path: str) -> None:
        return

    def reset(self) -> None:
        return

    def _initialize(self) -> None:
        """

        """
        return

    def _gdown(self) -> Optional[str]:
        """
        Loads the necessary file from Google Drive, returning a file path.
        Or, returns None, if the agent does not require loaded files.

        :return:
        """
        return

"""### Agent Classes"""

class ConstantAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = np.zeros_like(self.action_space.sample())
        return action

class RandomAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.action_space.sample()
        return action

"""## StableBaselines3 Integration

### Reward Configuration
"""

@dataclass
class RewTerm():
    """Configuration for a reward term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the reward signals as torch float tensors of
    shape (num_envs,).
    """

    weight: float = MISSING
    """The weight of the reward term.

    This is multiplied with the reward term's value to compute the final
    reward.

    Note:
        If the weight is zero, the reward term is ignored.
    """

    params: dict[str, Any] = field(default_factory=dict)
    """The parameters to be passed to the function as keyword arguments. Defaults to an empty dict.

    .. note::
        If the value is a :class:`SceneEntityCfg` object, the manager will query the scene entity
        from the :class:`InteractiveScene` and process the entity's joints and bodies as specified
        in the :class:`SceneEntityCfg` object.
    """

class RewardManager():
    """Reward terms for the MDP."""

    # (1) Constant running reward
    def __init__(self,
                 reward_functions: Optional[Dict[str, RewTerm]]=None,
                 signal_subscriptions: Optional[Dict[str, Tuple[str, RewTerm]]]=None) -> None:
        self.reward_functions = reward_functions
        self.signal_subscriptions = signal_subscriptions
        self.total_reward = 0.0
        self.collected_signal_rewards = 0.0

    def subscribe_signals(self, env) -> None:
        if self.signal_subscriptions is None:
            return
        for _, (name, term_cfg) in self.signal_subscriptions.items():
            getattr(env, name).connect(partial(self._signal_func, term_cfg))

    def _signal_func(self, term_cfg: RewTerm, *args, **kwargs):
        term_partial = partial(term_cfg.func, **term_cfg.params)
        self.collected_signal_rewards += term_partial(*args, **kwargs) * term_cfg.weight


    def process(self, env, dt) -> float:
        # reset computation
        reward_buffer = 0.0
        # iterate over all the reward terms
        if self.reward_functions is not None:
            for name, term_cfg in self.reward_functions.items():
                # skip if weight is zero (kind of a micro-optimization)
                if term_cfg.weight == 0.0:
                    continue
                # compute term's value
                value = term_cfg.func(env, **term_cfg.params) * term_cfg.weight
                # update total reward
                reward_buffer += value

        reward = reward_buffer + self.collected_signal_rewards
        self.collected_signal_rewards = 0.0

        self.total_reward += reward

        log = env.logger[0]
        log['reward'] = f'{reward_buffer:.3f}'
        log['total_reward'] = f'{self.total_reward:.3f}'
        env.logger[0] = log
        return reward

    def reset(self):
        self.total_reward = 0
        self.collected_signal_rewards

"""### Save, Self-play, and Opponents"""

class SaveHandlerMode(Enum):
    FORCE = 0
    RESUME = 1

class SaveHandler():
    """Handles saving.

    Args:
        agent (Agent): Agent to save.
        save_freq (int): Number of steps between saving.
        max_saved (int): Maximum number of saved models.
        save_dir (str): Directory to save models.
        name_prefix (str): Prefix for saved models.
    """

    # System for saving to internet

    def __init__(
            self,
            agent: Agent,
            save_freq: int=10_000,
            max_saved: int=20,
            run_name: str='experiment_1',
            save_path: str='checkpoints',
            name_prefix: str = "rl_model",
            mode: SaveHandlerMode=SaveHandlerMode.FORCE
        ):
        self.agent = agent
        self.save_freq = save_freq
        self.run_name = run_name
        self.max_saved = max_saved
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.mode = mode

        self.steps_until_save = save_freq
        # Get model paths from exp_path, if it exists
        exp_path = self._experiment_path()
        self.history: List[str] = []
        if self.mode == SaveHandlerMode.FORCE:
            # Clear old dir
            if os.path.exists(exp_path) and len(os.listdir(exp_path)) != 0:
                while True:
                    answer = input(f"Would you like to clear the folder {exp_path} (SaveHandlerMode.FORCE): yes (y) or no (n): ").strip().lower()
                    if answer in ('y', 'n'):
                        break
                    else:
                        print("Invalid input, please enter 'y' or 'n'.")

                if answer == 'n':
                    raise ValueError('Please switch to SaveHandlerMode.FORCE or use a new run_name.')
                print(f'Clearing {exp_path}...')
                if os.path.exists(exp_path):
                    shutil.rmtree(exp_path)
            else:
                print(f'{exp_path} empty or does not exist. Creating...')

            if not os.path.exists(exp_path):
                os.makedirs(exp_path)
        elif self.mode == SaveHandlerMode.RESUME:
            if os.path.exists(exp_path):
                # Get all model paths
                self.history = [os.path.join(exp_path, f) for f in os.listdir(exp_path) if os.path.isfile(os.path.join(exp_path, f))]
                # Filter any non .csv
                self.history = [f for f in self.history if f.endswith('.zip')]
                if len(self.history) != 0:
                    self.history.sort(key=lambda x: int(os.path.basename(x).split('_')[-2].split('.')[0]))
                    self.history = self.history[-max_saved:]
                    print(f'Best model is {self.history[-1]}')
                else:
                    print(f'No models found in {exp_path}.')
                    raise FileNotFoundError
            else:
                print(f'No file found at {exp_path}')


    def update_info(self) -> None:
        self.num_timesteps = self.agent.get_num_timesteps()

    def _experiment_path(self) -> str:
        """
        Helper to get experiment path for each type of checkpoint.

        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, self.run_name)

    def _checkpoint_path(self, extension: str = '') -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self._experiment_path(), f"{self.name_prefix}_{self.num_timesteps}_steps.{extension}")

    def save_agent(self) -> None:
        print(f"Saving agent to {self._checkpoint_path()}")
        model_path = self._checkpoint_path('zip')
        self.agent.save(model_path)
        self.history.append(model_path)
        if len(self.history) > self.max_saved:
            os.remove(self.history.pop(0))

    def process(self) -> bool:
        self.num_timesteps += 1

        if self.steps_until_save <= 0:
            # Save agent
            self.steps_until_save = self.save_freq
            self.save_agent()
            return True
        self.steps_until_save -= 1

        return False

    def get_latest_model_path(self) -> str:
        if len(self.history) == 0:
            return None
        return self.history[-1]

class SelfPlaySelectionMode(Enum):
    LATEST = 0
    ELO = 1

class SelfPlayHandler():
    """Handles self-play."""

    def __init__(self, agent_partial: partial, mode: SelfPlaySelectionMode=SelfPlaySelectionMode.LATEST):
        self.agent_partial = agent_partial
        self.mode = mode

    def get_opponent(self) -> Agent:
        assert self.save_handler is not None, "Save handler must be specified for self-play"

        if self.mode == SelfPlaySelectionMode.LATEST:
            # Get the best model from the save handler
            self.best_model = self.save_handler.get_latest_model_path()
            if self.best_model:
                try:
                    opponent = self.agent_partial(file_path=self.best_model)
                    opponent.get_env_info(self.env)
                    return opponent
                except FileNotFoundError:
                    print(f"Warning: Self-play file {self.best_model} not found. Defaulting to constant agent.")
                    opponent = ConstantAgent()
                    opponent.get_env_info(self.env)
            else:
                print("Warning: No self-play model saved. Defaulting to constant agent.")
                opponent = ConstantAgent()
                opponent.get_env_info(self.env)

        elif self.mode == SelfPlaySelectionMode.ELO:
            raise NotImplementedError

        return opponent

@dataclass
class OpponentsCfg():
    """Configuration for opponents.

    Args:
        swap_steps (int): Number of steps between swapping opponents.
        opponents (dict): Dictionary specifying available opponents and their selection probabilities.
    """
    swap_steps: int = 10_000
    opponents: dict[str, Any] = field(default_factory=lambda: {
                'random_agent': (0.8, partial(RandomAgent)),
                'constant_agent': (0.2, partial(ConstantAgent)),
                #'recurrent_agent': (0.1, partial(RecurrentPPOAgent, file_path='skibidi')),
            })

    def validate_probabilities(self) -> None:
        total_prob = sum(prob if isinstance(prob, float) else prob[0] for prob in self.opponents.values())

        if abs(total_prob - 1.0) > 1e-5:
            print(f"Warning: Probabilities do not sum to 1 (current sum = {total_prob}). Normalizing...")
            self.opponents = {
                key: (value / total_prob if isinstance(value, float) else (value[0] / total_prob, value[1]))
                for key, value in self.opponents.items()
            }

    def process(self) -> None:
        pass

    def on_env_reset(self) -> Agent:

        agent_name = random.choices(
            list(self.opponents.keys()),
            weights=[prob if isinstance(prob, float) else prob[0] for prob in self.opponents.values()]
        )[0]

        # If self-play is selected, return the trained model
        print(f'Selected {agent_name}')
        if agent_name == "self_play":
            selfplay_handler: SelfPlayHandler = self.opponents[agent_name][1]
            return selfplay_handler.get_opponent()
        else:
            # Otherwise, return an instance of the selected agent class
            opponent = self.opponents[agent_name][1]()

        opponent.get_env_info(self.env)
        return opponent

"""### Self-Play Warehouse Brawl"""

class SelfPlayWarehouseBrawl(gymnasium.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 reward_manager: Optional[RewardManager]=None,
                 opponent_cfg: OpponentsCfg=OpponentsCfg(),
                 save_handler: Optional[SaveHandler]=None,
                 render_every: int | None = None,
                 resolution: CameraResolution=CameraResolution.LOW):
        """
        Initializes the environment.

        Args:
            reward_manager (Optional[RewardManager]): Reward manager.
            opponent_cfg (OpponentCfg): Configuration for opponents.
            save_handler (SaveHandler): Configuration for self-play.
            render_every (int | None): Number of steps between a demo render (None if no rendering).
        """
        super().__init__()

        self.reward_manager = reward_manager

        self.save_handler = save_handler
        self.opponent_cfg = opponent_cfg
        self.render_every = render_every
        self.resolution = resolution

        self.games_done = 0


        # Give OpponentCfg references, and normalize probabilities
        self.opponent_cfg.env = self
        self.opponent_cfg.validate_probabilities()

        # Check if using self-play
        if (selfplay_data := self.opponent_cfg.opponents.get('self_play')) is not None:
            assert self.save_handler is not None, "Save handler must be specified for self-play"

            # Give SelfPlayHandler references
            selfplay_handler: SelfPlayHandler = selfplay_data[1]
            selfplay_handler.save_handler = self.save_handler
            selfplay_handler.env = self

        self.best_model = None

        self.raw_env = WarehouseBrawl(resolution=resolution, train_mode=True)
        # My edit here (Subscribe on env creation):
        self.reward_manager.subscribe_signals(self.raw_env)
        self.action_space = self.raw_env.action_space
        self.act_helper = self.raw_env.act_helper
        self.observation_space = self.raw_env.observation_space
        self.obs_helper = self.raw_env.obs_helper

    def on_training_start(self):
        # Update SaveHandler
        if self.save_handler is not None:
            self.save_handler.update_info()

    def on_training_end(self):
        if self.save_handler is not None:
            self.save_handler.agent.update_num_timesteps(self.save_handler.num_timesteps)
            self.save_handler.save_agent()

    def step(self, action):

        full_action = {
            0: action,
            1: self.opponent_agent.predict(self.opponent_obs),
        }

        observations, rewards, terminated, truncated, info = self.raw_env.step(full_action)



        if self.save_handler is not None:
            self.save_handler.process()

        if self.reward_manager is None:
            reward = rewards[0]
        else:
            reward = self.reward_manager.process(self.raw_env, 1 / 30.0)

        return observations[0], reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset MalachiteEnv
        observations, info = self.raw_env.reset()

        self.reward_manager.reset()

        # Select agent
        new_agent: Agent = self.opponent_cfg.on_env_reset()
        if new_agent is not None:
            self.opponent_agent: Agent = new_agent
        self.opponent_obs = observations[1]


        self.games_done += 1
        #if self.games_done % self.render_every == 0:
            #self.render_out_video()

        return observations[0], info

    def render(self):
        img = self.raw_env.render()
        return img

    def close(self):
        pass


from tqdm import tqdm

def run_match(agent_1: Agent | partial,
              agent_2: Agent | partial,
              max_timesteps=30*90,
              video_path: Optional[str]=None,
              agent_1_name: Optional[str]=None,
              agent_2_name: Optional[str]=None,
              resolution = CameraResolution.LOW,
              reward_manager: Optional[RewardManager]=None
              ) -> MatchStats:
    # Initialize env
    env = WarehouseBrawl(resolution=resolution, train_mode=False)
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


    writer = None
    if video_path is None:
        print("video_path=None -> Not rendering")
    else:
        print(f"video_path={video_path} -> Rendering")
        # Initialize video writer
        writer = skvideo.io.FFmpegWriter(video_path, outputdict={
            '-vcodec': 'libx264',  # Use H.264 for Windows Media Player
            '-pix_fmt': 'yuv420p',  # Compatible with both WMP & Colab
            '-preset': 'fast',  # Faster encoding
            '-crf': '20',  # Quality-based encoding (lower = better quality)
            '-r': '30'  # Frame rate
        })

    # If partial
    if callable(agent_1):
        agent_1 = agent_1()
    if callable(agent_2):
        agent_2 = agent_2()

    # Initialize agents
    if not agent_1.initialized: agent_1.get_env_info(env)
    if not agent_2.initialized: agent_2.get_env_info(env)
    # 596, 336

    for _ in tqdm(range(max_timesteps), total=max_timesteps):
        # actions = {agent: agents[agent].predict(None) for agent in range(2)}

        # observations, rewards, terminations, truncations, infos

        full_action = {
            0: agent_1.predict(obs_1),
            1: agent_2.predict(obs_2)
        }

        observations, rewards, terminated, truncated, info = env.step(full_action)
        obs_1 = observations[0]
        obs_2 = observations[1]

        if reward_manager is not None:
            reward_manager.process(env, 1 / env.fps)

        if video_path is not None:
            img = env.render()
            writer.writeFrame(img)
            del img

        if terminated or truncated:
            break
        #env.show_image(img)

    if video_path is not None:
        writer.close()

    env.close()

    player_1_stats = env.get_stats(0)
    player_2_stats = env.get_stats(1)

    player_1_win = Result.WIN
    if player_1_stats.lives_left > player_2_stats.lives_left:
        player_1_win = Result.WIN
    elif player_1_stats.lives_left == player_2_stats.lives_left:
        if player_2_stats.damage_taken < player_2_stats.damage_taken:
            player_1_win = Result.WIN
        else:
            player_1_win = Result.LOSS
    else:
        player_1_win = Result.LOSS

    # visualize
    # Video(video_path, embed=True, width=800) if video_path is not None else None

    match_stats = MatchStats(
        match_time=env.steps / env.fps,
        player1=player_1_stats,
        player2=player_2_stats,
        player1_result=player_1_win
    )

    del env

    return match_stats

class ConstantAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = np.zeros_like(self.action_space.sample())
        return action

class RandomAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.action_space.sample()
        return action

class BasedAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)
        return action

class ClockworkAgent(Agent):

    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (30, []),
                (7, ['d']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (20, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
            ]
        else:
            self.action_sheet = action_sheet


    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)


        self.steps += 1  # Increment step counter
        return action

"""## Example Reward Functions
Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).
"""

def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Extract the used quantities (to enable type-hinting)
    obj: GameObject = env.objects[obj_name]

    # Compute the L2 squared penalty
    return (obj.body.position.y - target_height)**2

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0

    return reward * env.dt

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState]=BackDashState,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = 1 if isinstance(player.state, desired_state) else 0.0

    return reward * env.dt

def head_to_middle_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def head_to_opponent(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    else:
        return -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0