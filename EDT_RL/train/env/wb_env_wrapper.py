from torchrl.envs import GymWrapper
import numpy as np
from .wb_env import *
from gymnasium.spaces import Discrete
from tensordict import TensorDict, TensorDictBase
from datetime import datetime


class WBGymWrapper(GymWrapper):
    def __init__(self, env: Any = None, categorical_action_encoding=False, flatten_action_space=True, **kwargs):
        if env is None:
            raise NotImplementedError()

        # Write video
        self.writer = None
        self.use_writer = False  # Disable video writing
        self.use_cv2 = False
        if self.use_writer:
            import skvideo.io
        if self.use_cv2:
            import cv2
        self.writer_frame_skip = 1
        self.frames_written = 0
        self.writer_cool_down = 1024
        self.writer_cool_down_counter = 0

        self.flatten_action_space = flatten_action_space
        if self.flatten_action_space:
            # Ensure the environment has a Box action space
            assert hasattr(env.action_space, 'shape'), "Environment must have a Box action space to flatten."

            self.original_action_space = env.action_space

            # Flattened action space: 8 directions * 5 actions
            env.action_space = Discrete(40)

            # Define the action mapping
            # Define the action mapping
            self.directions = [
                [1, 0, 0, 0],  # W
                [0, 1, 0, 0],  # A
                [0, 0, 1, 0],  # S
                [0, 0, 0, 1],  # D
                [1, 1, 0, 0],  # W + A
                [1, 0, 0, 1],  # W + D
                [0, 1, 1, 0],  # A + S
                [0, 0, 1, 1],  # S + D
            ]

            # ["W", "A", "S", "D", "space", 'h', 'l', 'j', 'k', 'g']

            self.actions = [
                [0, 0, 0, 0, 0, 0],  # No action
                [1, 0, 0, 0, 0, 0],  # Jump
                # [0, 1, 0, 0, 0, 0],  # Pickup/Throw
                [0, 0, 1, 0, 0, 0],  # Dodge
                [0, 0, 0, 1, 0, 0],  # Light Attack
                [0, 0, 0, 0, 1, 0],  # Heavy Attack
                # [0, 0, 0, 0, 0, 1],  # Taunt (Doesn't do anything?)
            ]

            # obs = [
            #     x_norm,  # 1. X Position (Clamped between -18 and 18)
            #     y_norm,  # 2. Y Position (Clamped between -7 and 7)
            #     vx_norm,  # 3. X Velocity (Clamped between -10 and 10)
            #     vy_norm,  # 4. Y Velocity (Clamped between -10 and 10)
            #     1.0 if self.facing == Facing.RIGHT else 0.0,  # 5. Facing Direction (1.0 if right, 0.0 if left)
            #     1.0 if self.is_on_floor() else 0.0,  # 6. Grounded Status (1.0 if on floor, 0.0 otherwise)
            #     0.0 if self.is_on_floor() else 1.0,  # 7. In-Air Status (1.0 if in air, 0.0 if grounded)
            #     float(self.state.jumps_left) if hasattr(self.state, 'jumps_left') else 0.0,  # 8. Jumps Left
            #     float(self.state_mapping.get(type(self.state).__name__, 0)),  # 9. Current State Index
            #     float(self.state.jumps_left) if hasattr(self.state, 'recoveries_left') else 0.0,
            #     # 10. Recoveries Left (note: might be incorrect in code)
            #     float(self.state.dodge_timer) if hasattr(self.state, 'dodge_timer') else 0.0,  # 11. Dodge Timer
            #     float(self.state.stun_frames) if hasattr(self.state, 'stun_frames') else 0.0,  # 12. Stun Frames
            #     float(self.damage) / 700.0,  # 13. Damage (Normalized)
            #     float(self.stocks),  # 14. Stocks (expected between 0 and 3)
            #     float(self.state.move_type) if hasattr(self.state, 'move_type') else 0.0  # 15. Move Type
            # ]

        super().__init__(env=env, categorical_action_encoding=categorical_action_encoding, **kwargs)
        env.action_space = self.original_action_space

        # self._env.max_timesteps = 300

    def reset(
        self,
        tensordict: Optional[TensorDictBase] = None,
        **kwargs,
    ) -> TensorDictBase:
        return super(WBGymWrapper, self).reset(tensordict, **kwargs)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.flatten_action_space:
            tensordict = self._discrete_to_box(tensordict)

        action = tensordict.get(self.action_key)
        if self._convert_actions_to_numpy:
            action = self.read_action(action)

        action = action.tolist()  # WB gym uses list

        reward = 0
        for _ in range(self.wrapper_frame_skip):
            (
                obs,
                _reward,
                terminated,
                truncated,
                done,
                info_dict,
            ) = self._output_transform(self._env.step(action))

            if _reward is not None:
                reward = reward + _reward

            terminated, truncated, done, do_break = self.read_done(
                terminated=terminated, truncated=truncated, done=done
            )
            if do_break:
                break

        reward = self.read_reward(reward)
        obs_dict = self.read_obs(obs)
        obs_dict[self.reward_key] = reward

        # if truncated/terminated is not in the keys, we just don't pass it even if it
        # is defined.
        if terminated is None:
            terminated = done
        if truncated is not None:
            obs_dict["truncated"] = truncated
        obs_dict["done"] = done
        obs_dict["terminated"] = terminated
        validated = self.validated
        if not validated:
            tensordict_out = TensorDict(obs_dict, batch_size=tensordict.batch_size)
            if validated is None:
                # check if any value has to be recast to something else. If not, we can safely
                # build the tensordict without running checks
                self.validated = all(
                    val is tensordict_out.get(key)
                    for key, val in TensorDict(obs_dict, []).items(True, True)
                )
        else:
            tensordict_out = TensorDict._new_unsafe(
                obs_dict,
                batch_size=tensordict.batch_size,
            )
        if self.device is not None:
            tensordict_out = tensordict_out.to(self.device)

        if self.info_dict_reader and (info_dict is not None):
            if not isinstance(info_dict, dict):
                warnings.warn(
                    f"Expected info to be a dictionary but got a {type(info_dict)} with values {str(info_dict)[:100]}."
                )
            else:
                for info_dict_reader in self.info_dict_reader:
                    out = info_dict_reader(info_dict, tensordict_out)
                    if out is not None:
                        tensordict_out = out

        # Write to video
        if self.use_writer:
            if self.writer is not None:
                if self.frames_written % self.writer_frame_skip == 0:
                    img = self._env.render()
                    self.writer.writeFrame(img)
                    del img
                self.frames_written += 1
            if self.frames_written > 512 or self.writer is None:
                if self.writer is not None:
                    self.writer.close()
                self.frames_written = 0
                file_name = datetime.now().strftime('%Y%m%d_%H_%M_%S_') + ".mp4"
                if os.path.exists(file_name):
                    self.use_writer = False
                self.writer = skvideo.io.FFmpegWriter(file_name, outputdict={
                    '-vcodec': 'libx264',  # Use H.264 for Windows Media Player
                    '-pix_fmt': 'yuv420p',  # Compatible with both WMP & Colab
                    '-preset': 'fast',  # Faster encoding
                    '-crf': '20',  # Quality-based encoding (lower = better quality)
                    '-r': str(30)  # Frame rate
                })
        elif self.use_cv2:
            # Display frame using cv2
            img = self._env.render()
            cv2.imshow('Frame', img)
            cv2.waitKey(1)  # Display the frame for 1 ms
            del img

        return tensordict_out

    def _discrete_to_box(self, tensordict):
        action = tensordict.get('action')
        action_int = int(action.item())

        num_directions = len(self.directions)
        num_actions = len(self.actions)

        direction_idx = action_int // num_actions
        action_idx = action_int % num_actions

        binary_action = self.directions[direction_idx] + self.actions[action_idx]

        tensordict.set('action', np.array(binary_action, dtype=np.float32))

        return tensordict
