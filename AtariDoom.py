import vizdoom as vzd
import gym
from gym.core import Env
from gym.envs.registration import EnvSpec
from gym.spaces import Discrete, Box
from gym import logger
import numpy as np
import cv2

class AtariDoom(Env):
    def __init__(self, cfg_config_path, window_visible=False):
        super(AtariDoom, self).__init__()
        game = vzd.DoomGame()
        game.load_config(cfg_config_path)
        game.set_window_visible(window_visible)
        game.init()
        self.game = game
        self.timestep = 0
        self.timeout = game.get_episode_timeout()
        # print(self.timeout)
        
        self.cfg_config_path = cfg_config_path
        # for e in dir(game):
        #     print(e)

    @property
    def spec(self):
        id = self.cfg_config_path.split('/')[-1].split('.')[0]
        id = "{}NoFrameskip-v4".format(id)
        timeout = self.game.get_episode_timeout()
        return EnvSpec(id, max_episode_steps=timeout)
    
    @property
    def action_space(self):         # The Space object corresponding to valid actions
        return Discrete(len(self.game.get_available_buttons()))

    @property
    def observation_space(self):    # The Space object corresponding to valid observations
        height = self.game.get_screen_height()
        width = self.game.get_screen_width()
        depth = self.game.get_screen_channels()
        low = 0
        high = 255
        return Box(low, high, shape=(height, width, depth), dtype=np.uint8)

    @property
    def actions(self):
        action_choice = len(self.game.get_available_buttons())
        actions = np.zeros((action_choice, action_choice), dtype=np.bool)
        actions[np.arange(action_choice), np.arange(action_choice)] = 1
        return actions

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (int): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.timestep += 1
        
        reward = self.game.make_action(self.actions[action].tolist())
        state = self.game.get_state()
        observation = np.swapaxes(state.screen_buffer, 0, 2)
        observation = np.swapaxes(observation, 0, 1)
        self.ob = observation
        # done = self.game.is_episode_finished()
        done = (self.timeout - 1 <= self.timestep)
        # print(done)
        # print(self.timestep)
        # if done:
        #     self.reset()
        info = dict()
        
        return observation, reward, done, info

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        # print("Reset")
        self.game.new_episode()
        self.timestep = 0
        state = self.game.get_state()
        observation = np.swapaxes(state.screen_buffer, 0, 2)
        observation = np.swapaxes(observation, 0, 1)
        self.ob = observation
        
        return observation

    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        if mode == 'human':
            img = self.ob
            img = img[::-1]
            cv2.imshow(self.spec.id, img)
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return self.ob
        else:
            raise NotImplementedError

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self.game.close()
        return

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.game.set_seed(seed)
        logger.info("Set environment seed to {}".format(seed))
        return

if __name__ == '__main__':
    atari_doom = AtariDoom('./ViZDoom/scenarios/basic.cfg')
    print(atari_doom.action_space)
    print(atari_doom.observation_space)
    print(atari_doom.reset().shape)

    import cv2
    img = atari_doom.reset()
    print(img)
    cv2.imwrite('img.png',img)
    # print(atari_doom.step(0))
    
