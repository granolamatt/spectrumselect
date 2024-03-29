import sys
import numpy as np
import pygame
from signals import SignalGenerator


class Environment:
    def __init__(self, screen, signal_generator, agent_size=10, render_on=False):
        self.screen = screen
        self.screen_width, self.screen_height = screen.get_size()
        self.render_on = render_on
        self.signal_generator = signal_generator
        self.agent_size = agent_size
        self.selbox = False

        self.agent_vertical_location = 0
        self.agent_horizontal_location = 400 - (self.agent_size / 2)
        self.signal_generator.agent_horizontal_location = self.agent_horizontal_location
        self.agentRect = pygame.Rect(self.agent_horizontal_location, self.agent_vertical_location, self.agent_size, self.screen_height)

    def reset(self):
        self.signal_generator.reset()
        if self.render_on:
            self.render()
        return self.signal_generator.state.astype(np.complex64).view(np.float32)   

    def render(self):
        self.screen.fill((0, 200, 255))
        pygame.draw.rect(self.screen, (0, 0, 255), self.agentRect)
        self.signal_generator.render()
        pygame.display.flip()
    
    def move_agent(self, action):
        
        if action == 1:
            if not self.selbox:
                self.signal_generator.setStart()
                self.selbox = True
        else:
            if self.selbox:
                self.signal_generator.setEnd()
                self.selbox = False
    
        
    def step(self, action):
        self.move_agent(action)
        #print('reward', reward)
        

        self.signal_generator.step()
        # if self.render_on:
        self.render()
        next_state = self.signal_generator.state.astype(np.complex64).view(np.float32) * 100
        reward = self.signal_generator.score
        #print("Reward is", reward)
        done = self.signal_generator.deaths >= 1
        return reward, next_state, done
    
if __name__ == "__main__":
    print("Starting signal game")
    pygame.init()
    screen_width = 1024
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()

    signal_generator = SignalGenerator(screen)
    env = Environment(screen=screen, render_on=True, signal_generator=signal_generator)

    state = env.reset()
    playing = True
    select_action = 0
    while playing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                select_action = 1
            elif event.type == pygame.KEYUP:
                select_action = 0

        reward, state, done = env.step(select_action)
        if done:
            playing = False
        # clock.tick(20)
    
    pygame.quit()
    sys.exit()
