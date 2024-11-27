from ourhexenv import OurHexGame
from g03agent import G03Agent
from g03agent import G03Agent
import random

env = OurHexGame(board_size=11)
env.reset()

# player 1
gXXagent = G03Agent(env)
# player 2
gYYagent = G03Agent(env)

smart_agent_player_id = random.choice(env.agents)

done = False
while not done:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            break
        if agent == 'player_1':
            action = gXXagent.select_action(observation, reward, termination, truncation, info)
        else:
            action = gYYagent.select_action(observation, reward, termination, truncation, info)

        env.step(action)
        env.render()
