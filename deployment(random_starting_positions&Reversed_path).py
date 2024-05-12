import gym
from gym import spaces
import numpy as np
from controller import Supervisor
import json
import csv
import matplotlib.pyplot as plt
import os
GRID_SIZE = 1.0  
MAX_SPEED = 6.28   
ORIENTATION_TOLERANCE = 0.1
MOVEMENT_TOLERANCE = 0.1

class TiagoMazeSolver(gym.Env, Supervisor):
    def __init__(self,q_table_directory='data/q_tables'):
        super(TiagoMazeSolver, self).__init__()
        num_lidar_readings = 1
        lidar_range = 5
        self.step_counter = 0
        self.stuck_counter = 0
        self.stuck_threshold = 10  
        self.movement_threshold = 0.1
        self.visited_states = set()
        self.timestep = int(self.getBasicTimeStep())
        self.init_devices()
        self.robot_node = self.getFromDef("tiagopp")
        self.action_space = spaces.Discrete(3) 
        self.observation_space = spaces.Box(low=np.array([0.0]*num_lidar_readings + [-10/2, -10/2, -np.pi]),high=np.array([lidar_range]*num_lidar_readings + [10/2, 10/2, np.pi]),dtype=np.float64)
        self.q_table = np.zeros((100, 3)) 
        self.last_state_index = None
        self.epsilon = 1.0  
        self.epsilon_min = 0.05  
        self.epsilon_decay = 0.995  
        self.learning_rate = 0.8  
        self.discount_factor = 0.9
        self.wheel_diameter = 0.5831 
        self.linear_speed = 0.5       
        self.wheel_rotation_speed = self.linear_speed / (self.wheel_diameter / 2)
        self.q_table_path = os.path.join(self.q_table_directory, 'q_table.npy')
        self.params_path = os.path.join(self.q_table_directory, 'training_params.json')
        os.makedirs(self.q_table_directory, exist_ok=True)
    def init_devices(self):
        self.left_wheel_motor = self.getDevice('wheel_left_joint')
        self.right_wheel_motor = self.getDevice('wheel_right_joint')
        self.left_wheel_motor.setPosition(float('inf'))
        self.left_wheel_motor.setVelocity(0)
        self.right_wheel_motor.setPosition(float('inf'))
        self.right_wheel_motor.setVelocity(0)
        self.lidar = self.getDevice('Hokuyo URG-04LX-UG01')  
        self.lidar.enable(self.timestep)
        self.arm_left_1 = self.getDevice('arm_left_1_joint')
        self.arm_right_1 = self.getDevice('arm_right_1_joint')
        self.arm_left_2 = self.getDevice('arm_left_2_joint')
        self.arm_right_2 = self.getDevice('arm_right_2_joint')
        self.arm_left_3= self.getDevice('arm_left_3_joint')
        self.arm_right_3= self.getDevice('arm_right_3_joint')
        self.arm_left_4 = self.getDevice('arm_left_4_joint')
        self.arm_right_4 = self.getDevice('arm_right_4_joint')
        self.arm_left_5=self.getDevice('arm_left_5_joint')
        self.arm_right_5 = self.getDevice('arm_right_5_joint')
        self.arm_left_6 = self.getDevice('arm_left_6_joint')
        self.arm_right_6 = self.getDevice('arm_right_6_joint')
        self.arm_left_7= self.getDevice('arm_left_7_joint')
        self.arm_right_7= self.getDevice('arm_right_7_joint')
        self.torso_lift_joint=self.getDevice('torso_lift_joint')
        self.arm_left_1.setPosition(1.5)
        self.arm_right_1.setPosition(1.5)
        self.arm_left_2.setPosition(-1.11)
        self.arm_right_2.setPosition(-1.11)
        self.arm_left_3.setPosition(3.86)
        self.arm_right_3.setPosition(3.86)
        self.arm_left_4.setPosition(-0.32)
        self.arm_right_4.setPosition(-0.32)
        self.arm_left_5.setPosition(2.07)
        self.arm_right_5.setPosition(2.07)
        self.arm_left_6.setPosition(1.39)
        self.arm_right_6.setPosition(1.39)
        self.arm_left_7.setPosition(-2.07)
        self.arm_right_7.setPosition(-2.07)
        self.torso_lift_joint.setPosition(0)
        
    def get_robot_position(self):
        translation_field = self.robot_node.getField('translation')
        position = translation_field.getSFVec3f()
        return position[0], position[1]  

    def get_robot_orientation(self):
        rotation_field = self.robot_node.getField('rotation')
        current_orientation = rotation_field.getSFRotation()
        yaw_angle = current_orientation[3]
        return yaw_angle
    
    def get_lidar_data(self):
        lidar_values = self.lidar.getRangeImage() 
        front_distance = min(lidar_values)
        return front_distance
    
    def get_observation(self):
        position_x, position_y = self.get_robot_position()
        yaw = self.get_robot_orientation()  
        front_lidar_distance = self.get_lidar_data() 
        return np.array([front_lidar_distance, position_x, position_y, yaw])
    
    def step(self, action):
        self.perform_action(action)

        for _ in range(10):  
            if Supervisor.step(self,self.timestep) == -1:
                break  

            if self.is_action_completed(action):
                break  
        if self.is_stuck():
            done = True
        self.step_counter += 1 
        new_observation = self.get_observation()
        new_state_index = self.state_to_index(new_observation)
        done = self.is_done()
        reward = self.calculate_reward(new_observation, action)
        return new_observation, reward, done, new_state_index 
    def reset(self):
        self.step_counter = 0
  
        initial_positions = [
            [0.5, 4.5, 0],
            [4.5, 3.5, 0],
            [-4.5, -0.5, 0],
            [-3.5, -2.5, 0],
            [-3.5, -4.5, 0]
        ]

        initial_position = initial_positions[np.random.randint(0, len(initial_positions))]

        initial_rotation = [0, 0, 1, -np.pi/2] 
        position_field = self.robot_node.getField('translation')
        position_field.setSFVec3f(initial_position)
        rotation_field = self.robot_node.getField('rotation')
        rotation_field.setSFRotation(initial_rotation)
        self.previous_positions = []
        self.stuck_counter = 0
        self.visited_states.clear()
        observation = self.get_observation()
        initial_state_index = self.state_to_index(self.get_observation())
        self.last_state_index = initial_state_index
        return observation
    def move_forward(self):

        self.left_wheel_motor.setVelocity(self.wheel_rotation_speed)
        self.right_wheel_motor.setVelocity(self.wheel_rotation_speed)
        num_steps = int(6 / (self.timestep / 1000.0))  
        for _ in range(num_steps):
            if Supervisor.step(self, self.timestep) == -1:
                break
        self.left_wheel_motor.setVelocity(0)
        self.right_wheel_motor.setVelocity(0)
    def turn_left(self):
        initial_rotation=[0,0,1,0]
        rotation_field = self.robot_node.getField('rotation')
        rotation_field.setSFRotation(initial_rotation)
        self.move_forward()

    def turn_right(self):
        initial_rotation=[0,0,1,np.pi]
        rotation_field = self.robot_node.getField('rotation')
        rotation_field.setSFRotation(initial_rotation)
        self.move_forward()
    def set_orientation_for_forward_movement(self):
        forward_orientation = [0, 0, 1, -np.pi/2]
        rotation_field = self.robot_node.getField('rotation')
        rotation_field.setSFRotation(forward_orientation)
        self.move_forward()
    def perform_action(self, action):
 
        if action == 0:
            self.set_orientation_for_forward_movement()
        elif action == 1:
            self.turn_right()
        elif action == 2:
            self.turn_left()
        else:
            print("Received invalid action:", action)
            
    def state_to_index(self, state):
        position_x, position_y = state[1], state[2]
        #  start from (0,0)
        self.maze_width=10
        self.maze_height=10
        normalized_x = position_x + (self.maze_width / 2)
        normalized_y = position_y + (self.maze_height / 2)
        grid_x_index = int(normalized_x // GRID_SIZE)
        grid_y_index = int(normalized_y // GRID_SIZE)
        state_index = grid_y_index * self.maze_width + grid_x_index
        return state_index
    def is_action_completed(self, action):
        current_state_index = self.state_to_index(self.get_observation())
        if action in [0, 1, 2]:  
            return current_state_index != self.last_state_index

        return False
    def is_new_state(self, observation):
        state_index = self.state_to_index(observation)
        if state_index not in self.visited_states:
            self.visited_states.add(state_index)
            return True
        return False
    def is_revisiting_state(self, observation):
        state_index = self.state_to_index(observation)
        return state_index in self.visited_states


    def calculate_reward(self, new_observation, action):
        reward = 0
        if action == 0:  
            penalty = -0.2  
            reward += penalty
        elif action == 1:  
            penalty = -0.2  
            reward += penalty
        elif action == 2: 
            penalty = -0.2  
            reward += penalty
        front_lidar_distance = new_observation[0]
        collision_threshold = 0.21 
        if front_lidar_distance < collision_threshold:
            reward -= 0.7
        if self.is_new_state(new_observation):
            reward += 0.3
        if self.is_revisiting_state(new_observation):
            reward -= 0.1
        current_state_index = self.state_to_index(new_observation)
        if current_state_index == 8:
            reward += 250

        return reward

    def is_done(self):
        current_state_index = self.state_to_index(self.get_observation())
        if current_state_index == 8:
            return True
        front_lidar_distance = self.get_lidar_data()
        collision_threshold = 0.21 
        if front_lidar_distance < collision_threshold:
            return True
        max_steps = 6000 
        if self.step_counter >= max_steps:
            return True

        return False

    def is_stuck(self):
        current_position = self.get_robot_position()
        self.previous_positions.append(current_position)
        if len(self.previous_positions) > self.stuck_threshold:
            self.previous_positions.pop(0)
        if len(self.previous_positions) < self.stuck_threshold:
            return False
        distance_moved = np.linalg.norm(np.array(current_position) - np.array(self.previous_positions[0]))
        if distance_moved < self.movement_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        return self.stuck_counter >= self.stuck_threshold
    
    def save_model(self):
        np.save(self.q_table_path, self.q_table)  
        params = {
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }
        with open(self.params_path, 'w') as file:
            json.dump(params, file)  
    def load_model(self):
        if os.path.exists(self.q_table_path):
            self.q_table = np.load(self.q_table_path)
        if os.path.exists(self.params_path):
            with open(self.params_path, 'r') as file:
                params = json.load(file)
                self.epsilon = params['epsilon']
                self.learning_rate = params['learning_rate']
                self.discount_factor = params['discount_factor']

    def train(self, num_episodes, evaluation_interval=50, num_evaluation_episodes=10, q_table_file_path=None, params_file_path=None):
        
        if q_table_file_path is not None and params_file_path is not None:
            self.load_model(q_table_file_path, params_file_path)

        save_interval = 100  

        for episode in range(num_episodes):
            observation = self.reset()
            state = self.state_to_index(observation) 
            total_reward = 0
            done = False

            while not done:
                if np.random.rand() < self.epsilon:
                    action = self.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])
                observation, reward, done, _ = self.step(action)
                total_reward += reward
                new_state_index = self.state_to_index(observation)
                self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[new_state_index]) - self.q_table[state, action])
                state = new_state_index

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

           
            if (episode + 1) % evaluation_interval == 0:
                print("Evaluating...")
                self.evaluate(num_evaluation_episodes)

            if (episode + 1) % save_interval == 0:
                self.save_model(f'q_table_episode_{episode + 1}.npy', f'training_params_episode_{episode + 1}.json')

        self.save_model('q_table_final.npy', 'training_params_final.json')

    def evaluate(self, num_episodes):
        total_rewards = 0
        rewards_per_episode = []

        with open('evaluation_results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Reward'])

            for episode in range(num_episodes):
                state = self.reset()  
                state_index = self.state_to_index(state)

                done = False
                episode_reward = 0

                while not done:
                    action = np.argmax(self.q_table[state_index])
                    observation, reward, done, _ = self.step(action)
                    state = self.state_to_index(observation)
                    episode_reward += reward

                total_rewards += episode_reward
                rewards_per_episode.append(episode_reward)
                writer.writerow([episode + 1, episode_reward])

            average_reward = total_rewards / num_episodes
            writer.writerow(['Average', average_reward])

        
        plt.plot(rewards_per_episode)
        plt.title('Rewards per Episode during Evaluation')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()
        
    def run_episode(self, num_episodes=1):
        for episode in range(num_episodes):
            state = self.reset()
            state_index = self.state_to_index(state)
            done = False
            total_reward = 0

            while not done:
                action = np.argmax(self.q_table[state_index])
                observation, reward, done, _ = self.step(action)
                total_reward += reward
                state_index = self.state_to_index(observation)

            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    def get_inverse_action(self, action):
        if action == 0:  
            return [0, 0, 1, np.pi/2]  
        elif action == 1: 
            return [0, 0, 1, 0]  
        elif action == 2:  
            return [0, 0, 1, np.pi]  
        return None
    
    def reverse_actions(self):
        reversed_history = reversed(self.action_history)

        for state, action in reversed_history:
            self.perform_action(action, is_inverse=True)
            current_state = self.get_observation()
            current_state_index = self.state_to_index(current_state)
            if current_state_index == self.initial_state_index:
                self.stop_robot()  
                break
    def stop_robot(self):
        self.left_wheel_motor.setVelocity(0)
        self.right_wheel_motor.setVelocity(0)


if __name__ == "__main__":
    env = TiagoMazeSolver()
    num_episodes = 10
    env.load_model()  
    for episode in range(num_episodes):
        env.run_episode(1)  
        env.reverse_actions()