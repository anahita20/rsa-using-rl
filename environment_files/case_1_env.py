import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Case1Env(gym.Env):

    def __init__(self) -> None:

        self.observation_space = spaces.Dict(
            {
                "edge_capacities": spaces.Box(-1, 20, shape=(15, 10), dtype=np.float64), # high = max holding time for request, shape = (total edges, total capacity of each edge)
                "request": spaces.Dict({
                    "src": spaces.Box(0, 12, shape=(1,), dtype=int), # any node
                    "dest": spaces.Box(0, 12, shape=(1,), dtype=int), # any node
                    "ht": spaces.Box(10, 20, shape=(1,), dtype=int), # min_ht 10 and max_ht 20 
                })
                
            }
        )

        # define action space - all paths
        # CASE 1: 7 possible paths and last action is blocking
        self.action_space = spaces.Discrete(8)

        self.num_req = 0

        self.blocked = 0

    # generate a new request with source = node (id=7), dest = node (id=1)
    def _generate_req(self):
        min_ht = 10
        max_ht = 20
        ht = np.random.randint(min_ht, max_ht)

        new_req = {
            "src" : np.array([7]),
            "dest": np.array([1]),
            "ht": np.array([ht]),
        }
        
        return new_req

    
    def _get_obs(self):
        return {
            "edge_capacities": self._edges_state,
            "request": self._current_req
        }
    
    def _get_smallest_index_for_given_edge(self, edge):
        edge_index = self._get_edge_to_edgestate_index_mapping(edge)

        for idx, val in enumerate(self._edges_state[edge_index]):
            if val == -1:
                return idx
            
        return -1
    
    def _get_utilization_of_edge(self, edge):
        edge_index = self._get_edge_to_edgestate_index_mapping(edge)
        occupied_slots = len([slot for slot in self._edges_state[edge_index] if slot != -1])

        return occupied_slots


    def _generate_all_paths(self):
        # CASE 1: hardcoded all paths between source=7 and dest=1
        return [[(7, 0), (0, 2), (2, 1)], 
                [(7, 0), (0, 11), (11, 9), (9, 5), (5, 6), (6, 12), (12, 4), (4, 1)], 
                [(7, 0), (0, 11), (11, 12), (12, 4), (4, 1)], 
                [(7, 6), (6, 5), (5, 9), (9, 11), (11, 0), (0, 2), (2, 1)], 
                [(7, 6), (6, 5), (5, 9), (9, 11), (11, 12), (12, 4), (4, 1)], 
                [(7, 6), (6, 12), (12, 4), (4, 1)], 
                [(7, 6), (6, 12), (12, 11), (11, 0), (0, 2), (2, 1)]]

    def _get_edge_to_edgestate_index_mapping(self, edge):
        # mapping of every edge in network to an index in edge state in state space
        mapping = {
            "(0, 2)": 0,
             "(2, 0)": 0,
            "(0, 11)": 1, 
             "(11, 0)": 1, 
            "(0, 7)": 2, 
             "(7, 0)": 2, 
            "(1, 2)": 3, 
             "(2, 1)": 3, 
            "(1, 4)": 4, 
             "(4, 1)": 4, 
            "(3, 12)": 5, 
             "(12, 3)": 5,
            "(4, 12)": 6, 
             "(12, 4)": 6, 
            "(5, 9)": 7, 
             "(9, 5)": 7, 
            "(5, 6)": 8, 
             "(6, 5)": 8, 
            "(6, 12)": 9, 
             "(12, 6)": 9, 
            "(6, 7)": 10, 
             "(7, 6)": 10, 
            "(8, 9)": 11, 
             "(9, 8)": 11, 
            "(9, 11)": 12, 
             "(11, 9)": 12, 
            "(10, 11)": 13, 
             "(11, 10)": 13, 
            "(11, 12)": 14, 
             "(12, 11)": 14, 
        }

        return mapping[edge]
    
    # to update holding times of every request in network
    def _update_holding_times(self):
        for i in range(len(self._edges_state)):
            for j in range(len(self._edges_state[0])):
                if self._edges_state[i][j] != -1:
                    self._edges_state[i][j] -= 1
                    # if holding time becomes 0, change it to -1 to indicate empty slot
                    if self._edges_state[i][j] == 0:
                        self._edges_state[i][j] = -1

    # to get current utilization of every edge
    def _get_utilization_per_link(self):
        link_utilizations = {}
        total_capacity = 10

        for i in range(len(self._edges_state)):
            occupied_slots = sum(1 for slot in self._edges_state[i] if slot != -1)
            link_utilizations[i] = occupied_slots / total_capacity

        return link_utilizations
    
    # update state variable of utilization histories
    def _update_historical_utilizations(self, current_utilizations):
        for i, util in current_utilizations.items():
            self._historical_utilizations[i].append(util)

    # to get average utilizations per edge
    def _calculate_average_utilization_per_link(self, num_req):
        average_utilizations = {}
        for i, utilizations in self._historical_utilizations.items():
            average_utilizations[i] = sum(utilizations) / num_req if num_req > 0 else 0
        return average_utilizations
    
    # to get average network-wide utilization
    def _calculate_average_network_wide_utilization(self, average_utilizations):
        total_utilization = sum(average_utilizations.values())
        number_of_links = 15
        network_wide_utilization = total_utilization / number_of_links
        return network_wide_utilization

    # to get total current network utilization
    def _calculate_current_network_utilization(self):
        total_slots = 100
        occupied_slots = 0
        for edge_state in self._edges_state:
            occupied_slots += sum(1 for slot in edge_state if slot != -1)
        return (occupied_slots / total_slots) 


    # info contains current network utilization, total average network utilization, number of blocked requests
    def _get_info(self):
        current_utilization = self._calculate_current_network_utilization()

        link_utilizations = self._get_utilization_per_link()
        self._update_historical_utilizations(link_utilizations)
        avg_utilizations_per_link = self._calculate_average_utilization_per_link(self.num_req)
        total_average_network_util = self._calculate_average_network_wide_utilization(avg_utilizations_per_link)
        
        return {
            "current_total_network_utilization": current_utilization,
            "total_average_network_utilizations" : total_average_network_util,
            "blocked_req": self.blocked,
        }


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # to recover the original state (empty matrix)
        self._edges_state = np.full((15, 10), -1)

        # to generate a request
        self._current_req = self._generate_req()

        # to keep track of all utilizations of every edge
        self._historical_utilizations = {i: [] for i in range(15)}

        observation = self._get_obs()
        info = self._get_info()
        
        self.num_req = 0

        self.blocked = 0

        return observation, info

    def step(self, action):

        self.num_req += 1
        terminated = (self.num_req == 100)

        reward = 0

        all_paths = self._generate_all_paths()

        # if agent chooses to block req, negative reward
        if action == 7:
            reward = -10
            self.blocked += 1

        else:
            chosen_path = all_paths[action]

            # check if first edge has capacity for req
            first_edge = chosen_path[0]
            available_idx = self._get_smallest_index_for_given_edge(str(first_edge))

            # if doesn't have capacity, block req
            if available_idx == -1:
                reward = -10
                self.blocked += 1

            # if it does have capacity, set the capacities for whole path to holding time of req
            else:
                # calculate average utilization of all edges in chosen path
                avg_utilization = 0
                for edge in chosen_path:
                    edge_idx = self._get_edge_to_edgestate_index_mapping(str(edge))
                    curr_utilization = self._get_utilization_of_edge(str(edge))
                    avg_utilization += curr_utilization
                    # update edge state
                    self._edges_state[edge_idx][available_idx] = self._current_req["ht"][0]
                
                # calculate reward based on average utilization of path
                avg_utilization = avg_utilization / len(chosen_path)
                reward = int(round(10 - avg_utilization))
         

        # Now, update holding times for every other edge
        self._update_holding_times()


        # generate a new request for next round and new observations
        self._current_req = self._generate_req()
        observation = self._get_obs()


        info = self._get_info()
        return observation, reward, terminated, False, info

