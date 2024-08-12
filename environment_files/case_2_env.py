import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx

class Case2Env(gym.Env):

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
        # CASE 2: For all s-t pairs, atleast 1 path and maximum 8 possible paths and last action is blocking
        self.action_space = spaces.Discrete(9)

        self.num_req = 0

        self.blocked = 0

        self._G = nx.read_gml("/Users/mrunal/Documents/ccs-graphs/nsfnet.gml", label='id')


    # generate a new request with random src and dest
    def _generate_req(self):
        min_ht = 10
        max_ht = 20
        ht = np.random.randint(min_ht, max_ht)

        s, t = 0, 0
        while s == t:
            s = np.random.randint(0, 13)
            t = np.random.randint(0, 13)

        new_req = {
            "src" : np.array([s]),
            "dest": np.array([t]),
            "ht": np.array([ht]),
        }
        return new_req

    
    def _get_obs(self):
        return {
            "edge_capacities": self._edges_state,
            "request": self._current_req
        }
    
    # return all available empty slots in increasing order
    def _get_all_available_indices_for_given_edge(self, edge):
        edge_index = self._get_edge_to_edgestate_index_mapping(edge)

        available_indices = []

        for idx, val in enumerate(self._edges_state[edge_index]):
            if val == -1:
                available_indices.append(idx)
            
        return available_indices

    # to generate all paths between given src and dest
    def _generate_all_paths_src_dest(self, src, dest):
        all_paths = []

        for path in nx.all_simple_paths(self._G, source=src, target=dest):
            edges = [(path[i], path[i+1]) for i in range(len(path[:-1]))]
            all_paths.append(edges)

        return all_paths

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

        self._G = nx.read_gml("/Users/mrunal/Documents/ccs-graphs/nsfnet.gml", label='id')

        return observation, info

    def step(self, action):

        self.num_req += 1
        terminated = (self.num_req == 100)

        reward = 0

        current_src = self._current_req["src"][0]
        current_dest = self._current_req["dest"][0]

        # utilization of network before accomodating action
        before_action_network_utilization = self._calculate_current_network_utilization()

        all_paths = self._generate_all_paths_src_dest(current_src, current_dest)

        # if agent chooses to block req, give smaller negative reward
        if action == 8:
            reward = -4
            self.blocked += 1
        
        elif len(all_paths) <= action:
            # block with higher reward since agent chose a path that doesn't exist for current pair
            reward = -8
            self.blocked += 1

        else:
            chosen_path = all_paths[action]

            # get all available indices for first edge, then find the smallest available one out of those for all edges
            first_edge = chosen_path[0]
            available_indices = self._get_all_available_indices_for_given_edge(str(first_edge))
            selected_idx = None

            for idx in available_indices:
                found_smallest_idx = True
                for edge in chosen_path[1:]:
                    edge_idx = self._get_edge_to_edgestate_index_mapping(str(edge))
                    if self._edges_state[edge_idx][idx] != -1:
                        found_smallest_idx = False
                        break

                if found_smallest_idx:
                    selected_idx = idx
                    break


            # if doesn't have capacity, block req
            if len(available_indices) == 0 or selected_idx is None:
                # again penalize with higher reward since agent chose wrong path with no available space for req
                reward = -8
                self.blocked += 1

            # if it does have capacity, set the capacities for whole path to holding time of req
            else:

                for edge in chosen_path:
                    edge_idx = self._get_edge_to_edgestate_index_mapping(str(edge))

                    self._edges_state[edge_idx][selected_idx] = self._current_req["ht"][0]
                

                # set reward to difference in network utilizations after accomodating action
                after_action_network_utilization = self._calculate_current_network_utilization()
                reward = int(round((after_action_network_utilization - before_action_network_utilization) * 100))


        # Now, update holding times for every other edge
        self._update_holding_times()


        # generate a new request for next round and new observations
        self._current_req = self._generate_req()
        observation = self._get_obs()


        info = self._get_info()
        return observation, reward, terminated, False, info

