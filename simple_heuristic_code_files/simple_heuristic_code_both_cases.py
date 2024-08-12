import networkx as nx
import numpy as np
import csv

class Request:
    """This class represents a request. Each request is characterized by source and destination nodes and holding time (represented by an integer).

    The holding time of a request is the number of time slots for this request in the network. You should remove a request that exhausted its holding time.
    """
    def __init__(self, s, t, ht, id):
        self.s = s
        self.t = t
        self.ht = ht
        self.id = id

    def __str__(self) -> str:
        return f'req({self.s}, {self.t}, {self.ht}, {self.id})'
    
    def __repr__(self) -> str:
        return self.__str__()
        
    def __hash__(self) -> int:
        # used by set()
        return self.id


class EdgeStats:
    """This class saves all state information of the system. In particular, the remaining capacity after request mapping and a list of mapped requests should be stored.
    """
    def __init__(self, u, v, cap) -> None:
        self.id = (u,v)
        self.u = u
        self.v = v 
        # remaining capacity
        self.cap = cap 

        # spectrum state (a list of requests, showing color <-> request mapping). Each index of this list represents a color
        self.__slots = [None] * cap
        # a list of the remaining holding times corresponding to the mapped requests
        self.__hts = [-1] * cap # initialize with -1 to distinguish from expired requests

    def __str__(self) -> str:
        return f'{self.id}, cap = {self.cap}: {self.__slots}'
    
    def add_request(self, req: Request, color:int):
        """update self.__slots by adding a request to the specific color slot

        Args:
            req (Request): a request
            color (int): a color to be used for the request
        """
        self.__slots[color] = req
        self.__hts[color] = req.ht
        

    def remove_requests(self):
        """update self.__slots by removing the leaving requests based on self.__hts; Also, self.__hts should be updated in this function.
        """

        # decrement holding times for requests to simulate end of time round
        self.__hts = [ht - 1 if ht != -1 else -1 for ht in self.__hts]

        indices_to_remove = []
        for index, ht in enumerate(self.__hts):
            if ht == 0:
                indices_to_remove.append(index)

        for index in indices_to_remove:
            self.__slots[index] = None
            self.__hts[index] = -1


    def get_available_colors(self) -> list[int]:
        """return a list of integers available to accept requests
        """
        available_colors = []
        for index, slot in enumerate(self.__slots):
            if not slot:
                available_colors.append(index)

        return sorted(available_colors)
    

    def show_spectrum_state(self) -> int:
        """Come up with a representation to show the utilization state of a link (by colors) 
        
        Returns:
            int: utilization state for link at current time round
        """
        occupied_slots = len([slot for slot in self.__slots if slot is not None])
        utilization = occupied_slots/self.cap
        state = "Utilization state u_t" + str(self.id) + " = " + str(utilization) + " -> " + str(self.__slots)
        print(state)

        return utilization
    
    def calculate_utilization(self) -> float:
        occupied_slots = len([slot for slot in self.__slots if slot is not None])
        return occupied_slots / self.cap


def generate_requests(num_reqs: int, min_ht: int, max_ht: int, generate_src_dest: bool=False) -> list[Request]:
    """Generate a set of requests, given the number of requests and an optical network (topology)

    Args:
        num_reqs (int): the number of requests
        g (nx.Graph): network topology

    Returns:
        list[Request]: a list of request instances
    """
    requests = list()
    ids = 1
    source = 7
    destination = 1

    for _ in range(num_reqs):
        holding_time = np.random.randint(min_ht, max_ht)

        if generate_src_dest:
            source, destination = 0, 0
            while source == destination:
                source = np.random.randint(0, 13)
                destination = np.random.randint(0, 13)


        new_req = Request(source, destination, holding_time, ids)
        requests.append(new_req)
        ids += 1

    return requests

def simulate_network(G, requests, edges_stats):
    total_utilizations = []
    i = 0

    for req in requests:
        i+=1
        path = nx.shortest_path(G, source=req.s, target=req.t)
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        path_stats = [es for es in edges_stats if es.id in edges]
        print(path_stats)

        edges_stats= route(G, edges_stats, req)
        
        current_utilization = sum(es.calculate_utilization() for es in edges_stats) / len(edges_stats)
        total_utilizations.append(current_utilization)

        [es.show_spectrum_state() for es in edges_stats]

        # Remove requests that have expired
        for es in edges_stats:
            es.remove_requests()

    average_utilization = sum(total_utilizations) / len(total_utilizations)
    return average_utilization

def route(g: nx.Graph, estats: list[EdgeStats], req:Request) -> tuple[list[EdgeStats], bool]:
    """Use a routing algorithm to decide a mapping of requests onto a network topology. The results of mapping should be reflected. Consider available colors on links on a path. 

    Args:
        g (nx.Graph): a network topology
        req (Request): a request to map

    Returns:
        list[EdgeStats]: updated EdgeStats
        bool: indicating if request was blocked or not
    """

    path = nx.shortest_path(g, source=req.s, target=req.t)
    edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    edges = [tuple(sorted(t)) for t in edges]


    first_edge = [edge for edge in estats if edge.id == edges[0]][0]

    # check if first edge has capacity for this request
    available_colors = first_edge.get_available_colors()

    if len(available_colors) > 0:
        # add request to first index (color)
        curr_color = available_colors[0]
        # for all edges in the path, reserve the color and add request
        for n1, n2 in edges:
            curr_edge = [edge for edge in estats if edge.id == (n1,n2)][0]
            curr_edge.add_request(req, curr_color)
    else:
        print("Blocked the request due to lack of capacity!")

    return estats


if __name__ == "__main__":
    # 1. generate a network
    G = nx.read_gml("/Users/mrunal/Documents/ccs-graphs/nsfnet.gml", label='id')

    # CASE 1
    with open("utils_simple_case1.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode ID', 'Network Utilization'])

    for i in range(800):
        edges_stats = [EdgeStats(u, v, 10) for u, v in G.edges()]
        requests = generate_requests(100, 10, 20)
        avg_utilization = simulate_network(G, requests, edges_stats)

        with open("utils_simple_case1.csv", mode='a', newline='') as file:
              writer = csv.writer(file)
              writer.writerow([i+1, avg_utilization])


    # CASE 2
    with open("utils_simple_case2.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode ID', 'Network Utilization'])

    for i in range(400):
        edges_stats = [EdgeStats(u, v, 10) for u, v in G.edges()]
        requests = generate_requests(100, 10, 20, True)
        avg_utilization = simulate_network(G, requests, edges_stats)

        with open("utils_simple_case2.csv", mode='a', newline='') as file:
              writer = csv.writer(file)
              writer.writerow([i+1, avg_utilization])
              

    