import numpy as np

# Problem Parameters
NUM_CUSTOMERS = 10
NUM_VEHICLES = 3
VEHICLE_CAPACITY = 20
DEMAND = np.random.randint(1, 8, NUM_CUSTOMERS)  # Random demand for each customer
COORDINATES = np.random.rand(NUM_CUSTOMERS + 1, 2) * 100  # Depot + Customer locations

NUM_ANTS = 20
MAX_ITER = 50
ALPHA = 1  # Pheromone importance
BETA = 5   # Heuristic importance
RHO = 0.5  # Evaporation rate
INITIAL_PHEROMONE = 1.0

# Distance Matrix
def calculate_distance_matrix(coords):
    """Calculate Euclidean distance between all pairs of locations."""
    num_points = len(coords)
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            distance_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
    return distance_matrix

DISTANCE_MATRIX = calculate_distance_matrix(COORDINATES)

# Fitness Function
def calculate_route_cost(route, distance_matrix):
    """Calculate the total distance of a given route."""
    cost = 0
    for i in range(len(route) - 1):
        cost += distance_matrix[route[i]][route[i + 1]]
    cost += distance_matrix[route[-1]][route[0]]  # Return to depot
    return cost

# ACO Algorithm
def ant_colony_optimization():
    num_locations = NUM_CUSTOMERS + 1
    pheromones = np.full((num_locations, num_locations), INITIAL_PHEROMONE)
    best_route = None
    best_cost = float('inf')

    for iteration in range(MAX_ITER):
        all_routes = []
        all_costs = []

        for _ in range(NUM_ANTS):
            visited = [0]  # Start at the depot
            route = [0]  # Start at the depot
            demand_left = VEHICLE_CAPACITY

            while len(visited) < num_locations:
                probabilities = []
                current_location = route[-1]

                for next_location in range(num_locations):
                    if next_location not in visited and DEMAND[next_location - 1] <= demand_left:
                        pheromone = pheromones[current_location][next_location] ** ALPHA
                        heuristic = (1 / DISTANCE_MATRIX[current_location][next_location]) ** BETA
                        probabilities.append((next_location, pheromone * heuristic))
                    else:
                        probabilities.append((next_location, 0))

                if sum(prob[1] for prob in probabilities) == 0:
                    break

                probabilities = [(loc, prob / sum(p[1] for p in probabilities)) for loc, prob in probabilities]
                next_location = np.random.choice([loc for loc, prob in probabilities], p=[prob for loc, prob in probabilities])
                route.append(next_location)
                visited.append(next_location)
                demand_left -= DEMAND[next_location - 1] if next_location != 0 else 0

            all_routes.append(route)
            cost = calculate_route_cost(route, DISTANCE_MATRIX)
            all_costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_route = route

        # Update Pheromones
        pheromones *= (1 - RHO)  # Evaporation
        for route, cost in zip(all_routes, all_costs):
            for i in range(len(route) - 1):
                pheromones[route[i]][route[i + 1]] += 1 / cost

        print(f"Iteration {iteration + 1}: Best Cost = {best_cost}")

    return best_route, best_cost

# Run ACO
best_route, best_cost = ant_colony_optimization()

print("\nCustomer Demands:", DEMAND)
print("Best Route Found:", best_route)
print("Best Cost Achieved:", best_cost)
