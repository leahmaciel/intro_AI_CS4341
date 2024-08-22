#Leah Maciel CS 4341 HW 2 Problem 4.3
import math 
import random
#algorithms based on pseudocode given during lecture and the textbook chapters

"""
traveling salesperson problem:
given a list/graph of cities and the distances between them,
we want to find the shortest path to visit all cities once and return to the original city
"""

"""
To set up the problem we need to define the initial city the salesperson starts in and distances/ connections between the cities
The program should return the path the salesperson takes, checking that it is the shortest path, all cities are visited only once, and the salesperson ends at the starting city

To solve the problem I need to:
- to create a map of cities and coordinates
- calculate distances between city pairs
- create a hill climbing or genetic algorithm that will select a path
- keep track of the path taken and total distance
- check if the current path is the shortest and return
"""

# city generation
# this is an example of cities and their locations. I randomly picked having 6 cities and their coordinates
# but this could easily be adjusted to a real world scenario
cities_coords = {  #each x,y coordinate represents the 2D location of 1 city
    'A': (0, 5),
    'B': (1, 3),
    'C': (2, 3),
    'D': (4, 4),
    'E': (4, 3),
    'F': (5, 3)
}

# Function to calculate the Euclidean (straight line) distance between the cities
# but could be modified in a real world scenario to use the actual distances between cities
def find_euclid_distance(i,j): #takes in 2 cities i and j
    x1, y1 = i
    x2, y2 = j
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) #equation for Euclidean distance

#function to calculate the total distance of a solution, including the fact that we need to return to the starting city
def calc_total_distance(path):
    total_distance = 0
    for i in range(len(path) -1): #calculates the distance between each city pair in the path
        total_distance += find_euclid_distance(cities_coords[path[i]], cities_coords[path[i+1]]) #the Euclidean distance between the 2 given cities
        total_distance += find_euclid_distance(cities_coords[path[-1]], cities_coords[path[0]]) #the distance from the last city back to the beginning
    return total_distance

#function to generate a random solution/ path of cities given the city coordinates
#used to generate an initial solution for hill climbing and the population for genetic algorithm
def generate_path(cities_coords):
    path = list(cities_coords.keys())
    random.shuffle(path)  #randomly shuffles the list of cities
    return path


#part a- implement and test a hill-climbing method to solve TSPs.
"""
a hill climbing search returns the state that is the global/local maximum
greedy search that picks the best choice from neigbors to keep on moving up the hill
"""

starting_solution = generate_path(cities_coords) #a starting solution for us to compare neighbors to
def hill_climbing(path):
    current_path = path
    for k in range(100): #controlling how many times it runs. I picked 100 since I'm only using 6 cities but it could be adjusted based on the path size
        path_neighbor = current_path.copy() #creating a copy of the path that we can modify and see if it improves
        i, j = random.sample(range(1, len(path_neighbor)), 2) #randomly select 2 cities to change in the path as long as they aren't the starting city
        path_neighbor[i], path_neighbor[j] = path_neighbor[j], path_neighbor[i]  #swapping the 2 cities
        if calc_total_distance(path_neighbor) <= calc_total_distance(current_path): #checking if the neighboring path distance is less than or equal to the current distance
            current_path = path_neighbor  #updating the current path to be the neighbor path if its better

    return current_path, calc_total_distance(current_path)

path_solution, path_distance = hill_climbing(starting_solution)
starting_path_distance = calc_total_distance(starting_solution)
starting_solution.append(starting_solution[0])  #appending the final city to the list. When calculating the distance I include the ending city, but it isn't written in the path
path_solution.append(path_solution[0])

print("starting path: ", starting_solution)
print("starting distance: ", starting_path_distance)
print("TSP hill climbing solution: ", path_solution)
print("TSP hill climbing path distance: ", path_distance)
print()


#part b- Repeat part (a) using a genetic algorithm instead of hill climbing.
"""
a genetic algorithm works by representing potential solutions in a string
- in this case I could list the cities in a string
2 parents (solutions) are combined with the potential for mutations
the resulting child string is a new potential solution

Just as before I need a fitness function to score every parent and child (distance calculation)
I also want to make sure that I am keeping the same starting city and the salesperson returns back there
"""
#I picked 100 to be consistent with the amount of times I run the hill climbing algorithm
#I know the mutation rate should be a small probability so I set it to 0.01
# These could be changed based on more information about the starting map or how likely the user wants mutations to be
pop_size = 100 #the starting size of the population- number of potential parents
mutation_rate = 0.01 #the probability that there will be a mutation after crossover happens
number_generations = 100 #the number of times I'm going to create children

#generate a list of paths that is the population size
def generate_population(cities_coords, pop_size):
    population = []
    for k in range(pop_size):
        path = generate_path(cities_coords)
        population.append(path)
    return population

#function to reproduce and create a child- where 2 parents (x and y) combine to generate a child path and potentially have a mutation
def reproduce(x, y, mutation_rate):
    crossover_point = random.randint(1, len(x) - 1) #selecting a position between the first and last cities in the path to be the crossover point
    child = [None] * len(x) #generating a child that is the correct path length, but is empty
    for i in range(len(x)): #cycling through the child. For each index the child will either get the city from parent x or parent y
        if i < crossover_point:
            child[i] = x[i] #child gets city from parent x if the index is before the crossover point
        else:  #child gets city from parent y if the index is at or after the crossover point
            parent_city = y[i]
            while parent_city in child: #first we ensure the parent y city is not already in the child
                parent_city = y[x.index(parent_city)] #replace the parent y city with the parent x city
            child[i] = parent_city
    for i in range(len(child)): #loop to create a mutation
        if random.random() <= mutation_rate:  # generate a random float. If it less than or equal to the mutation rate a mutation occurs
            j = random.randint(0,len(child) - 1)  # generate a random integer that corresponds to the index of the city that will be mutated
            child[i], child[j] = child[j], child[i]  # switch the 2 cities
    return child

#function for the genetic algorithm that utilizes the reproduce function
def genetic_algorithm(cities_coords, population_size, mutation_rate, num_generations):
    population = generate_population(cities_coords, population_size) #creating a population of the input size, each item in the population is a random solution
    for generation in range(num_generations): #running the function for the inputted number of generations (times)
        children = [] #list to hold the created children
        while len(children) < population_size:
            parent1, parent2 = random.sample(population, 2) #randomly selecting 2 parents from the population
            child = reproduce(parent1, parent2, mutation_rate) #reproducing to create a child
            children.append(child) #adding the newly created child to the list of children
        population = children #updating the population to be the children that were just created. That way in the next generation the parents are the children from the previous generation
    best_path = [None] #creating a variable to hold the best path solution
    best_distance = float('inf') #variable to compare the path distance to. Starting at infinity as a max distance
    for path in population:
        if calc_total_distance(path) <= best_distance: #comparing the distance for each path in the population (final generation children) to the current best distance
            best_distance = calc_total_distance(path) #updating the best distance and path if the current path is better
            best_path = path
    return best_path, best_distance


GA_path_solution, GA_path_distance = genetic_algorithm(cities_coords, pop_size, mutation_rate, number_generations)
GA_path_solution.append(GA_path_solution[0]) #appending the final city to the list. When calculating the distance I include the ending city, but it isn't written in the path
print("TSP genetic algorithm solution: ", GA_path_solution)
print("TSP genetic algorithm path distance: ", GA_path_distance)