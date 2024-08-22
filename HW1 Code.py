#Leah Maciel CS 4341 HW 1 Problem 2.9
"""
For the performance measure I decided to award 5 points for cleaning a room, and subtract one point for each movement
Since cleaning is the point of the vacuum I decided to value that action more, and therefore award more points for it
However, since movement is an inevitable but important cost I also decided to deduce points for each movement
Based on the needs of the designer or user these values could easily be modified to better fit the desired robot outcome
"""


#This function takes in the room names, if they are dirty or clean, and the room that the robot vacuum is starting in
#it returns the action sequence, final state of the rooms, and the performance measure
def action(state, starting):
    action_list = "" #creating string to store the list of actions

     #I'm using indexes for the rooms so that it can be generalized beyond 2 rooms labeled a and b
    all_rooms = list(state.keys())   #this turns the state keys (rooms) into a list so I can access a specific key based on index.
    #initalizing the index and location based on the starting location
    index = all_rooms.index(starting)
    current_location = starting
    performance_measure = 0 #initializing the performance measure
    action_list=action_list+ starting + "-" + state[current_location] + "; " #initalizing the action sequence to include starting in the given room
    
     #I decided to run actions until all rooms are clean. If I didn't have a way of stopping the robot would forever move between rooms, losing points
     #However, this could be changed if the problem wanted the robot to periodically check that rooms didn't become dirty again
    while "dirty" in state.values(): 
     if state[current_location].lower() =="dirty": #cleaning the dirty room and updating the performance measure
        state[current_location] = "clean"
        performance_measure +=5 
        action_list=action_list+ current_location + "-" + state[current_location] + "; " #updating the action sequence
     elif current_location == all_rooms[0]: #if the robot is currently in room a. This is based on the pseudocode but could be modified for other scenarios
        #Need to turn "right". I've defined this as updating the location by increasing the index and accessing 
        #the next room in the list. If the index is larger than the list length, then go to previous room (decrease index by 1)
        if index < len(all_rooms)-1:
         current_location= all_rooms[index + 1]
         index +=1
        else:
            current_location= all_rooms[index - 1]
            index -=1
        performance_measure -=1  #updating the performance measure
        action_list=action_list+ current_location + "-" + state[current_location] + "; " #updating the action sequence
     elif current_location == all_rooms[1]: #if the robot is currently in room b. This is based on the pseudocode but could be modified for other scenarios
        #Need to turn "left". I've defined this as updating the location by decreasing the index and accessing 
        #the previous room in the list. If the index is 0, increase index by one
        if index > 0:
         current_location= all_rooms[index - 1]
         index -=1
        else:
            current_location= all_rooms[index + 1]
            index +=1
        performance_measure -=1  #updating the performance measure
        action_list=action_list+ current_location + "-" + state[current_location] + "; " #updating the action sequence
    action_list= action_list + "end" #finishing the action sequence once all rooms are clean
    print("Sequence of actions:", action_list) #printing out the action sequence
    return state, performance_measure #returning the final state and performance measure
    
    

#testing all possible cases for 2 rooms. Each section will print out the scenario that is being tested, the sequence of actions,
#the final solution (which is that both room are clean), and the performance score
#it also keeps track of the average performance measure and prints it at the end
avg_performance_measure = 0 
case_count =0 
print ("Testing A clean, B clean, start in A:") 
result, performance_measure = action({"A":"clean", "B":"clean"} , "A")
print("Final result: ", result)
print("Performance measure: ",performance_measure)
avg_performance_measure += performance_measure
case_count += 1
print()


print ("Testing A clean, B clean, start in B:") 
result, performance_measure = action({"A":"clean", "B":"clean"} , "B")
print("Final result: ", result)
print("Performance measure: ",performance_measure)
avg_performance_measure += performance_measure
case_count += 1
print()

print ("Testing A clean, B dirty, start in A:") 
result, performance_measure = action({"A":"clean", "B":"dirty"} , "A")
print("Final result: ", result)
print("Performance measure: ",performance_measure)
avg_performance_measure += performance_measure
case_count += 1
print()

print ("Testing A clean, B dirty, start in B:") 
result, performance_measure = action({"A":"clean", "B":"dirty"} , "B")
print("Final result: ", result)
print("Performance measure: ",performance_measure)
avg_performance_measure += performance_measure
case_count += 1
print()

print ("Testing A dirty, B dirty, start in A:") 
result, performance_measure = action({"A":"dirty", "B":"dirty"} , "A")
print("Final result: ", result)
print("Performance measure: ",performance_measure)
avg_performance_measure += performance_measure
case_count += 1
print()

print ("Testing A dirty, B dirty, start in B:") 
result, performance_measure = action({"A":"dirty", "B":"dirty"} , "B")
print("Final result: ", result)
print("Performance measure: ",performance_measure)
avg_performance_measure += performance_measure
case_count += 1
print()

print ("Testing A dirty, B clean, start in A:") 
result, performance_measure = action({"A":"dirty", "B":"clean"} , "A")
print("Final result: ", result)
print("Performance measure: ",performance_measure)
avg_performance_measure += performance_measure
case_count += 1
print()

print ("Testing A dirty, B clean, start in B:") 
result, performance_measure = action({"A":"dirty", "B":"clean"} , "B")
print("Final result: ", result)
print("Performance measure: ",performance_measure)
avg_performance_measure += performance_measure
case_count += 1
print()

avg_performance_measure = avg_performance_measure / case_count
print("average performance measure: ", avg_performance_measure) #printing out the average performance measure