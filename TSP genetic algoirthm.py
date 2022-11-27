#!/usr/bin/env python
# coding: utf-8

# In[1]:


# OPtimizing TSP with GA

# Initializing Libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import random as r
import matplotlib.pyplot as plt
import math
import pandas as pd
kgf=0
def main():
    #Initializing Hyperparameters
    plt.figure(figsize=(12,12))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    Population_number = 100
    Iterations = 500
    Elite = 20
    mutation_factor = 0.1
    crossover_factor = 0.95

    # Mutation function
    def mutation(individual):
       
        position_1 = r.randint(0, len(individual)-1)
        position_2 = r.randint(0, len(individual)-1)

        mutant = individual.copy()

       
        mutant[position_1] = individual[position_2]
        mutant[position_2] = individual[position_1]

        return mutant


    # Town plot function to plot cities
    def plotTowns():
        global kgf
        l=0
        while(l<len(towns)):
            if kgf==0:
                
                plt.subplot(2, 2, 1)
                plt.title("Randomized Path",fontsize=15)
                plt.xlabel("Distance")
                plt.ylabel("Distance")
                plt.legend(["Towns"])
                plt.plot(towns[l][1], towns[l][2], 'bo',color="red")
                plt.annotate(towns[l][0], (towns[l][1], towns[l][2]),fontsize=12)
            else:
                plt.subplot(2, 2, 2)
                plt.title("Optimized Path",fontsize=15)
                plt.xlabel("Distance")
                plt.ylabel("Distance")
                plt.legend(["Towns"])
                plt.plot(towns[l][1], towns[l][2], 'bo',color="red")
                plt.annotate(towns[l][0], (towns[l][1], towns[l][2]),fontsize=12)

            l=l+1
        kgf=kgf+1
        
  
  

    # Plotting path function to plot routes
    def plotPath(path_taken):
        global kgf
        for i in range(len(path_taken)):
            if i+1==len(path_taken):
                x_values = [path_taken[i][1], path_taken[1][1]]
                y_values = [path_taken[i][2], path_taken[1][2]]
                if kgf==1:
                    plt.subplot(2, 2, 1)
                    plt.plot(x_values, y_values,color="blue")
                else:
                    plt.subplot(2, 2, 2)
                    plt.plot(x_values, y_values,color="blue")

            else:
                x_values = [path_taken[i][1], path_taken[i+1][1]]
                y_values = [path_taken[i][2], path_taken[i+1][2]]
                if kgf==1:
                    plt.subplot(2, 2, 1)
                    plt.plot(x_values, y_values,color="blue")
                else:
                    plt.subplot(2, 2, 2)
                    plt.plot(x_values, y_values,color="blue")
                #plt.plot(x_values, y_values,color="blue")

    # Path generator (random)
    def generatePath():
        path_taken = towns
        r.shuffle(path_taken)
        return path_taken
    
    
    # Euclidian distance calculator function using formula sqrt((x1-x2)^2+ (y1-y2)^2)
    def calculateDistance(path_taken):

        distance = 0
        for i in range(0,len(path_taken)):
            if i+1==len(path_taken):
                distance += math.sqrt(((path_taken[i][1]-path_taken[0][1])**2)+((path_taken[i][2]-path_taken[0][2])**2))
            else:
                distance += math.sqrt(((path_taken[i][1]-path_taken[i+1][1])**2)+((path_taken[i][2]-path_taken[i+1][2])**2))

        return distance
    
    
    #fitness function
    def computeFitness(path_taken):
        return 1/float(calculateDistance(path_taken))
    
    #crossover function
    def crossover(ancestor1, ancestor2):
        offspring = []
        offspringP1 = []
        offspringP2 = []

        geneticA = int(r.random() * len(ancestor1))
        geneticB = int(r.random() * len(ancestor1))

        initialGene = min(geneticA, geneticB)
        finalGene = max(geneticA, geneticB)
        while(initialGene<finalGene):
       
            offspringP1.append(ancestor1[initialGene])
            initialGene=initialGene+1

        offspringP2 = [item for item in ancestor2 if item not in offspringP1]

        offspring = offspringP1 + offspringP2
        offspring
        return offspring


    #selection function

    def selectAncestors(species):
        prob_dist = []
        fit_sum = sum(species['fitness'])
        for i in species['fitness']:
            prob_dist.append(i/fit_sum)

        done = False
        p1_idx = 0
        p2_idx = 0
        while not done:
            if p1_idx==p2_idx:
                p1_idx = np.random.choice(np.arange(0, 100), p=prob_dist)
                p2_idx = np.random.choice(np.arange(0, 100), p=prob_dist)
            else:
                done = True

        ancestor1 = species.iloc[p1_idx]['results']
        ancestor2 = species.iloc[p2_idx]['results']

        return ancestor1, ancestor2

    # initializing program
    
    n_towns=input("Plaease enter number of towns ")
    xy_range = input("Please enter the range for x and y coordinates ")
    n_towns=int(n_towns)
    xy_range= int(xy_range)
    #n_towns = 25
    #xy_range = 2400
    towns = []

    for t in range(n_towns):
        x = r.randint(0, xy_range)
        y = r.randint(0, xy_range)
        towns.append((t,x, y))
    
    
    random_path = generatePath()
    #constant route to check hyperparameters
    
    #towns= [(0, 529, 1071), (1, 2396, 1224), (2, 2189, 705), (3, 779, 840), (4, 1207, 211), (5, 182, 1118), (6, 1622, 1689), (7, 2050, 869), (8, 1851, 646), (9, 2175, 1409), (10, 420, 1443), (11, 453, 1518), (12, 466, 2436), (13, 1867, 174), (14, 963, 928), (15, 2290, 2145), (16, 2144, 1755), (17, 2388, 2179), (18, 3, 1137), (19, 647, 1886), (20, 1317, 2197), (21, 315, 335), (22, 1696, 188), (23, 1292, 1643), (24, 71, 710)]   #baseline for our model

    #random_path=[(17, 2388, 2179), (22, 1696, 188), (4, 1207, 211), (20, 1317, 2197), (18, 3, 1137), (0, 529, 1071), (3, 779, 840), (2, 2189, 705), (10, 420, 1443), (11, 453, 1518), (6, 1622, 1689), (19, 647, 1886), (16, 2144, 1755), (23, 1292, 1643), (15, 2290, 2145), (24, 71, 710), (8, 1851, 646), (7, 2050, 869), (14, 963, 928), (21, 315, 335), (5, 182, 1118), (13, 1867, 174), (9, 2175, 1409), (1, 2396, 1224), (12, 466, 2436)]
    


    print('Initial path distance is: '+str(calculateDistance(random_path)))

    chosen_path=[]

    for i in random_path:
        chosen_path.append(i[0])
    chosen_path.append(chosen_path[0])

    print("Random path chosen is ")
    for i in chosen_path:
        print(i)
       


    # printing result and random path
    addfirst=[]
    addfirst=random_path
    addfirst.append(addfirst[0])

    plotTowns()
    plotPath(addfirst) 




    # GA Approach
    # Initializing population
   
    species = pd.DataFrame({'results':[], 'fitness': []})
    for _ in range(Population_number):
       
        species = species.append({'results': random_path, 'fitness': computeFitness(random_path)}, ignore_index=True)

    species = species.sort_values('fitness', ascending=False)



    lis=[]

    check = species.iloc[0]['results']
    print('Initial route distance is: '+str(calculateDistance(check)))
    lis.append(calculateDistance(check))
   

    # loop for iterations
    for _ in range(Iterations):
        evolution_species = []
        
        #perform elitist method, selection, crossover and mutation
        for i in range(Elite):
            
            evolution_species.append(species.iloc[i]['results'])
           
            
        
        while len(evolution_species)<Population_number:
             
            
            
            ancestor1, ancestor2 = selectAncestors(species)
            
            #crossover and mutation
          
            if r.random() <= crossover_factor:
                evolution_species.append(crossover(ancestor1, ancestor2))

            if r.random() <= mutation_factor:

                evolution_species= mutation(evolution_species)



        #redifiining species      
        species['results']=evolution_species
        for index, row in species.iterrows():
            species.at[index, 'fitness'] = computeFitness(row['results'])

        species = species.sort_values('fitness', ascending=False)
        newsol = species.iloc[0]['results']
        hff=calculateDistance(newsol)
        lis.append(hff)


    species = species.sort_values('fitness', ascending=False)   
    result = species.iloc[0]['results']


    print('Optimized path distance is: '+str(calculateDistance(result)))
  
  
    # adding first city to the list and printing optimized route
    #print(result)
    optimal=[]
    #print("i is")
    for i in result:
        optimal.append(i[0])
    optimal.append(optimal[0])

    print("Optimal Path visited is")
  
    for i in optimal:
     
        print(i)
    

   
    new_list=[]
    new_list=result
    new_list.append(new_list[0])

    #plotting the optimized graph and progress
    plotTowns()
    plotPath(result)
    plt.subplot(2, 2, 3)
    plt.title("Progress",fontsize=15)
    plt.xlabel("Iterations")
    plt.ylabel("Distance")
    plt.plot(lis, color= "royalblue")
    plt.show()
if __name__ == "__main__":
    main()

