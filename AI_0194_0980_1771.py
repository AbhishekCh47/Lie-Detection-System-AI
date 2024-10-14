
import random
import numpy as np
import pandas as pd
import copy
import time
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

class Network(object):

    def __init__(self, sizes):
        


        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
        
        self.bias_nitem = sum(sizes[1:])
        self.weight_nitem = sum([self.weights[i].size for i in range(self.num_layers-2)])

    def feedforward(self, a):
        inner=0
        '''Return the output of the network if ``a`` is input.'''
        for b, w in zip(self.biases, self.weights):
            if(inner==0):
                a=self.relu(np.dot(w,a)+b)
                inner+=1
            else:
                
                a = self.sigmoid(np.dot(w,a)+b)
        return a

    def sigmoid(self, z):
        '''The sigmoid function.'''
        return 1.0/(1.0+np.exp(-z))
    
    def relu(self,a):
        return np.maximum(0,a)

    def score(self, X, y):



        total_score=0
        for i in range(X.shape[0]):
            predicted = self.feedforward(X[i].reshape(-1,1))
            actual = y[i].reshape(-1,1)
            total_score += np.sum(np.power(predicted-actual,2)/2)  # mean-squared error
        return total_score

    def accuracy(self, X, y):



        accuracy = 0
        for i in range(X.shape[0]):
            output = self.feedforward(X[i].reshape(-1,1))
            accuracy += int(np.argmax(output) == np.argmax(y[i]))
        return accuracy / X.shape[0] * 100

    def __str__(self):
        s = "\nBias:\n\n" + str(self.biases)
        s += "\nWeights:\n\n" + str(self.weights)
        s += "\n\n"
        return s
    
    

class NNGeneticAlgo:

    def __init__(self, n_pops, net_size, mutation_rate, crossover_rate, retain_rate,X, y):



        self.n_pops = n_pops
        self.net_size = net_size
        self.nets = [Network(self.net_size) for i in range(self.n_pops)]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.retain_rate = retain_rate
        self.X = X[:]
        self.y = y[:]
    
    def get_random_point(self, type):



        nn = self.nets[0]
        layer_index, point_index = random.randint(0, nn.num_layers-2), 0
        if type == 'weight':
            row = random.randint(0,nn.weights[layer_index].shape[0]-1)
            col = random.randint(0,nn.weights[layer_index].shape[1]-1)
            point_index = (row, col)
        elif type == 'bias':
            point_index = random.randint(0,nn.biases[layer_index].size-1)
        return (layer_index, point_index)

    def get_all_scores(self):
        return [net.score(self.X, self.y) for net in self.nets]

    def get_all_accuracy(self,X,y):
        return [net.accuracy(self.X, self.y) for net in self.nets]

    def crossover(self, father, mother):



       
        nn = copy.deepcopy(father)

       
        for _ in range(self.nets[0].bias_nitem):
            
            layer, point = self.get_random_point('bias')
           
            if random.uniform(0,1) < self.crossover_rate:
                nn.biases[layer][point] = mother.biases[layer][point]

        
        for _ in range(self.nets[0].weight_nitem):
          
            layer, point = self.get_random_point('weight')
           
            if random.uniform(0,1) < self.crossover_rate:
                nn.weights[layer][point] = mother.weights[layer][point]
        
        return nn
        
    def mutation(self, child):



        nn = copy.deepcopy(child)

       
        for _ in range(self.nets[0].bias_nitem):
           
            layer, point = self.get_random_point('bias')
            
            if random.uniform(0,1) < self.mutation_rate:
                nn.biases[layer][point] += random.uniform(-0.5, 0.5)

       
        for _ in range(self.nets[0].weight_nitem):
           
            layer, point = self.get_random_point('weight')
            
            if random.uniform(0,1) < self.mutation_rate:
                nn.weights[layer][point[0], point[1]] += random.uniform(-0.5, 0.5)

        return nn

    def evolve(self,X,y):

        
        score_list = list(zip(self.nets, self.get_all_scores()))

        
        score_list.sort(key=lambda x: x[1])

       
        score_list = [obj[0] for obj in score_list]

       
        retain_num = int(self.n_pops*self.retain_rate)
        score_list_top = score_list[:retain_num]

       
        retain_non_best = int((self.n_pops-retain_num) * self.retain_rate)
        for _ in range(random.randint(0, retain_non_best)):
            score_list_top.append(random.choice(score_list[retain_num:])) 

       
        while len(score_list_top) < self.n_pops:

            father = random.choice(score_list_top)
            mother = random.choice(score_list_top)

            if father != mother:
                new_child = self.crossover(father, mother)
                new_child = self.mutation(new_child)
                score_list_top.append(new_child)
        
       
        self.nets = score_list_top
    
def main():

    
   
    l = []
    def generateColumns(start, end):
        for i in range(start, end+1):
            l.extend([str(i)+'X', str(i)+'Y'])
        return l

    eyes = generateColumns(1, 12)

    
    import pandas as pd
    df = pd.read_csv('EYES.csv')

    
    X = df[eyes]
    y = df['truth_value']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)

    
    from sklearn.preprocessing import StandardScaler as SC
    sc = SC()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    import numpy as np
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

   
    N_POPS = 10
    NET_SIZE = [24,4,2]
    MUTATION_RATE = 0.2
    CROSSOVER_RATE = 0.4
    RETAIN_RATE = 0.4

    
    nnga = NNGeneticAlgo(N_POPS, NET_SIZE, MUTATION_RATE, CROSSOVER_RATE, RETAIN_RATE, X_train, y_train)
    start_time = time.time()
       
       
    for i in range(100):
        nnga.evolve(X_train,y_train)
        if i % 10 == 0:
            print("Current iteration : {}".format(i+1))
            print("Time taken by far : %.1f seconds" % (time.time() - start_time))
            print("Current top member's network accuracy: %.2f%%\n" % max(nnga.get_all_accuracy(X_train,y_train)))
    


    new_network = nnga.nets[0]
    new_weights=new_network.weights
    Network.weights = new_network.weights




    print(l)
    def update(x):
       
        temp=copy.deepcopy(x)
            
        for i in range(4):
            
            temp.biases[0][i] += random.uniform(-0.5, 0.5)
            for j in range(24):
             
                temp.weights[0][i][j] += random.uniform(-0.5, 0.5)
        return temp

    def energy(x):
        z = x.accuracy(X_train,y_train)
        return z


    def hillclimbing(l):
        

        current_state = copy.deepcopy(new_network)
        next_state = update(current_state)
        i=0
        final_state=copy.deepcopy(new_network)
        while(i<1000):
            i+=1
            next_state = update(current_state)
            if (energy(next_state) > energy(current_state)):
                current_state = copy.deepcopy(next_state)
            if(energy(next_state) >energy(final_state)):
                final_state = copy.deepcopy(current_state)
                
            next_state = copy.deepcopy(current_state)
        return final_state

    z = hillclimbing(l)
    res = energy(z)
    print(res)

    print(z.accuracy(X_test,y_test))

    print("Execution time in seconds = ", datetime.now() - startTime)  
        


if __name__ == "__main__":
    startTime = datetime.now()
    main()
    