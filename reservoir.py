import sys
import numpy as np
import network as Network

class Reservoir():
    
    def __init__(self, d = None, filter_name = None, num_nodes = None, classifier = None,
                 input_probability = None, reservoir_probability = None):
        self.d = d
        self.filter_name=filter_name
        self.classifier=classifier
        self.num_nodes=num_nodes
        self.input_probability=input_probability
        self.reservoir_probability=reservoir_probability
        self.Network=Network
        
        
    #Function that creates each reservoir separately
    def setup_reservoir(self, filter_name, classifier, num_nodes,
                 input_probability, reservoir_probability, d, Network):
        
        self.Network=Network
        self.d=d
        self.filter_name=filter_name
        self.classifier=classifier
        self.num_nodes=num_nodes
        self.input_probability=input_probability
        self.reservoir_probability=reservoir_probability
        #Setting the right data for all the possible combinations of problems and classifiers
    
        if classifier == 'lin':
        	self.d.build_train_labels_lin()
        	self.d.build_test_labels_lin()
        		
        elif classifier == 'log':
        	self.d.build_train_labels_log()
        	self.d.build_test_labels_log()
        
        else:
        	print("This classifier is not supported for this test.")
        	sys.exit(1)
        
        self.d.build_training_matrix()
        self.d.build_test_matrix()
        self.Network.L = 5
        
        #Filtering the data
        if filter_name not in self.d.spectral_bands.keys():
        	print("The specified frequency band is not supported")
        	sys.exit(1)
        
        self.d.training_data = self.d.filter_data(self.d.training_data,filter_name)
        self.d.test_data = self.d.filter_data(self.d.test_data,filter_name)

        #Computing the absolute value of the data, to get rid of negative numbers
        self.d.training_data = np.abs(self.d.training_data)
        self.d.test_data = np.abs(self.d.test_data)
        
        ########################
        # Define the network parameters
        ########################
        
        self.Network.T = self.d.training_data.shape[1] #Number of training time steps
        self.Network.n_min = 2540 #Number time steps dismissed
        self.Network.K = self.d.data.shape[0] #Input layer size
        
        self.Network.u = self.d.training_data
        self.Network.y_teach = self.d.training_results
        
        
    def training_partialnetwork(self):
        self.Network.N = self.num_nodes #Reservoir layer size
        
        self.Network.setup_network(self.d,self.num_nodes,self.input_probability,self.reservoir_probability,self.d.data.shape[-1])
        
        self.Network.train_network(self.d.data.shape[-1],self.classifier,self.d.num_columns, self.d.num_trials_train, self.d.train_labels, self.Network.N) 
        
        
    def training_fullnetwork(self, rc_pm, rc_m):
        self.Network.N = rc_pm.Network.N + rc_m.Network.N #Reservoir layer size
        
        self.Network.link_adjacency_matrix(self.d,2*self.num_nodes,self.d.data.shape[-1],rc_pm, rc_m, self.reservoir_probability,self.input_probability)
        
        self.Network.train_network(self.d.data.shape[-1],self.classifier,self.d.num_columns, self.d.num_trials_train, self.d.train_labels, self.Network.N) 
        
    
    def testing_network(self):
        self.Network.mean_test_matrix = np.zeros([self.Network.N,self.d.num_trials_test,self.d.data.shape[-1]])
        
        self.Network.test_network(self.d.test_data, self.d.num_columns,self.d.num_trials_test, self.Network.N, self.d.data.shape[-1], t_autonom=self.d.test_data.shape[1])
        
        if self.classifier == 'lin':
            acc=self.d.accuracy_lin(self.Network.regressor.predict(self.Network.mean_test_matrix.T),self.d.test_labels)
            print(f'Performance using {self.classifier} : {acc}')
        
        elif self.classifier == 'log':
            acc=self.Network.regressor.score(self.Network.mean_test_matrix.T,self.d.test_labels.T)
            print(f'Performance using {self.classifier} : {acc}')
        
        elif self.classifier == '1nn':
            acc=self.Network.regressor.score(self.Network.mean_test_matrix.T,self.d.test_labels)
            print(f'Performance using {self.classifier} : {acc}')

