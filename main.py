import reservoir as Reservoir
import network as Network
import data as Data

filter_name = 'lowmultiunit'
classifier = 'log'
num_nodes = 200
input_probability = 0.15
reservoir_probability = 0.5

#Define the class data
file = 'dataSorted_allOrientations.mat' 
d = Data.Data(80) #80% training 20% testing
d.import_data(file)

#################################
# Define the parcial reservoirs #
#################################

#Creating the premotor reservoir
premotor = Reservoir.Reservoir()
network_pm = Network.Network()
d_pm = Data.Data(80)
d_pm.import_from_matrix(d.data[0:64,:,:,:])
premotor.setup_reservoir(filter_name, classifier, num_nodes, 
                         input_probability, reservoir_probability, d_pm, network_pm)
        
#Training and testing the premotor reservoir
premotor.training_partialnetwork()
print('\nTESTING PREMOTOR RESERVOIR')
premotor.testing_network()
        
#Creating the motor reservoir
motor = Reservoir.Reservoir()
network_m = Network.Network()
d_m = Data.Data(80)
d_m.import_from_matrix(d.data[64:128,:,:,:])
motor.setup_reservoir(filter_name, classifier, num_nodes, 
                      input_probability, reservoir_probability, d_m, network_m)
        
#Training and testing the motor reservoir
motor.training_partialnetwork()
print('\nTESTING MOTOR RESERVOIR')
motor.testing_network()
        
##############################
# Define the full reservoir #
#############################
        
fullreservoir = Reservoir.Reservoir()
fullnetwork = Network.Network()
fullreservoir.setup_reservoir(filter_name, classifier, num_nodes, 
                              input_probability, reservoir_probability, d, fullnetwork)
        
#Training the fullnetwork
fullreservoir.training_fullnetwork(premotor, motor)
            
#Testing the fullnetwork
print('\nTESTING FULL RESERVOIR')
fullreservoir.testing_network()

 
            
            