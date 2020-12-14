import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Global variables
phase = "train"  # phase can be set to either "train" or "eval"

""" 
You are allowed to change the names of function arguments as per your convenience, 
but it should be meaningful.

E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
but abc, a, b, etc. are not.

"""

def get_features(file_path):
	# Given a file path , return feature matrix and target labels 
    data = pd.read_csv(file_path, parse_dates=[0], infer_datetime_format=True)
    data.pickup_datetime = data.pickup_datetime.apply(lambda x: x.hour)
    y = data.fare_amount.to_numpy()
    y = y.reshape(-1,1)

    data["pickup_datetime"]=data.pickup_datetime/data.pickup_datetime.max()
    data["pickup_longitude"]=data.pickup_longitude/data.pickup_longitude.max()
    data["pickup_latitude"]=data.pickup_latitude/data.pickup_latitude.max()
    data["dropoff_latitude"]=data.dropoff_latitude/data.dropoff_latitude.max()
    data["dropoff_longitude"]=data.dropoff_longitude/data.dropoff_longitude.max()
    data["passenger_count"]=data.passenger_count/data.passenger_count.max()
    data["pickup_datetime"]=data.pickup_datetime/data.pickup_datetime.max()
    data["pickup_datetime"]=data.pickup_datetime/data.pickup_datetime.max()
    x1 = data.pickup_longitude
    y1 = data.pickup_latitude
    x2 = data.dropoff_longitude
    y2 = data.dropoff_latitude
    data["distance"]=(((y2-y1)**2)+((x2-x1)**2))**0.5
    data["distance"]=data.distance/data.distance.max()
    data=data.drop(columns=["fare_amount","pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"]).to_numpy()
    phi=data
	
	return phi, y

def get_features_basis1(file_path):
	# Given a file path , return feature matrix and target labels 
	
	
	return phi, y

def get_features_basis2(file_path):
	# Given a file path , return feature matrix and target labels 
	
	
	return phi, y

def compute_RMSE(phi, w , y) :
	# Root Mean Squared Error
	error=(np.sum((y-np.dot(phi,w))**2)/len(y))**0.5
    
	return error

def generate_output(phi_test, w):
	# writes a file (output.csv) containing target variables in required format for Kaggle Submission.
    fields = ['Id', 'fare']
    pr = np.dot(phi_test,w)
    pd.DataFrame(pr).to_csv('output.csv')
	
def closed_soln(phi, y):
    # Function returns the solution w for Xw=y.
    return np.linalg.pinv(phi).dot(y)
	
def gradient_descent(phi, y) :
	# Mean Squared Error
    def gradient_descent(phi, y) :
    eta=5*10**(-9)
    w=np.zeros((phi.shape[1], 1))
    grad=np.sum(-2*phi*(y - np.dot(phi,w)),0).reshape(-1,1)
    w = w - (eta*(compute_RMSE(phi,w,y)**2)*grad)
    while(1):
        if(((np.sum(grad**2))**0.5)<3000):
            return w
        else:
            grad=np.sum(-2*phi*(y - np.dot(phi,w)),0).reshape(-1,1)
            w = w - (eta*(compute_RMSE(phi,w,y)**2)*grad)

def sgd(phi, y) :
	# Mean Squared Error
    eta=5*10**(-9)
    w=np.zeros((phi.shape[1], 1))
    i = np.random.randint(phi.shape[0], size = 1)
    grad=np.sum(-2*phi[i,:]*(y - np.dot(phi[i,:],w)),0).reshape(-1,1)
    w = w - (eta*(compute_RMSE(phi,w,y)**2)*grad)
    while(1):
        if(((np.sum(grad**2))**0.5)<3000):
            
            return w
        else:
            i = np.random.randint(phi.shape[0], size = 1)
            grad=np.sum(-2*phi[i,:]*(y - np.dot(phi[i,:],w)),0).reshape(-1,1)
            w = w - (eta*(compute_RMSE(phi,w,y))*grad)


def pnorm(phi, y, p) :
	# Mean Squared Error
    def gradient_descent(phi, y) :
    eta=5*10**(-9)
    w=np.zeros((phi.shape[1], 1))
    grad=np.sum(-2*phi*(y - np.dot(phi,w)),0).reshape(-1,1) + p*0.01*(np.power(w,p-1))
    w = w - (eta*(compute_RMSE(phi,w,y)**2)*grad)
    while(1):
        if(((np.sum(grad**2))**0.5)<3000):
            return w
        else:
            grad=np.sum(-2*phi*(y - np.dot(phi,w)),0).reshape(-1,1)
            w = w - (eta*(compute_RMSE(phi,w,y)**2)*grad)

	
def main():
""" 
The following steps will be run in sequence by the autograder.
"""
        ######## Task 1 #########
        phase = "train"
        phi, y = get_features('train.csv')
        w1 = closed_soln(phi, y)
        w2 = gradient_descent(phi, y)
        phase = "eval"
        phi_dev, y_dev = get_features('dev.csv')
        r1 = compute_RMSE(phi_dev, w1, y_dev)
        r2 = compute_RMSE(phi_dev, w2, y_dev)
        print('1a: ')
        print(abs(r1-r2))
        w3 = sgd(phi, y)
        r3 = compute_RMSE(phi_dev, w3, y_dev)
        print('1c: ')
        print(abs(r2-r3))

        ######## Task 2 #########
        w_p2 = pnorm(phi, y, 2)  
        w_p4 = pnorm(phi, y, 4)  
        r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
        r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
        print('2: pnorm2')
        print(r_p2)
        print('2: pnorm4')
        print(r_p4)

        ######## Task 3 #########
        phase = "train"
        phi1, y = get_features_basis1('train.csv')
        phi2, y = get_features_basis2('train.csv')
        phase = "eval"
        phi1_dev, y_dev = get_features_basis1('dev.csv')
        phi2_dev, y_dev = get_features_basis2('dev.csv')
        w_basis1 = pnorm(phi1, y, 2)  
        w_basis2 = pnorm(phi2, y, 2)  
        rmse_basis1 = compute_RMSE(phi1_dev, w_basis1, y_dev)
        rmse_basis2 = compute_RMSE(phi2_dev, w_basis2, y_dev)
        print('Task 3: basis1')
        print(rmse_basis1)
        print('Task 3: basis2')
        print(rmse_basis2)
        
main()
