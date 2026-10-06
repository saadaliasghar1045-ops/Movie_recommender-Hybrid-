import pandas as pd 
import numpy as np 

def model_fit(x_train,y_train,learning_rate):
    w = 0
    b = 0
    x = 0
    
    return w,b,x

class colab_filt_model:
    
    def fit(self,Y,n_latent,reg_lambda,num_iters,learning_rate):
        self.Y = Y
        self.n_latent = n_latent
        
        # defining arguments using the inputs 
        R = (Y > 0).astype(int)
        num_users,num_movies = Y.shape
        
        # 2. Compute error matrix (only where a rating exists: R == 1)
        
        w = np.random.randint(num_users,n_latent) # random weights for all latent features for every user 
        b = np.random.randint(0,n_latent) # every user will have one unique bias for latent features 
        x = np.random.randint(num_movies,n_latent) # every movie will have its own latent features 
        
        for i in range(num_iters):
            cost,x_grad,w_grad,b_grad = self.cofi_cost_gradient(w,b,x,Y,R,reg_lambda)
            
            # updating arguments (w,b,x)
            x -= learning_rate*x_grad
            w -= learning_rate*w_grad
            b -= learning_rate*b_grad
            
            # printing cost
            if (num_iters + 1) % 200 == 0:
                print(f"Epoch {num_iters + 1}/{num_iters} | Cost: {cost:.4f}")
        
    @staticmethod
    def cofi_cost_gradient(w,b,x,Y,R,reg_lambda):
        predictions = np.dot(w,x) + b
        # 2. Compute error matrix (only where a rating exists: R == 1)
        error = (predictions-Y)*R
    
        mean_squared_error = 0.5*np.sum(error**2) 
        reg_x = 0.5*np.sum(x**2)
        reg_w = 0.5*np.sum(w**2)
        cost = mean_squared_error + reg_lambda*reg_x + reg_lambda*reg_w
        
        # calculating gradients
        x_grad = np.dot(error, w) + reg_lambda* x
        w_grad = np.dot(error.T, x) + reg_lambda * w
        b_grad = np.sum(error, axis=0, keepdims=True)
        
        return cost, x_grad, w_grad, b_grad

model = colab_filt_model()
model.fit(None,4)
    
    
    