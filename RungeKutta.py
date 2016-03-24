from numpy import dot, zeros
from scipy import tril, triu, exp
from numpy.linalg import norm
from scipy.linalg import solve
from pylab import plot, clf, axis, legend, grid, loglog

#Perform newton's method with multiple equations
def newton_multi(F, J, x0, f, dfdy, tn, yn, dt, c, a, tol = 1e-8):
  
  #start with initial guess
  x_old = x0
  
  #Loop until break condition met
  while 1:
    #Find Jacobian
    M = J(dfdy, tn, yn, dt, c, a, x_old)
    #Find RHS(-F)
    RHS = dot(-1 , F(f, tn, yn, dt, c, a, x_old))
    
    #Solve for Sigma
    sig = solve( M, RHS )
    
    #Get next x iteration value
    x_new = x_old + sig
    
    #Get error
    res = x_new - x_old
    res_norm = norm(res)
    
    #if error is small enough, exit loop
    if res_norm < tol: break
    
    #assign next iteration value to previous value
    x_old = x_new
  
  #return root
  return x_new

#Perform Runge-Kutta method for an input butcher's table
def runge_kutte_allorder(a, b, c, f, dfdy, T, dt, y0, exact_sol_available=False, exact_sol=0):

  #initialize GTE
  gte = 0
  
  # calculate number of iterations
  n = int(T/float(dt) + 0.5)
  
  #Calculate order
  s = len(b)
  
  #initialize variables
  k_n = zeros(s)
    
  # Use initial condition:
  t_list = [0]
  y_list = [y0]
  
  # Calculate approximations for all time steps:
  for i in range(n):
    
    #find k values
    k_n = newton_multi(function, Jacobian, k_n, f, dfdy, t_list[i], y_list[i], dt, c, a)
    
    #initialize temp sum
    tempSum = 0
    
    #calculate weighted sum of kis
    for j in range(s):
      
      tempSum += b[j]*k_n[j] 
    
    #calculate y and append 
    y_n_plus_one = y_list[i] + dt * tempSum
    y_list.append(y_n_plus_one)
    
    #Calculate time and append
    t_n_plus_one = t_list[i] + dt
    t_list.append(t_n_plus_one)
    
    #Find GTE if exact solution available
    if exact_sol_available:
      err_n = abs(exact_sol(t_n_plus_one) - y_n_plus_one)
      if err_n > gte: gte = err_n
  
  # Return the solution and error:
  return[t_list,y_list, gte]

#generates F vector for a RK method, where Fi = ki-f(tn+ci*h,yn+sum(aij*ki)
def function( f, tn, yn, dt, c, a, k ):
  
  #find RK order
  s = len( c )
  
  #initialize function array
  result = zeros( s )
  
  #loop through each column of result
  for i in range( s ):
  
    #initialize temp sum
    tempSum = 0
  
    #calculate weighed sum of kjs
    for j in range( s ):
      
      tempSum += a[i][j]*k[j]
  
    #calculate each column of F
    result[i] = k[i] - f(tn + c[i]*dt, yn + dt*tempSum)

  #return F
  return result

#generates jacobian for dFj/dkis
def Jacobian( dfdy, tn, yn, dt, c, a, k ):
  
  #find RK order
  s = len( c )
  
  #initialize Jacobian
  result = zeros( [s,s] )
  
  #loop through each row of Jacobian(kis)
  for i in range( s ):
    
    #initialize temp sum
    tempSum = 0
  
    #calculate weighted sum of kjs
    for j in range( s ):
      
      tempSum += a[i][j]*k[j]
    
    #loop through columns of Jacobian(Fis)
    for j in range( s ):
      
      #calculate each element of J
      result[i][j] = kdel(i,j)-dfdy(tn + c[i]*dt, yn + dt*tempSum)*dt*a[i][j]
  
  #return J
  return result

#evaluates kronecker delta function for an i and j
def kdel( i, j ):
  
  #if i = j, result is 1
  if (i == j):
    return 1
  
  #if i != j, result is 0
  else:
    return 0

#Example butcher's tables
'''
a = [[0.,0.],[0.,1.]]

b = [0.,1.]

c=[0.0,0.5]
'''
'''
a = [[1.]]
b = [1.]
c = [0.]
'''
a = [[0.1666667,-0.3333333,0.1666667],[0.1666667,0.4166667,-0.0833333],[0.1666667,0.6666667,0.1666667]]
b = [0.1666667,0.6666666,0.1666667]
c = [0.,0.5,1.]

#input derivative functions and its derivative
def f(t,y):
  return y*((1/(t+1))-1)

def dfdy(t,y):
  return ((1/(t+1))-1)

#final time
T = 5.

#time step(h)
dt = 0.1

#inital condition
y0 = 1.

#set exact solution available flag to true
exact_sol_available =True

def exact_sol(t):
  return (t+1)*exp(-t)


#run RK method
[t_list,y_list,error] = runge_kutte_allorder(a, b, c, f, dfdy, T, dt, y0, exact_sol_available, exact_sol)

#plot Result
clf()
axis('equal')
label = "Runge-Kutta Results of order"
plot(t_list, y_list, "b-", label=label)
if exact_sol_available:
  label = "Exact Solution"
  plot(t_list,[exact_sol(t) for t in t_list], "r-", label = label)
grid(True)
legend()
lab.show()


#if exact solution available, generate and plot error vs timestep
if exact_sol_available:
  
  gte=[]
  
  time_steps = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.0125]
  for dt in time_steps:
    [t_list, y_list, error] = runge_kutte_allorder(a, b, c, f, dfdy, T, dt, y0, exact_sol_available, exact_sol)
    gte.append(error)
  
  # Plot the solution:
  clf()
  label = "GTE as a function of dt"
  axis('equal')
  loglog(time_steps, gte, "b-", label=label)
  grid(True)
  legend()
  lab.show()
