import random
import numpy as np
import matplotlib.pyplot as plt  
import time

def RandSeed( m, b, samples ):
    x_coor = []
    y_coor = []
    label = []

    positive = int( samples / 2 )
    negtive = samples - positive

    for i in range( positive ):
        x = random.randint( 0, samples )
        r = random.randint( 1, samples )

        y = m * x + b - r - ( samples / 10 )

        x_coor.append(x)
        y_coor.append(y)

        if m >= 0:
            label.append( 1 )
        else:
            label.append( -1 )

    for i in range( negtive ):
        x = random.randint( 0, samples )
        r = random.randint( 1, samples )

        y = m * x + b + r + ( samples / 10 )

        x_coor.append(x)
        y_coor.append(y)

        if m >= 0:
            label.append( -1 )
        else:
            label.append( 1 )
        
    return x_coor, y_coor, label

def InitialSlope( x_coor, y_coor, label, centerOfx_coor, centerOfy_coor ):
    randomPoint = random.randint( 0, 29 )
    
    slope = ( y_coor[randomPoint] - centerOfy_coor ) / ( x_coor[randomPoint] - centerOfx_coor ) + smallValue
    
    b_constant = centerOfy_coor - slope * centerOfx_coor
    
    return slope, b_constant, randomPoint

def NormalSlope( slope, centerOfx_coor, centerOfy_coor ):
    normalSlope = -1 / slope
    b_constant = centerOfy_coor - ( normalSlope * centerOfx_coor )
    
    return normalSlope, b_constant


def PLA(x_coor, y_coor, label, Fixpoint_x_coor, Fixpoint_y_coor, centerOfx_coor, centerOfy_coor):
    slope = (Fixpoint_x_coor - centerOfy_coor) / (Fixpoint_y_coor - centerOfx_coor) + smallValue
    normalSlope = -1 / slope
    b_constant = centerOfy_coor - normalSlope * centerOfx_coor
    conditionLabel = 0
    if normalSlope * Fixpoint_x_coor + b_constant < Fixpoint_y_coor:
        conditionLabel = 1
    else:
        conditionLabel = -1
    
    misclassifiedList = []
    for i in range(len(x_coor)): 
        # Record the misclassified point in the modify part
        if conditionLabel == 1 and normalSlope * x_coor[i] + b_constant < y_coor[i] and label[i] != baselabel:
           misclassifiedList.append(i)
        elif conditionLabel == -1 and normalSlope * x_coor[i] + b_constant > y_coor[i] and label[i] != baselabel:
            misclassifiedList.append(i)
        elif conditionLabel == 1 and normalSlope * x_coor[i] + b_constant > y_coor[i] and label[i] == baselabel:
            misclassifiedList.append(i)
        elif conditionLabel == -1 and normalSlope * x_coor[i] + b_constant < y_coor[i] and label[i] == baselabel:
            misclassifiedList.append(i)

    return misclassifiedList, normalSlope

if __name__ == '__main__':    
    smallValue = 0.001
    
    m, b = 2, 2  # particular parameters m and b
    
    samples = int( input( "Please enter the number of samples: " ) )
    
    start = time.time()
    
    x = np.arange( samples )   # x = [0, 1,..., 29]
    
    y = m * x + b
    
    x_coor, y_coor, label = RandSeed( m, b, samples )
    
    plt.plot( x_coor[:len(label)//2], y_coor[:len(label)//2], 'o', color = 'red' )

    centerOfx_coor = np.mean( x_coor )
    centerOfy_coor = np.mean( y_coor )
    
    plt.plot( centerOfx_coor, centerOfy_coor, 'o', color = 'black' )

    # Initial Vector: y = slope * x 
    slope, b_constant, randomPoint = InitialSlope( x_coor, y_coor, label, centerOfx_coor, centerOfy_coor )
    
    # Normal Vector: y = normalSlope * x
    normalSlope, b_constant = NormalSlope( slope, centerOfx_coor, centerOfy_coor )
    
    x_previous_basePoint = x_coor[randomPoint]
    y_previous_basePoint = y_coor[randomPoint]
    
    baselabel = label[randomPoint]

    conditionLabel = 0
    if ( normalSlope * x_previous_basePoint ) + b_constant < y_previous_basePoint:
        conditionLabel = 1
    else:
        conditionLabel = -1
    
    misclassifiedList = []
    for i in range(len(x_coor)): 
        # Record the misclassified point in the modify part
        if conditionLabel == 1 and normalSlope * x_coor[i] + b_constant < y_coor[i] and label[i] != baselabel:
           misclassifiedList.append(i)
        elif conditionLabel == -1 and normalSlope * x_coor[i] + b_constant > y_coor[i] and label[i] != baselabel:
            misclassifiedList.append(i)
        elif conditionLabel == 1 and normalSlope * x_coor[i] + b_constant > y_coor[i] and label[i] == baselabel:
            misclassifiedList.append(i)
        elif conditionLabel == -1 and normalSlope * x_coor[i] + b_constant < y_coor[i] and label[i] == baselabel:
            misclassifiedList.append(i)

    
    plt.plot(x_previous_basePoint, y_previous_basePoint, 'o', color='black')
    
    splitSlope = 0
    iteration = 1
    
    # Adjust the ratio of W(t) and X
    beta = 0.05
    while len(misclassifiedList) != 0:
        # Randomly pick a misclassified point
        misclassifiedPoint = misclassifiedList[random.randint(0, len(misclassifiedList) - 1)]
        print("Iteration: ", iteration)
        
        if label[misclassifiedPoint] == baselabel:
            misclassToBaseSlope = (y_coor[misclassifiedPoint] - y_previous_basePoint) / (x_coor[misclassifiedPoint] - x_previous_basePoint) + smallValue
            misclassToBaseS_B_constant = y_previous_basePoint - misclassToBaseSlope * x_previous_basePoint
            Fixpoint_x_coor = beta * (x_coor[misclassifiedPoint] - x_previous_basePoint) + x_previous_basePoint
            Fixpoint_y_coor = Fixpoint_x_coor * misclassToBaseSlope + misclassToBaseS_B_constant
        else:
            x_RevMisclassifiedPoint = centerOfx_coor + centerOfx_coor - x_coor[misclassifiedPoint]
            y_RevMisclassifiedPoint = centerOfy_coor + centerOfy_coor - y_coor[misclassifiedPoint]
            misclassToBaseSlope = (y_RevMisclassifiedPoint - y_previous_basePoint) / (x_RevMisclassifiedPoint - x_previous_basePoint) + smallValue
            misclassToBaseS_B_constant = y_previous_basePoint - misclassToBaseSlope * x_previous_basePoint
            Fixpoint_x_coor = beta * (x_RevMisclassifiedPoint - x_previous_basePoint) + x_previous_basePoint
            Fixpoint_y_coor = Fixpoint_x_coor * misclassToBaseSlope + misclassToBaseS_B_constant
            
        slope = (Fixpoint_x_coor - centerOfy_coor) / (Fixpoint_y_coor - centerOfx_coor) + smallValue
        
        misclassifiedList, splitSlope = PLA(x_coor, y_coor, label, Fixpoint_x_coor, 
                                            Fixpoint_y_coor, centerOfx_coor, centerOfy_coor)
        
        x_previous_basePoint = Fixpoint_x_coor
        y_previous_basePoint = Fixpoint_y_coor
        
        iteration = iteration + 1
        
        if iteration == 10000:
            break

    if len(misclassifiedList) == 0:
        print("Halts!!!\n")
        
    end = time.time()
    
    print("Elapsed Time：%f sec" % (end - start))

    plt.plot(x_coor[:len(label)//2], y_coor[:len(label)//2], 'o', color='blue')
    plt.plot(x_coor[len(label)//2:], y_coor[len(label)//2:], 'o', color='red')
    
    b_constant = centerOfy_coor - splitSlope * centerOfx_coor
    
    y = splitSlope * x + b_constant
    
    plt.plot( x, y, color = 'orange' )
    plt.show()

























