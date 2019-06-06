################################################################
def OR(x,y,epsilon):
    outage = []
    
    for i,val in enumerate(epsilon):
        if(abs(x[i]-y[i]) > val):
            outage.append(i)

    return (len(outage)/float(len(epsilon)))*100