import numpy as np

def init(xs,k):
    ''' Wähle zufällig k Daten '''
    return np.array(np.random.default_rng().choice(xs,k))

def zuordnung(xs,ms):
    '''Bildung der Cluster aus den Punken, abhängig vom aktuellen Mittelpunkt'''
    cs = [[] for m in ms]

    for x in xs:   
        dist = [euklid(x,m) for m in ms]
        index = dist.index(min(dist))  
        cs[index].append(x)  
     
    return np.array([np.array([x for x in c]) for c in cs],dtype='object')

def aktualisierung(cs):
    '''Berechnet den neuen Cluster-Schwerpunkts'''
    ms = [[] for _ in cs]
    for n,c in enumerate(cs):
        ms[n] = schwerpunkt(c)
    return np.array(ms)

def kmeans(xs,ms):
    '''Der k-Means-Algorithmus; Startwerte ms sind vorgegeben'''
    ms_alt = None
    #while changed(ms,ms_alt):
    for _ in range(200):
        ms_alt = ms
        cs = zuordnung(xs,ms)
        ms = aktualisierung(cs)
    return cs,ms

def schwerpunkt(xs):
    '''Berechnet den Schwerpunkt einer Menge von Vektoren'''
    sum = np.zeros(xs.shape[1])
    for x in xs:
        sum += x
    return sum / len(xs)
    
def euklid(x,y):
    '''Berechnet das Quadrat des Euklidischen Abstands zwischen zwei Punkten.'''
    return np.dot(x-y,x-y) ** 2


def changed(ms,ms_alt):
    if ms_alt is None:
        return True
    for m in ms:
        for ma in ms_alt:
            if np.array_equal(m,ma):
                continue
            return True 
    return False

def print_clusters(cs,ms,elements=False):
    '''Formatierte Ausgabe der Cluster'''
    for n,(c,m) in enumerate(zip(cs,ms)):
        print(f'Cluster {n} ({len(c)} Punkte): Mittelpunkt {m}')
        if elements is True:
            print(f'Punkte: {c}')
        

if __name__ == "__main__":
    print("k-means-Algorithmus")