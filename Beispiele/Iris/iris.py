import numpy as np

mapping = {'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2 } 
inv_mapping = {v: k for k, v in mapping.items()}

cmap = {'Iris-setosa' : 'r', 'Iris-versicolor' : 'g', 'Iris-virginica' : 'b' } 

class IrisDataSet:    
    """ Wrapper-Klasse für den Iris-Datensatz. """
    def __init__(self,csv_file):  
        ''' Liest das Iris Dataset ein. '''    
        with open(csv_file, 'r') as file: 
            lines = file.read().splitlines()      
        self._rohdaten = np.array([line.split(',') for line in lines[1:]])
        werte,names = self._rohdaten[:,:-1].astype(float),self._rohdaten[:,-1]

        # Bestimme Minimal- und Maximalwerte und erzeuge einen Skalierer
        alle_werte = werte.flatten()
        min, max = alle_werte.min(), alle_werte.max()
        self.scale = lambda x : x / (max - min)
        self._struktdaten = list(map(list, (zip(self.scale(werte),[mapping[n] for n in names]))))

    def __len__(self):
        return len(self._rohdaten)
    
    def __getitem__(self,index):
        """ Liefert den Eintrag mit dem angegebenen Index. """        
        return self._rohdaten[index]
    
    def values(self):
        return self._rohdaten[:,:-1].astype(float),self._rohdaten[:,-1]
    
    def daten(self):
        return self._rohdaten[:,:-1].astype(float)
        
    def tt_daten(self,anzahl_training):
        ''' Bereitet das DataSet vor und teilt es in Training- und Testdaten auf '''
        np.random.shuffle(self._struktdaten)           # Gut mischen
        return self._struktdaten[:anzahl_training],self._struktdaten[anzahl_training:]

if __name__ == "__main__":
    print("Iris-DB")