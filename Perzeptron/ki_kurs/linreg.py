type Vector = np.ndarray[float]

def koeff(x: Vector,y: Vector) -> (float,float,float,):
    """
    Berechnung der Koeffizienten des Gleichungssystems
    a1*m + b1*b = c1
    a2*m + b2*b = c2
    """
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x * x)
    sum_xy = np.sum(x * y)
    N = x.shape[0]
    return sum_xx, sum_x, sum_xy, sum_x, N, sum_y

def lin_reg(x: Vector,y: Vector):
    """
    Einfache Implementierung der linearen Regression
    """
    a1, b1, c1, a2, b2, c2 = koeff(x,y)
    # Lösung des Gleichungssystems
    b = (a2 * c1 - a1 * c2) / (a2 * b1 - a1 * b2)
    m = (c1 - b1 * b) / a1
    return (m,b)

def D(x: Vector,y: Vector,f: Callable) -> float:
    """
    Die Distanzfunktion berechnet die Fehlerquadratsumme
    """
    return np.sum((f(x) - y)**2)


if __name__ == "__main__":
    print("Funktionen zur Linearen Regression.")