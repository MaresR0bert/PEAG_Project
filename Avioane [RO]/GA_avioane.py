import numpy
import matplotlib.pyplot as mpplot

#GA avioane
def problema_avioane(fis, dim, pm, pr, nmax, buget, vizmin):
    # rezolvarea problemei achizitionarii avioanelor
    # I: fis - numele fisierului din care se preiau datele
    #    dim - dimenisiunea populatiei
    #    pm, pr - probabilitate de mutatie, respectiv de recombinare
    #    nmax - numarul maxim de generatii
    #    buget - bugetul companiei
    #    vizmin - minim de vizibilitate acceptat
    # E: sol - sirul de nr intregi care reprezinta cea mai buna solutie gasita
    #    v - calitatea maxima
    # Testare:
    #    import GA_avioane
    #    GA_avioane.problema_avioane("date.txt",100,0.3,0.7,200,5000,2000)
    # Rezultate in cazul urmatoarelor constrangeri COST FINAL <= BUGET SI VIZIBILITATE MEDIE < VIZMIN
    #    Solutia Optima [27, 34, 0]
    #    Calitatea maxima 4996
    #    Resurse utilizate 4740/5000

    # initializari
    date = numpy.genfromtxt(fis)
    cost = date[0].copy()                 # cost=[100, 60, 50] => costul fiecarui model (maxim 5000)
    vizibilitate = date[1].copy()       # vizibilitate=[1500, 2400, 1600] => vizibiliitatea fiecarui model (minim 2000)
    autonomie = date[2].copy()          # autonomie=[6000, 4200, 2800] => autonomia fiecarui model (maximizare)
    achizmaxim = date[3].copy()         # achizmaxim=[50,83,100] => cantitatea maxima care poate fi achizitionata din fiecare
    m = len(autonomie)
    v = []

    # generare populatie
    pop=gen_pop(dim, autonomie, cost, vizibilitate, achizmaxim, buget, vizmin)

    # bucla GA
    for i in range(nmax):
        # selectie parinti
        parinti = s_ruleta_SUS(pop)
        # recombinare
        desc = recombinare(parinti, pr, autonomie, cost, vizibilitate, buget, vizmin)
        # mutatie
        descm = mutatie(desc, pm, autonomie, cost, vizibilitate, achizmaxim, buget, vizmin)
        # selectie gen urmatoare
        pop=s_elitista(pop, descm)
        vmax = max(pop[:, m])
        i = numpy.argmax(pop[:, m])
        sol = pop[i][:m].copy().astype(int)
        v.append(vmax)
    desen_grafic(sol, v, cost, vizibilitate)
    return sol, v


def f_ob(x,autonomie):
    # Functia obiectiv problema avioane
    # I: x - individul evaluat
    #    autonomie - vectorul cu autonomia fiecarui tip de avion
    # E: c - calitate (autonomia adusa de x)
    c=numpy.round(numpy.dot(x,autonomie)/numpy.sum(x),2)
    return c


def gen_pop(dim, autonomie, cost, vizibilitate, achizmax, buget, vizmin):
    # generare populatie (numere intregi)
    # I:dim - dimensiunea populatiei
    #   autonomie - vector cu autonomia fiecarui tip de avion
    #   cost - vector cu costul fiecarui tip de avion
    #   vizibilitate - vector cu vizibilitatile fiecarui tip de avion
    #   achizmax - numarul maxim de achizitii pentru fiecare model
    #   buget - bugetul companiei pentru achizitionarea de avioane
    #   vizmin - vizibilitatea minima acceptata
    # E: pop - populatia aleatoare generata, cu calitatea fiecarui individ pe ultima coloana
    m=len(autonomie)
    pop=numpy.zeros((dim, m+1), dtype=int)
    i = 0
    while i < dim:
        x = numpy.zeros(m, dtype=int)
        for k in range(m):
            x[k] = numpy.random.randint(0, achizmax[k], 1, dtype=int)
        if numpy.dot(x, cost) <= buget and numpy.round((numpy.dot(x, vizibilitate)/numpy.sum(x)), 2) > vizmin:
            pop[i, :m] = x
            pop[i, m] = f_ob(x, autonomie)
            i = i + 1
    return pop


def m_int_ra(x, pm, a, b):
    # operatorul de mutatie resetare aleatoare pentru intregi
    # I: x - individul supus mutatiei
    #    pm - probabilitatea de mutatie
    #    a, b - capete interval de definitie
    # E: y - individ rezultat

    y = x.copy()
    m = len(x)
    for i in range(m):
        r = numpy.random.uniform(0, 1)
        if r < pm:
            y[i] = numpy.random.randint(a, (b[i])+1)
    return y


def mutatie(desc,pm, autonomie, cost, vizibilitate, achizmax, buget, vizmin):
    # operatia de mutatie a descendentilor obtinuti din recombinare
    # I: desc - matricea descendentilor
    #    pm - probabilitatea de mutatie
    #    profit - vector de profituri
    #    cost - vectorul de costuri
    #    vizibilitate - vectorul de vizibilitati
    #    achizmax - vectorul cu cantitatea maxima ce poate fi achizitionata
    #    buget - bugetul disponibil
    #    vizmin - vizibilitatea minima acceptata
    # E: descm - matricea indivizilor obtinuti

    dim, n = numpy.shape(desc)
    descm = desc.copy()
    for i in range(dim):
        acceptabil = 0
        while not acceptabil:
            y = m_int_ra(descm[i, :n - 1], pm, 0, achizmax)
            if numpy.dot(y, cost) <= buget and numpy.round((numpy.dot(y, vizibilitate)/numpy.sum(y)), 2) > vizmin:
                acceptabil=1
        descm[i, :n-1] = y
        descm[i][n-1] = f_ob(y, autonomie)
    return descm


def r_uniforma(x, y, pr):
    # operatorul de recombinare uniforma (toate pozitiile)

    # I: x, y - cromozomi care se recombina
    #    pr - probabilitatea de recombinare
    # E: a, b - descendenti obtinuti

    a = x.copy()
    b = y.copy()
    m = len(x)
    for i in range(m):
        r = numpy.random.uniform(0, 1)
        if r < pr:
            a[i] = y[i]
            b[i] = x[i]
    return a, b


def recombinare(parinti, pr, autonomie, cost, vizibilitate, buget, vizmin):
    # operatia de recombinare a parintilor selectati
    # I: parinti - parintii selectati
    #    pr - probabilitatea de recombinare
    #    autonomie - vector de autonomi
    #    cost - vectorul de costuri
    #    vizibilitate - vector de vizibilitati
    #    buget - bugetul de cumparare (costul maxim)
    #    vizmin - vizibiltatea minima acceptata
    # E: desc - descendentii obtinuti

    dim, n = numpy.shape(parinti)
    desc = numpy.zeros((dim, n), dtype=int)
    # alegere aleatoare a perechilor de parinti
    perechi=numpy.random.permutation(dim)
    for i in range(0, dim, 2):
        x=parinti[perechi[i], :n-1]
        y=parinti[perechi[i+1], :n-1]
        acceptabil = 0
        while not acceptabil:
            c1, c2=r_uniforma(x, y, pr)
            if numpy.dot(c1, cost) <= buget and numpy.dot(c2, cost) <= buget and numpy.round((numpy.dot(c1, vizibilitate)/numpy.sum(c1)), 2) > vizmin and numpy.round((numpy.dot(c2,vizibilitate)/numpy.sum(x)), 2) > vizmin:
                acceptabil = 1
        desc[i, :n-1] = c1
        desc[i][n-1] = f_ob(c1, autonomie)
        desc[i+1, :n-1] = c2
        desc[i+1][n-1] = f_ob(c2, autonomie)
    return desc


def d_FPS_ss(pop, c):
    # distributia de selectie FPS cu sigma scalare
    # I: pop - bazinul de selectie
    #    c - constanta din formula de ajustare. uzual: 2
    # E: p - vector probabilitati de selectie individuale
    #    q - vector probabilitati de selectie cumulate

    m, n = numpy.shape(pop)
    medie = numpy.mean(pop[:, n-1])
    sigma = numpy.std(pop[:, n-1])
    val = medie-c*sigma
    g = [numpy.max([0, pop[i][n-1]-val]) for i in range(m)]
    s = numpy.sum(g)
    p = g/s
    q = [numpy.sum(p[:i+1]) for i in range(m)]
    return p, q


def s_elitista(pop,desc):
    # selectia elitista a generatiei urmatoare
    # I: pop - populatia curenta
    #    desc - descendentii populatiei curente
    # E: noua - matricea descendentilor selectati

    noua = desc.copy()
    dim, n = numpy.shape(pop)
    max1 = max(pop[:, n-1])
    i=numpy.argmax(pop[:, n-1])
    max2 = max(desc[:, n-1])
    if max1 > max2:
        k = numpy.argmin(desc[:,n-1])
        noua[k, :] = pop[i, :]
    return noua


def s_ruleta_SUS(pop):
    # selectia tip ruleta multibrat

    # I: pop - bazinul de selectie
    # E: rez - populatia selectata

    m, n = numpy.shape(pop)
    p,q = d_FPS_ss(pop,2)
    rez = pop.copy()
    i = 0
    k = 0
    r = numpy.random.uniform(0,1/m)
    while k < m:
        while r <= q[i]:
            rez[k, :n]=pop[i, :n]
            r += 1/m
            k += 1
        i += 1
    return rez


def desen_grafic(sol, v, cost, vizibilitate):
    # vizualizare rezultate Rucsac01

    # I: sol - vectorul de nr intregi care defineste cate avioane din fiecare tip s-au achizitionat
    #    v   - vectorul cu cea mai buna calitate din fiecare generatie
    # E: -

    n = len(sol)
    t = len(v)
    maxim = max(v)
    print("Calitatea maxima ", maxim)
    print("Cantitatile din fiecare tip de avion ce trebuiesc achizitionate ", sol)
    print("Resurse folosite din 5000 de unitati ", numpy.dot(sol, cost))
    print("Vizibilitate medie ", numpy.dot(sol, vizibilitate)/numpy.sum(sol))

    fig = mpplot.figure()
    x = [i for i in range(t)]
    y = [v[i] for i in range(t)]
    mpplot.plot(x, y, 'bo-')
    mpplot.ylabel("Calitate")
    mpplot.xlabel("Generația")
    mpplot.title("Evoluția calității celui mai bun individ din fiecare generație")

    fig.show()