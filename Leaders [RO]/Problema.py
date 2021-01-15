import numpy as np

def indieni(fis,dim,pr,pm,nmax):
    vmax = 0  #cea mai buna valoare a calitatii
    V = []    #vector cu calitati

    #import Problema as p
    #apel p.indieni("date.txt",10,0.7,0.1,10)
    # rezultat (array([3, 0, 2, 1, 7, 5, 4, 6]), [6, 6, 6, 7, 7, 8])

    conflict = np.genfromtxt(fis, dtype=int)
    n, q = np.shape(conflict)
    # generare populatie initiala
    pop = gen_pop_perm(dim, n, conflict)
    counter = 0
    # bucla algoritm genetic
    while (counter < nmax) and (vmax < n):
        #selectie parinti
        parinti = s_ruleta_SUS(pop)
        #recombinare
        desc = recombinare(parinti, pr, conflict)
        #mutatie
        descm = mutatie(desc, pm, conflict)
        #selectie gen noua
        pop = s_elitista(pop, descm)
        counter = counter + 1
        vmax = max(pop[:, n])
        i = np.argmax(pop[:, n])
        sol = pop[i][:n]
        V.append(vmax)
        #sol - cea mai buna asezare
    return sol, V



def gen_pop_perm(dim, n, confilicte):
    # generare populatie de permutari

    # I: dim - dimensiune populatie (nr. de indivizi)
    #    n - dimensiune individ (numar de gene)
    #    conflict - matricea conflictelor
    # E: pop - populatia aleatoare generata, cu calitatea fiecarui individ pe ultima coloana

    pop = np.zeros((dim, n + 1), dtype=int)
    for i in range(dim):
        pop[i, :n] = np.random.permutation(n)
        pop[i, n] = f_ob(pop[i, :n], confilicte)
    return pop


def f_ob(x, conflicte):
    #I:
    # x - individul evaluat (permutarea)
    # conflicte - matricea conflictelor
    #E: c - calitatea individului evaluat

    n = len(x)
    c = n

    for i in range(n - 1):
        if conflicte[x[i]][x[i + 1]] == 1:
            c = c - 1
    if conflicte[x[0]][x[n - 1]] == 1:
        c = c - 1
    return c



def d_FPS_ss(pop,c):
    # distributia de selectie FPS cu sigma scalare

    # I: pop - bazinul de selectie
    #    c - constanta din formula de ajustare. uzual: 2
    # E: p - vector probabilitati de selectie individuale
    #    q - vector probabilitati de selectie cumulate

    m,n=np.shape(pop)
    medie=np.mean(pop[:,n-1])
    sigma=np.std(pop[:,n-1])
    val=medie-c*sigma
    g=[np.max([0, pop[i][n-1]-val]) for i in range(m)]
    s=np.sum(g)
    p=g/s
    q=[np.sum(p[:i+1]) for i in range(m)]
    return p,q


def s_ruleta_SUS(pop):
    # selectia tip ruleta multibrat

    # I: pop - bazinul de selectie
    # E: rez - populatia selectata

    m,n=np.shape(pop)
    p,q=d_FPS_ss(pop,2)
    rez=pop.copy()
    i=0
    k=0
    r=np.random.uniform(0,1/m)
    while k<m:
        while r<=q[i]:
            rez[k,:n]=pop[i,:n]
            r+=1/m
            k+=1
        i+=1
    return rez


def s_elitista(pop,desc):
    # selectia elitista a generatiei urmatoare
    # I: pop - populatia curenta
    #    desc - descendentii populatiei curente
    # E: noua - matricea descendentilor selectati

    noua=desc.copy()
    dim,n=np.shape(pop)
    max1=max(pop[:,n-1])
    i=np.argmax(pop[:,n-1])
    max2=max(desc[:,n-1])
    if max1>max2:
        k=np.argmin(desc[:,n-1])
        noua[k,:]=pop[i,:]
    return noua


def OCX(x,y,p):
    # generarea unui descendent conform Order Crossover
    # I: x, y - parinti
    #   p - vector cu cele 2 pozitii
    # E: d - descendentul creat

    m=len(x)
    d=np.zeros(m,dtype=int)-1
    d[p[0]:p[1]+1]=x[p[0]:p[1]+1]
    unde=p[1]+1
    for i in (k for j in (range(p[1],m),range(0,p[1])) for k in j):
        if not(y[i] in d):
            if unde>=m:
                unde=0
            d[unde]=y[i]
            unde+=1
    return d


def r_OCX(x,y,pr):
    # operatorul de recombinare Order Crossover pentru indivizi permutari
    # I: x, y - indivizi care se recombina (permutari)
    #    pr - probabilitatea de recombinare
    # E: a, b - descendenti obtinuti

    a=x[:]
    b=y[:]
    r=np.random.uniform(0,1)
    if r<pr:
        m=len(x)
        p=np.random.randint(0,m,2)
        while p[0]==p[1]:
            p[1]=np.random.randint(m)
        p.sort()
        a=OCX(x,y,p)
        b=OCX(y,x,p)
    return a,b


def recombinare(parinti, pr, conflict):
    # schema recombinare
    # I: parinti - parintii selectati
    #    pr - probabilitatea de recombinare
    #    m - matricea conflictelor
    # E: desc - descendentii obtinuti

    dim, n = np.shape(parinti)
    desc = np.zeros((dim, n))
    # alegere aleatoare a perechilor de parinti
    perechi = np.random.permutation(dim)
    for i in range(0, dim-1, 2):
        x = parinti[perechi[i], :n - 1]
        y = parinti[perechi[i + 1], :n - 1]
        d1, d2 = r_OCX(x, y, pr)
        desc[i, :n - 1] = d1
        desc[i][n - 1] = f_ob(d1, conflict)
        desc[i + 1, :n - 1] = d2
        desc[i + 1][n - 1] = f_ob(d2, conflict)
    return desc.astype(int)



def mutatie(desc, pm, conflict):
    # schema mutatie
    # I: desc - matricea descendentilor
    #    pm - probabilitatea de mutatie
    #    m - matricea conflictelor
    # E: descm - matricea indivizilor obtinuti

    dim, n = np.shape(desc)
    descm = desc.copy()
    for i in range(dim):
        x = descm[i, :n -1]
        y = m_perm_schimb(x, pm)
        descm[i, :n -1] = y
        descm[i][n -1] = f_ob(y, conflict)
    return descm.astype(int)


def m_perm_schimb(x,pm):
    # operatorul de mutatie prin interschimbare pentru permutari
    # I: x - individul supus mutatiei
    #    pm - probabilitatea de mutatie
    # E: y - individul rezultat

    y=x.copy()
    r=np.random.uniform(0,1)
    if r<pm:
        m = len(x)
        p = np.random.randint(0, m, 2)
        while p[0] == p[1]:
            p[1] = np.random.randint(0,m)
        p.sort()
        y[p[1]]=x[p[0]]
        y[p[0]]=x[p[1]]
    return y