import random 
import copy

showConsoleLog = False
log = []

def swap(start, end, cross_list, pop, i):
  if(showConsoleLog):
    print("Realizando crossover entre: ",pop[cross_list[i]][0],"  ",pop[cross_list[i+1]][0])
  string = "Realizando crossover entre: "+str(pop[cross_list[i]][0])+"  "+str(pop[cross_list[i+1]][0])
  log.append(string)

  aux1 = copy.deepcopy(pop[cross_list[i]][0][start+1:end+1])
  aux2 = copy.deepcopy(pop[cross_list[i+1]][0][start+1:end+1])

  pop[cross_list[i]][0][start+1:end+1] = copy.deepcopy(aux2)
  pop[cross_list[i+1]][0][start+1:end+1] = copy.deepcopy(aux1)
  
  string = "Resultado:                  "+str(pop[cross_list[i]][0])+"  "+str(pop[cross_list[i+1]][0])
  log.append(string)
  if(showConsoleLog):  
    print("Resultado:                  ",pop[cross_list[i]][0],"  ",pop[cross_list[i+1]][0],"\n")

def crossover(pop, cross_perc, type=1):
  cross_list = []

  for i in range(int(len(pop)*cross_perc)):
    flag = 0

    # Define quais serao os parentes 
    while flag != 1:
      randPosition = random.randint(0,len(pop)-1)
      if(randPosition not in cross_list):
        cross_list.append(randPosition)
        flag = 1

  if(len(cross_list)%2 == 1):
    cross_list.pop()
  
  # Realizacao dos crossovers para cada casal
  for i in range(0,len(cross_list),2):
    
    if(type == 1): # Crossover de um ponto
      start = -1
      end = random.randint(0,len(pop[0][0])-2)
      
      if(showConsoleLog):
        print("\n<Cossover de um ponto> (",end,")")
      string = "<Cossover de um ponto> ("+str(end)+")"
      log.append(string)
      
      # Realiza o swap dos genes
      swap(start, end, cross_list, pop, i)

    elif(type == 2): # Crossover de dois pontos
      start = random.randint(0,len(pop[0][0])-3)
      end = random.randint(start+1,len(pop[0][0])-2)

      if(showConsoleLog):
        print("\n<Cossover de dois pontos> (",start,",",end,")")
      string = "<Cossover de dois pontos> ("+str(start)+","+str(end)+")"
      log.append(string)
      
      # Realiza o swap dos genes
      swap(start, end, cross_list, pop, i)
    else:
      return None

def binSearch(roulette, value, start, end): 
  if(start == end):
    return start
  
  middle = int((start+end)/2)

  if(value >= roulette[middle] ):
    return binSearch(roulette, value, middle+1, end)
  else:
    return binSearch(roulette, value, start, middle)

def selection(pop, type=1, k = 3):
  
  if(type == 1): # Roleta
    fitSum = 0
    for i in pop:
      fitSum += i[1]

    # Criacao da roleta
    roulette = [i[1]/fitSum for i in pop]
    for i in range(1,len(roulette)):
      roulette[i] += roulette[i-1]

    # Selecao da proxima geracao usando busca binaria na roleta
    newPop = copy.deepcopy(pop)
    for i in range(int(len(pop))):
      pop[i] = copy.deepcopy(newPop[binSearch(roulette , random.random(), 0, len(roulette))])

  elif(type == 2): # Torneio
    newPop = copy.deepcopy(pop)

    for i in range(int(len(pop))):
      
      # Define os competidores
      randPositions = [ random.randint(0,len(pop)-1) for j in range(k)] 
      competitors = [pop[randPositions[j]][1] for j in range(k)]

      # Seleciona o vencedor
      for count,elem in enumerate(competitors):
        if elem == max(competitors):
          pop[i] = copy.deepcopy(newPop[randPositions[count]])
  else:
    return None

def mutation(pop, mtn_perc):
  for i in pop:
    if(random.random() <= mtn_perc):
      
      if(showConsoleLog):
        print("\nMutacao do individuo:   ",i[0])
      string = "Mutacao do individuo:   "+str(i[0])
      log.append(string)
      
      i[0][random.randint(0,len(i[0])-1)] = random.randint(0,7)
      
      if(showConsoleLog):
        print("Individuo após mutação: ",i[0],"\n")
      string = "Individuo após mutação: "+str(i[0])
      log.append(string) 

def fitness(pop):
  for idv in pop:
    error = 0
    for i in range(len(idv[0])):
      for j in range(len(idv[0])):
        # Verifica se existe uma rainha na vertical, e nas diagonais   
        if( (i != j) and (idv[0][i] == idv[0][j] or abs(i-j) == abs(idv[0][i] - idv[0][j]))):
          error += 1
    
    # 28 eh o numero maximo de erros
    # Neste caso, um fitness igual a 28 significa erro igual a zero  
    idv[1] = 28 - int(error/2) 

def maxFitness(pop):
  m = 0
  for i in pop:
    if i[1] > m:
      m = i[1]
  return m 

def printBoard(idv):
  matrix = [['' for i in range(8)] for j in range(8)]
  
  for count,elem in enumerate(matrix):
    elem[idv[count]] = 'o'
    print(elem)

# MAIN #

# Parametros
nrIdv = 100
mutation_rate = 0.5 # De 0.0 até 1.0
crossover_rate = 1 # De 0.0 até 1.0
selection_type = 2 # 1 -> Roleta, 2 -> Torneio
crossover_type = 2 # 1 -> Um ponto, 2 -> Dois pontos

# Creating population
pop = [[[random.randint(0,7) for i  in range(8)],0] for j in range(nrIdv)]

bestIdv = copy.deepcopy(pop[0])

# Critério de parada -> Maximo de 1000 iteracoes ou encontrar uma solucao sem erros
for it in range(1000): # Verificacao de parada por iteracao
  fitness(pop)

  maxIdv = copy.deepcopy(pop[0])
  for i in pop:
    if i[1] == maxFitness(pop):
      maxIdv = copy.deepcopy(i)
  
  if( maxIdv[1] > bestIdv[1] ):
    bestIdv = copy.deepcopy(maxIdv)

  selection(pop, selection_type)

  crossover(pop, crossover_rate, crossover_type)

  mutation(pop, mutation_rate)
  
  flag = 0
  for count,elem in enumerate(pop):
    if( elem[1] != maxFitness(pop) or flag ):
      if(showConsoleLog):
        print("g",it+1," i",count+1,": ",elem[0], " fitness: ",elem[1])
      string = "g"+str(it+1)+" i"+str(count+1)+": "+str(elem[0])+" fitness: "+str(elem[1])
      log.append(string)
    else:
      flag = 1
      if(showConsoleLog):
        print("g",it+1," i",count+1,": ",elem[0], " fitness: ",elem[1]," Melhor Individuo")
      string = "g"+str(it+1)+" i"+str(count+1)+": "+str(elem[0])+" fitness: "+str(elem[1])+" Melhor Individuo"
      log.append(string)
  
  if(not showConsoleLog):
    print(it+1)
  
  if( bestIdv[1] == 28 ): # Verificacao de parada por solucao
    break

print("\nMelhor Solucao: ",bestIdv[0]," Fitness: ",bestIdv[1])
string = "Melhor Solucao: "+str(bestIdv[0])+" Fitness: "+str(bestIdv[1])
log.append(string)
printBoard(bestIdv[0])

with open('8rainhasLog.txt','w') as f:
  for i in log:
    f.write("%s\n" %i)

