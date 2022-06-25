from random import randint

class Datum(object): #objeto base, contem seu id, seu grupo e todas suas caracteristicas
	def __str__(self): #funcao chamada quando e requisitado a impressao do objeto. Facilita a organizacao
		auxString = "" 
		for i in self.variables:
			auxString += "%.2f," %  i
		return (auxString + str(self.group)) 

	def __init__(self,identification,variables,group = None):
		self.variables = []
		for variable in variables:
			self.variables.append(variable)
		self.group = group
		self.id = identification

	def distanceTo(self, datum):
		total = [(self.variables[i] - datum.variables[i])**2 for i in range(len(self.variables))]
		return sum(total)**0.5

	def normalized(self,variablesMax): #retorna uma versao normalizada dos dados com base nos valores maximos encontrados
		normalizedVariables = [i/j for i,j in zip(self.variables,variablesMax)]
		return Datum(self.id, normalizedVariables, group = self.group)

	def setGroup(self, newGroup):
		self.group = newGroup

class Cluster(object): #grupo de dados que mantem a sua posicao atual e a adapta com base na media dos dados

	def __str__(self):
		auxString = ""
		for i in self.positions:
			auxString += "%s," % str(i)

		formattedData = " ["
		for i in self.data:
			formattedData += "%s |" % i
		formattedData += "|"
		return auxString + formattedData

	def __init__(self, variables):
		self.positions = variables

		self.data = []

	def addData(self,i):
		self.data.append(i)

	def clearData(self):
		self.data = []

	def distanceTo(self,datum):
		total = [(self.positions[i] - datum.variables[i])**2 for i in range(len(self.positions))]
		return sum(total)**0.5

	def isCentered(self, normalValues): #se a posicao atual for igual a media dos seus valores, retorna true
		if(len(self.data) == 0):
			return True
		sums = [0 for i in self.positions]
		for i in self.data:
			for j in range(len(i.variables)):
				sums[j] += i.normalized(normalValues).variables[j] #soma valor normalizado ja que posicao esta sempre normalizado

		for i,j in zip(self.positions, sums): #zip = [(pos1, sum1), (pos2,sum2)...]
			if(i != j/len(self.data)):
				return False

		return True

	def updatePositions(self,normalValues):
		if(len(self.data) == 0):
			return
		sums = [0 for i in self.positions]
		for i in self.data:
			for j in range(len(i.variables)):
				sums[j] += i.normalized(normalValues).variables[j]

		for i in range(len(self.positions)):
			self.positions[i] = sums[i]/len(self.data)

class Categorizer(object): #Objeto principal que guarda todos os dados e altera eles devidamente

	def __init__(self):
		self.data = []
		self.HVPV = [] #valores guardados pra normalizar os dados, HighestValuePerVariable

	def knn(self, k,datum):	

		#Cria uma lista com a distancia entre todos os dados e o dado novo a ser inserido, acompanhado do seu respectivo dado
		sortedDistances = [(i, datum.normalized(self.HVPV).distanceTo(i.normalized(self.HVPV))) for i in self.data]
		#ordena essa lista de forma crescente, ou seja, dados no inicio da lista sao os mais proximos (distancia -> 0)
		sortedDistances = sorted(sortedDistances, key=lambda x: x[1])

		groups = dict()
		#sortedDistances[:k] "corta" a lista ate a k-esima posicao
		for result in sortedDistances[:k]:
			if(result[0].group not in groups.keys()):#se for a primeira vez vendo esse grupo enquanto conta, inicia o contador como 0
				groups[result[0].group] = 0 #guarda e conta o numero de vezes que cada grupo aparece entre esses k dados
			groups[result[0].group] += 1
		
		#seleciona o grupo que aparecer mais vezes
		biggest = 0
		for i in groups:
			if groups[i] > biggest:
				biggest = groups[i]
				bestGroup = i

		datum.setGroup(bestGroup) #define o grupo do dado novo como o grupo que mais ocorreu entre os k individuos
		self.train([datum]) #salva. Imprime
		print("Amostra nova:\n" + str(datum)[:-3] + "\nClasse: {0} ({1:.2f}%)".format(datum.group,(biggest/k)*100.0))
		print("k({}) mais similares:".format(k))
		n = len(datum.variables) #maior distancia possivel entre individuos = sqrt(n)
		for i in sortedDistances[:k]:
			semelhanca = (n**0.5 - i[1]) / (n**0.5) * 100 #raiz de n - distancia SOBRE raiz n, um numero de 0 a 100
			print("{0} ({1:.2f} %).".format(str(i[0]), semelhanca))

	def updateNormalParameters(self):
		self.HVPV = [-1 for i in self.data[0].variables]
		for dado in self.data: #itera pela lista de dados salva, para cada posicao na lista, salva o maior valor encontrado
			for i in range(len(dado.variables)):

				if(self.HVPV[i] < dado.variables[i]):
					self.HVPV[i] = dado.variables[i]


	def train(self, data):
		for datum in data: #insere um novo valor e atualiza a normalizacao
			self.data.append(datum)

		self.updateNormalParameters()

	def clearData(self):
		self.data = []

	def kmeans(self,k):
		#cria-se k clusters com posicao inicial igual alguma variavel aleatoria da lista de dados atuais
		clusters = [Cluster(self.data[randint(0,len(self.data))].normalized(self.HVPV).variables) for i in range(k)]

		end_flag = False
		while(not end_flag):
			end_flag = True

			for datum in self.data: #pra cada dado na lista, verifica qual o cluster mais proximo e salva-o nele
				closest = 10000.0 #menor distancia atual

				for i in range(len(clusters)):
					distanceToCluster = clusters[i].distanceTo(datum.normalized(self.HVPV)) #distancia sempre normalizada
					if distanceToCluster < closest:
						closest = clusters[i].distanceTo(datum.normalized(self.HVPV))
						bestCluster = i #salva o cluster mais proximo

				clusters[bestCluster].addData(datum) #insere nele o dado

			for cluster in clusters:
				if(not cluster.isCentered(self.HVPV)): #se algum cluster qualquer ainda nao esta centralizado, indique que o processo ainda nao terminou
					end_flag = False

			if(end_flag): #caso tenha encerrado
				self.clearData() #limpa os dados atuais salvos
				print("kmeans finalizado para {} grupos".format(k))
				group = 1
				for cluster in clusters:
					for data in cluster.data: #para cada cluster, altera o valor do dado ali para o novo grupo
						data.setGroup(group)

					print("grupo {}".format(group))
					print([str(i) for i in cluster.data])
					if(len(cluster.data) != 0):
						self.train(cluster.data) #salva os novos grupos na lista de novo
					group += 1
			else:
				for cluster in clusters: #se o processo nao tiver terminado, limpa os clusters e atualiza suas posicoes
					cluster.updatePositions(self.HVPV)
					cluster.clearData()



DataBase = Categorizer()

with open("dados.txt", "r") as dados:
	dadosLidos = []
	for line in dados:
		parse = line.split(",") #separa a entrada por virgulas
		parse = [float(i) for i in parse] #transforma tudo em floats
		#seleciona para o id o primeiro valor, do segundo ate o ultimo como as caracteristicas e o ultimo como grupo
		novoDado = Datum(identification = parse[0],variables = parse[1:-1], group = parse[-1])
		dadosLidos.append(novoDado)

DataBase.train(dadosLidos)
datumTeste = Datum(identification = 6, variables = [1,2,3])

while(True):
	k = int(input("Insira o valor de k da proxima operacao: "))
	opcao = input("1) knn 2) kmeans: ")
	if(opcao == "1"):
		variaveis = input("insira os valores na forma 'a,b,c...,n': ")
		variaveis = variaveis.split(",")
		variaveis = [float(i) for i in variaveis]
		novoDado = Datum(identification = 0, variables = variaveis)
		DataBase.knn(k, novoDado)
	else:
		print("fazendo k-means na lista")
		DataBase.kmeans(k)