from pyspark.sql import SQLContext, Row
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.util import MLUtils
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from numpy import array

conf = SparkConf()
conf.setMaster("local")
conf.setAppName("My application")
conf.set("spark.executor.memory", "1g")
sc = SparkContext(conf = conf)	
sqlContext = SQLContext(sc)
#lines = sc.textFile("/home/base/final/whole/ydata-ysm-advertiser-bids-v1_0.txt")
#xaa represents a part of the original data
lines = sc.textFile("/home/base/final/xaa")
parts = lines.map(lambda l: l.split("	"))
adv = parts.map(lambda p: (p[0], p[1], p[2], p[3], p[4].strip()))
schemaString = "TIMESTAMP	PHRASE_ID	ACCOUNT_ID	PRICE	AUTO"
fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
schema = StructType(fields)
schemaadv = sqlContext.createDataFrame(adv, schema)

# Register the DataFrame as a table.
schemaadv.registerTempTable("adv")

# SQL can be run over DataFrames that have been registered as a table.
resPHRASE_ID = sqlContext.sql("SELECT DISTINCT PHRASE_ID FROM adv WHERE PRICE>10")
lphrase=[]
PHRASE_IDs = resPHRASE_ID.map(lambda p: p.PHRASE_ID)
for PHRASE in PHRASE_IDs.collect():
	lphrase.append(int(str(PHRASE)))
	print(PHRASE)


resTIME = sqlContext.sql("SELECT DISTINCT TIMESTAMP FROM adv ")
ltime=[]
TIMEs = resTIME.map(lambda p: p.TIMESTAMP)
for TIME in TIMEs.collect():
	ltime.append((str(TIME)))
	print(TIME)

lltime=ltime[:]
llphrase=lphrase[:]



ltime=lltime[:]
lphrase=llphrase[:]
pid=lphrase.pop()
res=[]
N=10
while not not lphrase:
	res=[]
	ltime=lltime[:]
	pid=lphrase.pop()
	cpt=0
	f = open('/home/base/final/output/out_' + str(pid)+ '.txt', 'w')
	while not not ltime and cpt<N:
		tid=ltime.pop()
		resPRICE_ID = sqlContext.sql("SELECT MAX(PRICE) as MPRICE FROM adv WHERE PHRASE_ID=" + str(pid) + " AND TIMESTAMP='" + tid + "'")
		PRICE_IDs = resPRICE_ID.map(lambda q: q.MPRICE)
		for PRICE in PRICE_IDs.collect():
			res.append( str(cpt) + ' ' + str(PRICE))
		cpt=cpt+1
	for item in res:
		f.write("%s\n" % item)
	f.close()
	break	


	

f = open('/home/base/final/output/out_42.txt', 'r')

Matrix = [[0 for x in range(2)] for x in range(N)]
first=1
for line in f:
	s=line.split()
	t1=s[0]
	t2=s[1]
	if t2!='None':
		fv=t2
		break
	print(t1 + ' * '+ t2 + ' * ')
f.close()

f = open('/home/base/final/output/out_42.txt', 'r')
Matrix = [[0 for x in range(2)] for x in range(N)]
for line in f:
	s=line.split()
	t1=s[0]
	t2=s[1]
	print(t1)
	if t2=='None':
		Matrix[int(t1)][0]=t1
		Matrix[int(t1)][1]=fv
	else:		
		Matrix[int(t1)][0]=t1
		Matrix[int(t1)][1]=t2
		fv=t2	
f.close()	


	

f = open('/home/base/final/output/out_42_v1.txt', 'w')

for i in range(0,N):
	f.write("%s" % Matrix[i][1])	
	f.write(" 1:%s\n" % Matrix[i][0])

f.close()

f = open('/home/base/final/output/out_42_v2.txt', 'w')

for i in range(0,N):	
	f.write("%s	" % Matrix[i][0])
	f.write("%s\n" % Matrix[i][1])

f.close()

f = open('/home/base/final/output/out_42_v3.data', 'w')

for i in range(0,N):	
	f.write("%s," % Matrix[i][1])	
	f.write("%s\n" % Matrix[i][0])

f.close()

	
# Load and parse the data file into an RDD of LabeledPoint.
data = MLUtils.loadLibSVMFile(sc, '/home/base/final/output/out_42_v1.txt')
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  Note: Use larger numTrees in practice.
#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                    numTrees=3, featureSubsetStrategy="auto",
                                    impurity='variance', maxDepth=4, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
print('Test Mean Squared Error = ' + str(testMSE))
print('Learned regression forest model:')
print(model.toDebugString())


#Linear Regression:
# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.replace(',', ' ').split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile("/home/base/final/output/out_42_v3.data")
parsedData = data.map(parsePoint)

# Build the model
model = LinearRegressionWithSGD.train(parsedData,iterations=20)

# Evaluate the model on training data
valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
MSE









	

  
  	

