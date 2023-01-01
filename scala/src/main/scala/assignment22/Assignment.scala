package assignment22

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
//import neccessary library
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, DoubleType}
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions.{asc, avg, count, desc, max, min, sum, to_date, udf}
import org.apache.spark.ml.linalg.Vector
import breeze.plot._
import breeze.linalg._
class Assignment {

  val spark: SparkSession = SparkSession.builder()
                                        .appName("assignment22")
                                        .config("spark.driver.host", "localhost")
                                        .master("local")
                                        .getOrCreate()

  //threshold for values in a and b columns
  val MAX_VALUE = 20

  // schema for 2D dataframe
  val schema2D = new StructType(Array(
    StructField("a", DoubleType, true),
    StructField("b", DoubleType, true),
    StructField("LABEL", StringType, true)
  ))

  // schema for 3D dataframe
  val schema3D = new StructType(Array(
    StructField("a", DoubleType, true),
    StructField("b", DoubleType, true),
    StructField("c", DoubleType, true),
    StructField("LABEL", StringType, true)
  ))

  //accepted label values
  val labelValues: Seq[String] = Seq("Fatal", "Ok")

  // the data frame to be used in tasks 1 and 4
  val dataD2: DataFrame = spark.read.schema(schema2D)
                                .option("delimiter", ",")
                                .option("header", "true")
                                .csv("data/dataD2.csv")


  //to handle dirty data
  val dataD2dirty: DataFrame = spark.read.schema(schema2D)
                                    .option("delimiter", ",")
                                    .option("header", "true")
                                    .csv("data/dataD2_dirty.csv")
                                    .filter(col("a").isNotNull //collect only numeric values
                                      && col("a")<MAX_VALUE    //remove very big values
                                      && col("b").isNotNull
                                      && col("b")<MAX_VALUE
                                      && col("LABEL").isin(labelValues:_*))

  // the data frame to be used in task 2
  val dataD3: DataFrame = spark.read.option("delimiter", ",").schema(schema3D).option("header", "true").csv("data/dataD3.csv")

  //to handle dirty data
  //val dataD3: DataFrame = dataD3.filter(dataD3("a").isNotNull && dataD3("b").isNotNull && dataD3("c").isNotNull&& dataD3("LABEL").isin(labelValues:_*))


  // the data frame to be used in task 3 (based on dataD2 but containing numeric labels)
  val dataD2WithLabels: DataFrame = labelToNum(dataD2)

  // dataframe with dirtyD2 labels converted to numbers
  val dataD2dirtyWithLabels: DataFrame = labelToNum(dataD2dirty)

  //convert label columns to numbers
  def labelToNum(df: DataFrame): DataFrame={

    val indexer: StringIndexer = new StringIndexer().setInputCol("LABEL").setOutputCol("LABEL_NUM")

    val dfWithLabelNum = indexer.fit(df).transform(df)

    dfWithLabelNum
  }

  // feed 2D df to pipeline to extract transformed df
  def tranform2DDF(df: DataFrame, col1: String, col2: String): DataFrame = {

    //assemble a,b columns into vectors
    val vecAssembler: VectorAssembler = new VectorAssembler().setInputCols(Array(col1, col2)).setOutputCol("unscaled_features")

    //scale data
    val minMaxScaler: MinMaxScaler = new MinMaxScaler().setInputCol("unscaled_features").setOutputCol("features")

    //set pipeline to chain the processes above
    val transformationPipeline = new Pipeline().setStages(Array(vecAssembler,minMaxScaler))

    //obtain transformed DF
    val transformedDF = transformationPipeline.fit(df).transform(df)

    transformedDF
  }

  // feed 3D df to pipeline to extract transformed df
  def tranform3DDF(df: DataFrame, col1: String, col2: String, col3: String): DataFrame = {

    //assemble a,b, LABELS columns into vectors
    val vecAssembler: VectorAssembler = new VectorAssembler().setInputCols(Array(col1, col2, col3)).setOutputCol("unscaled_features")

    //scale data
    val minMaxScaler: MinMaxScaler = new MinMaxScaler().setInputCol("unscaled_features").setOutputCol("features")

    //set pipeline to chain the processes above
    val transformationPipeline = new Pipeline().setStages(Array(vecAssembler,minMaxScaler))

    //obtain transformed DF
    val transformedDF = transformationPipeline.fit(df).transform(df)

    transformedDF
  }

  //scales 2d cluster means to original
  def rescaler2D(df: DataFrame, scaledMeans : Array[Vector]): Array[(Double, Double)] = {

    //aggregate df to select min and max for a and b column
    val minmaxDF = df.agg(min("a"), max("a"), min("b"), max("b"))

    val minA = minmaxDF.take(1)(0).getDouble(0)

    val maxA = minmaxDF.take(1)(0).getDouble(1)

    val minB = minmaxDF.take(1)(0).getDouble(2)

    val maxB = minmaxDF.take(1)(0).getDouble(3)

    val originalMeans = scaledMeans.map(vec => ((vec(0)*(maxA -minA) + minA), (vec(1)* (maxB - minB) + minB)))

    originalMeans

  }

  //scales 3d cluster means to original
  def rescaler3D(df: DataFrame, scaledMeans : Array[Vector]): Array[(Double, Double, Double)] = {

    val minmaxDF = df.agg(min("a"), max("a"), min("b"), max("b"), min("c"), max("c"))

    //extract min and max val from a b c columns
    val minA = minmaxDF.take(1)(0).getDouble(0)

    val maxA = minmaxDF.take(1)(0).getDouble(1)

    val minB = minmaxDF.take(1)(0).getDouble(2)

    val maxB = minmaxDF.take(1)(0).getDouble(3)

    val minC = minmaxDF.take(1)(0).getDouble(4)

    val maxC = minmaxDF.take(1)(0).getDouble(5)

    //scale the mean to original
    val originalMeans = scaledMeans.map(vec => ((vec(0)*(maxA -minA) + minA), (vec(1)* (maxB -minB) + minB), (vec(2)*(maxC -minC) + minC)))

    originalMeans

  }

  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {

    val transformedDF = tranform2DDF(df, "a", "b")

    val kmeans = new KMeans().setK(k).setSeed(1L)

    val kmModel = kmeans.fit(transformedDF)

    val scaledMeans = kmModel.clusterCenters

    val originalMeans = rescaler2D(df,scaledMeans)

    originalMeans
  }

  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {

    val transformedDF = tranform3DDF(df, "a", "b", "c")

    val kmeans = new KMeans().setK(k).setSeed(1L)

    val kmModel = kmeans.fit(transformedDF)

    val means = kmModel.clusterCenters

    val originalMeans = rescaler3D(df,means)

    originalMeans
  }

  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {

    val transformedDF = tranform3DDF(df, "a", "b", "LABEL_NUM")

    val kmeans = new KMeans().setK(k).setSeed(1L)

    val kmModel = kmeans.fit(transformedDF)

    //group dataframe based on predicition and label column
    val predictionDF = kmModel.transform(transformedDF).groupBy("prediction", "LABEL").count()

    val sortFatalDF = predictionDF.where(predictionDF("LABEL")==="Fatal").sort(col("count").desc)

    //index  for cluster with most fatal
    val clust1Index = sortFatalDF.take(1)(0).getInt(0)

    //index  for cluster with second most fatal
    val clust2Index = sortFatalDF.take(2)(1).getInt(0)

    // select cluster means associated to col a and b
    val scaledMeans = kmModel.clusterCenters

    val originalMeans = rescaler2D(df, scaledMeans).toList

    //mean for cluster with most fatal
    val cluster1 = originalMeans(clust1Index)

    //mean for cluster with second most fatal
    val cluster2 = originalMeans(clust2Index)

    //concat the means into an array
    val largestClusters = Array(cluster1, cluster2)

    largestClusters
  }

  //compute silhouette score for a given k value
  def computeSilscore(df : DataFrame, k: Int) : Double = {

    val kmeans = new KMeans().setK(k).setSeed(1L)

    val kmModel = kmeans.fit(df)

    val predictions = kmModel.transform(df)

    val evaluator  = new ClusteringEvaluator()

    val silhouetteScore = evaluator.evaluate(predictions)

    silhouetteScore
  }

  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {

    val transformDF = tranform2DDF(df, "a", "b")

    val kvals = Array.range(low, high+1)

    val silScores : Array[(Int, Double)] = kvals.map(k => (k, computeSilscore(transformDF,k)))

    //convert array of k values and silhouette scores to densevector
    val kvec = DenseVector(kvals.map(value => value.toDouble))

    val scoresvec = DenseVector(silScores.map(pair=> pair._2))

    //plot the grapg of silhouette scores against k values
    val f = Figure()

    val p = f.subplot(0)

    p += plot(kvec, scoresvec)

    p.xlabel = "k"

    p.ylabel =  "Silhouette score"

    p.title = "Silhouette score vs k"

    silScores

  }

}


