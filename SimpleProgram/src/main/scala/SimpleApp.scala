import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

/**
 * Created by zhang on 2015/6/14.
 */
object SimpleApp extends App{
  //def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir", "C:\\Users\\zhang\\Downloads\\hadoop")
    System.setProperty("HADOOP_USER_NAME", "root")
    val logFile = "hdfs://Naruto.ccntgrid.zju.edu:8020/user/test/sample_libsvm_data.txt" // Should be some file on your system
    val conf = new SparkConf().setMaster("local").setAppName("Simple Application")
    val sc = new SparkContext(conf)
  /*
    val logData = sc.textFile(logFile, 2).cache()
    val numAs = logData.filter(line => line.contains("a")).count()
    val numBs = logData.filter(line => line.contains("b")).count()
    println("Lines with a: %s, Lines with b: %s".format(numAs, numBs))
  */
  val data = MLUtils.loadLibSVMFile(sc, logFile)

  val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
  val training = splits(0).cache()
  val test = splits(1)

  val numIterations = 100
  val model = SVMWithSGD.train(training, numIterations)

  model.clearThreshold()

  val scoreAndLabels = test.map {
    point =>
      val score = model.predict(point.features)
      (score, point.label)
  }

  val metrics = new BinaryClassificationMetrics(scoreAndLabels)
  val ouROC = metrics.areaUnderROC()

  println("Area under ROC = " + ouROC)

  model.save(sc, "myModelPath")
  val sameModel = SVMModel.load(sc, "myModelPath")
  //}

}
