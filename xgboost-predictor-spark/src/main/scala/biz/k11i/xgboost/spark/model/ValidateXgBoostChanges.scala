package biz.k11i.xgboost.spark.model

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

/**
  * Created by k0a0079 on 8/22/17.
  */
object ValidateXgBoostChanges {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("ValidateXgBoostChanges")
      .enableHiveSupport()
      .config("spark.sql.warehouse.dir", "/user/k0a0079/spark-warehouse")
      .getOrCreate()

    /*
    inputPath: features path
    modelPath1: /user/k0a0079/fbtest/FB_Test2_Desktop_Next3Days_XGBoost_java-1
    modelPath2: /user/k0a0079/fbtest/FB_Test2_Desktop_Next3Days_XGBoost_java-1/stages/5_XGBoostClassificationModel_5dcbde332c93/data
     */

    val (inputPath, outputPath) = (args(0), args(1))
    val (modelPath1, modelPath2) = (args(2), args(3))
    val (outputPath1, outputPath2) = (args(4), args(5))
    val inputDF = spark.sqlContext.read.parquet(inputPath)
    val DesktopFilterCndn = """is_mobile==false and (device_type=='mac' or device_type=='windows')"""
    val SameSessionConvCndn = """n_orders>0"""
    val ItemBrowseCndn = "n_item_views + n_item_views_prev_session > 0"


    val pipelineModel = PipelineModel.load(modelPath1)
    val binaryClassifier = XGBoostBinaryClassification.load(modelPath2)

    val dfTarget = inputDF.filter("not(" + SameSessionConvCndn +") and " + ItemBrowseCndn)
    val desktopActivity = dfTarget.filter(DesktopFilterCndn).limit(1000)
    val predDF = pipelineModel.transform(desktopActivity)
    val preDFCached = predDF.persist(StorageLevel.MEMORY_ONLY)

    val dfWithTransform = binaryClassifier.transform(preDFCached)
//    val dfWithTransformImpl = binaryClassifier.transformImpl(preDFCached)

    import spark.implicits._
    val outputDF1 = dfWithTransform.withColumnRenamed("probability", "probabilities").map { row =>
      val vtc = row.getAs[String]("vtc")
      val score = row.getAs[Vector]("probabilities")(1)
      (vtc, score)
    }.toDF("vtc", "score").rdd.map(row => List(row(0).toString, row(1).toString).mkString("\t"))

//    val outputDF2 = dfWithTransformImpl.withColumnRenamed("probability", "probabilities").map { row =>
//      val vtc = row.getAs[String]("vtc")
//      val score = row.getAs[Vector]("probabilities")(1)
//      (vtc, score)
//    }.toDF("vtc", "score").rdd.map(row => List(row(0).toString, row(1).toString).mkString("\t"))

    outputDF1 coalesce 1 saveAsTextFile outputPath1
//    outputDF2 coalesce 1 saveAsTextFile outputPath2
  }
}
