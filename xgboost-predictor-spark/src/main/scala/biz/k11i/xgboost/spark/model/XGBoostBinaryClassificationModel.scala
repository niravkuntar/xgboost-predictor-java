package biz.k11i.xgboost.spark.model

import biz.k11i.xgboost.spark.util.FVecMLVector
import biz.k11i.xgboost.{Predictor => XGBoostPredictor}
import org.apache.spark.ml.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * XGBoost prediction model for binary classification task.
 *
 * @param uid               uid
 * @param _xgboostPredictor [[XGBoostPredictor]] instance
 */
class XGBoostBinaryClassificationModel(
  override val uid: String,
  _xgboostPredictor: XGBoostPredictor)
  extends ProbabilisticClassificationModel[Vector, XGBoostBinaryClassificationModel]
    with XGBoostPredictionModel[XGBoostBinaryClassificationModel] {

  def this(xgboostPredictor: XGBoostPredictor) = this(Identifiable.randomUID("XGBoostPredictorBinaryClassificationModel"), xgboostPredictor)

  setDefault(xgboostPredictor, _xgboostPredictor)

  override def numClasses: Int = 2

  override protected def predictRaw(features: Vector): Vector = {
    val pred = getXGBoostPredictor.predictSingle(toFVec(features), true)
    Vectors.dense(Array(-pred, pred))
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        Vectors.dense(dv.values.map(v => 1 / (1 + Math.exp(-v))).array)

      case _: SparseVector =>
        throw new Exception("rawPrediction should be DenseVector")
    }
  }

  override protected def raw2probability(rawPrediction: Vector): Vector = raw2probabilityInPlace(rawPrediction)

//  override def transformImpl(dataset: Dataset[_]): DataFrame = {
//    transformSchema(dataset.schema, logging = true)
//    if (isDefined(thresholds)) {
//      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
//        ".transform() called with non-matching numClasses and thresholds.length." +
//        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
//    }
//
//    // Output selected columns only.
//    // This is a bit complicated since it tries to avoid repeated computation.
//    var outputData = dataset
//    var numColsOutput = 0
//    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
//    if ($(rawPredictionCol).nonEmpty) {
//      val predictRawUDF = udf { (features: Vector) =>
//        val model = bcastModel.value
//        model.predictRaw(features)
//      }
//      outputData = outputData.withColumn(getRawPredictionCol, predictRawUDF(col(getFeaturesCol)))
//      numColsOutput += 1
//    }
//    if ($(probabilityCol).nonEmpty) {
//      val probUDF = if ($(rawPredictionCol).nonEmpty) {
//        udf{ (features: Vector) =>
//          val model = bcastModel.value
//          model.raw2probability(features)
//        } apply (col($(rawPredictionCol)))
//      } else {
//        val probabilityUDF = udf { (features: Vector) =>
//          val model = bcastModel.value
//          model.predictProbability(features)
//        }
//        probabilityUDF(col($(featuresCol)))
//      }
//      outputData = outputData.withColumn($(probabilityCol), probUDF)
//      numColsOutput += 1
//    }
//
//    if ($(predictionCol).nonEmpty) {
//      val predUDF = if ($(rawPredictionCol).nonEmpty) {
//        udf { (features: Vector) =>
//          val model = bcastModel.value
//          model.raw2prediction(features)
//        } apply (col($(rawPredictionCol)))
//      } else if ($(probabilityCol).nonEmpty) {
//        udf { (features: Vector) =>
//          val model = bcastModel.value
//          model.probability2prediction(features)
//        } apply (col($(probabilityCol)))
//      } else {
//        val predictUDF = udf { (features: Vector) =>
//          val model = bcastModel.value
//          model.predict(features)
//        }
//        predictUDF(col($(featuresCol)))
//      }
//      outputData = outputData.withColumn($(predictionCol), predUDF)
//      numColsOutput += 1
//    }
//
//    if (numColsOutput == 0) {
//      this.logWarning(s"$uid: ProbabilisticClassificationModel.transform() was called as NOOP" +
//        " since no output columns were set.")
//    }
//    outputData.toDF
//  }
}

object XGBoostBinaryClassification extends XGBoostPrediction[XGBoostBinaryClassificationModel] {
  override protected def newXGBoostModel(xgboostPredictor: XGBoostPredictor): XGBoostBinaryClassificationModel = {
    new XGBoostBinaryClassificationModel(xgboostPredictor)
  }
}